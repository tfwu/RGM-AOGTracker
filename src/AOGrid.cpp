#include <queue>

#include "AOGrid.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilString.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilMath.hpp"
#include "util/UtilSerialization.hpp"

namespace RGM {

// ------ AOGrid::GridPrimitiveInstance ------

AOGrid::GridPrimitiveInstance::GridPrimitiveInstance(const GridPrimitive & bbox,
                                                     int idx) {
    setBbox(bbox);
    dictIdx_ = idx;
}

void AOGrid::GridPrimitiveInstance::init() {
    setX(0);
    setY(0);
    setWidth(0);
    setHeight(0);
    dictIdx_ = -1;
}

void AOGrid::GridPrimitiveInstance::setBbox(const GridPrimitive & bbox) {
    setX(bbox.x());
    setY(bbox.y());
    setWidth(bbox.width());
    setHeight(bbox.height());
}

template<class Archive>
void AOGrid::GridPrimitiveInstance::serialize(Archive & ar,
                                              const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Rectangle);
    ar & BOOST_SERIALIZATION_NVP(dictIdx_);
}

INSTANTIATE_BOOST_SERIALIZATION(AOGrid::GridPrimitiveInstance);



// ------ AOGrid::Vertex ------
void AOGrid::Vertex::init() {
    type_ = UNKNOWN_NODE;
    split_ = UNKNOWN_SPLIT;
    idx_.fill(-1);
    childIDs_.clear();
    parentIDs_.clear();
    numConfigurations_ = 0;

    pscores_.clear();
    pdxdy_.clear();
    nscores_.clear();
    ndxdy_.clear();
    goodness_ = 0;
}

void AOGrid::Vertex::computeGoodness(int method,
                                     const vector<Scalar> & totalScores) {
    switch(method) {
        case 0: {
            Scalar thr = 0;
            goodness_ = MathUtil_<Scalar>::calcErr(pscores_, nscores_, thr);
            break;
        }
        case 1: {
            Scalar thr = 0;
            //        vector<Scalar> diff(pscores_.size(), 0);
            //        std::transform(totalScores.begin(), totalScores.end(), pscores_.begin(),
            //                       diff.begin(), std::minus<Scalar>());

            //        goodness_ = MathUtil_<Scalar>::calcVar(diff, thr);
            goodness_ = MathUtil_<Scalar>::calcVar(pscores_, thr);
            break;
        }
        default: {
            std::cerr << "Wrong method specified" << std::endl;
            break;
        }
    }
}

const vector<Scalar> & AOGrid::Vertex::scores(AOGrid & g, bool isPos) const {
    if(type_ == OR_NODE &&
            (isPos ? (pscores_.size() == 0) : (nscores_.size() == 0))) {
        return g.nodeSet()[childIDs_[idx_[ID_BEST_CHILD]]].scores(g, isPos);
    } else {
        return isPos ? (pscores_) : (nscores_);
    }
}

vector<Scalar> & AOGrid::Vertex::getScores(AOGrid & g, bool isPos) {
    if(type_ == OR_NODE &&
            (isPos ? (pscores_.size() == 0) : (nscores_.size() == 0))) {
        int idx = childIDs_[idx_[ID_BEST_CHILD]];
        return g.getNodeSet()[idx].getScores(g, isPos);
    } else {
        return isPos ? (pscores_) : (nscores_);
    }
}

template<class Archive>
void AOGrid::Vertex::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(split_);
    ar & BOOST_SERIALIZATION_NVP(idx_);
    ar & BOOST_SERIALIZATION_NVP(childIDs_);
    ar & BOOST_SERIALIZATION_NVP(parentIDs_);
    ar & BOOST_SERIALIZATION_NVP(numConfigurations_);
}

INSTANTIATE_BOOST_SERIALIZATION(AOGrid::Vertex);


// ------ AOGrid ------

void AOGrid::create(const AOGridParam & param) {
    param_ = param;

    param_.cellWidth_ = param_.inputWindow_.width()   / param_.gridWidth_;
    param_.cellHeight_ = param_.inputWindow_.height() / param_.gridHeight_;

    param_.cellWidthLast_  = param_.cellWidth_ +
                             (param_.inputWindow_.width() -
                              param_.gridWidth_ * param_.cellWidth_);
    param_.cellHeightLast_ = param_.cellHeight_ +
                             (param_.inputWindow_.height() -
                              param_.gridHeight_ * param_.cellHeight_);

    build();
}

void AOGrid::clear() {
    nodeSet_.clear();
    rootTermNodeId_ = -1;
    numAndNodes_ = 0;
    numOrNodes_ = 0;
    numTermNodes_ = 0;

    DFSqueue_.clear();
    BFSqueue_.clear();

    dict_.clear();
    instanceSet_.clear();

    numConfigurations_ = 0;
}

void AOGrid::build() {
    clear();

    // The whole grid
    dict_.push_back(GridPrimitive(param_.gridWidth_, param_.gridHeight_));

    // its instance
    GridPrimitive bbox(0, 0, param_.gridWidth_, param_.gridHeight_);
    instanceSet_.push_back(GridPrimitiveInstance(bbox, 0));

    // The root Or-node
    Vertex node;
    node.type_ = OR_NODE;
    node.idx_(Vertex::ID_IN_SUBSET) = 0;
    node.idx_(Vertex::ID_IN_INSTANCE_SET) = 0;

    addNode(node);

    // Using BFS to build the AOGrid
    std::queue<int> BFS;
    BFS.push(0);

    int step     = -1;
    int numSplit = -1;
    vector<int> childIDs;

    float minSize = static_cast<float>(param_.minSize_);

    GridPrimitiveInstance instance;

    while(BFS.size() > 0) {
        // Pop out the head node from the queue
        int curID = BFS.front();
        const Vertex  curNode = nodeSet_[curID];
        const GridPrimitiveInstance  curInstance =
            instanceSet_[curNode.idx_(Vertex::ID_IN_INSTANCE_SET)];
        int curWd = curInstance.width();
        int curHt = curInstance.height();
        int curArea = curInstance.area();

        // Shrink the queue
        BFS.pop();

        // Get the children for curNode
        childIDs.clear();

        if(curNode.type_ == OR_NODE) {
            // 1) Add a terminal node
            if(param_.allowGridTermNode_ || curID > 0) {
                node.init();
                node.type_                          = T_NODE;
                node.idx_(Vertex::ID_IN_SUBSET)       = numTermNodes_;
                node.idx_(Vertex::ID_IN_INSTANCE_SET) =
                    curNode.idx_(Vertex::ID_IN_INSTANCE_SET);

                addNode(node);

                childIDs.push_back(node.idx_(Vertex::ID));

                if(curID == 0) {
                    rootTermNodeId_ = node.idx_(Vertex::ID);
                }
            }

            // 2) Add all And-nodes for horizontal and vertical splits
            bool doSplit = param_.controlSideLength_ ?
                           (curWd >= minSize && curHt >= minSize) :
                           (curArea >= minSize);

            if(!doSplit) {
                continue;
            }

            // 2.1) Split horizontally
            step = param_.controlSideLength_ ?
                   minSize : (curWd >= minSize ? 1 : ceil(minSize / curWd));

            for(int topHt = step; topHt <= curHt - step; ++topHt) {
                int bottomHt = curHt - topHt;

                numSplit = param_.allowOverlap_ ?
                           (1 + floor(topHt * param_.ratio_)) : 1;

                for(int b = 0; b < numSplit; ++b, ++bottomHt) {
                    node.init();
                    node.type_                          = AND_NODE;
                    node.idx_(Vertex::ID_IN_SUBSET)       = numAndNodes_;
                    node.idx_(Vertex::ID_IN_INSTANCE_SET) =
                        curNode.idx_(Vertex::ID_IN_INSTANCE_SET);
                    node.split_                         = HOR_SPLIT;
                    node.idx_(Vertex::SPLIT_STEP1)        = topHt;
                    node.idx_(Vertex::SPLIT_STEP2)        = curHt - bottomHt;

                    if(addNode(node)) {
                        BFS.push(node.idx_(Vertex::ID));
                    }

                    childIDs.push_back(node.idx_(Vertex::ID));

                } // for b

                if(!param_.allowGap_) {
                    continue;
                }

                bottomHt = curHt - topHt;

                numSplit = bottomHt - step - 1;
                --bottomHt;

                for(int b = 0; b < numSplit; ++b, --bottomHt) {
                    node.init();
                    node.type_                          = AND_NODE;
                    node.idx_(Vertex::ID_IN_SUBSET)       = numAndNodes_;
                    node.idx_(Vertex::ID_IN_INSTANCE_SET) =
                        curNode.idx_(Vertex::ID_IN_INSTANCE_SET);
                    node.split_                         = HOR_SPLIT;
                    node.idx_(Vertex::SPLIT_STEP1)        = topHt;
                    node.idx_(Vertex::SPLIT_STEP2)        = curHt - bottomHt;

                    if(addNode(node)) {
                        BFS.push(node.idx_(Vertex::ID));
                    }

                    childIDs.push_back(node.idx_(Vertex::ID));
                } // for b

            }// for topHt

            // 2.2) Split vertically
            step = param_.controlSideLength_ ?
                   minSize : (curHt >= minSize ? 1 : ceil(minSize / curHt));

            for(int leftWd = step; leftWd <= curWd - step; ++leftWd) {
                int rightWd = curWd - leftWd;

                numSplit = param_.allowOverlap_ ?
                           (1 + floor(leftWd * param_.ratio_)) : 1;

                for(int r = 0; r < numSplit; ++r, ++rightWd) {
                    node.init();
                    node.type_                          = AND_NODE;
                    node.idx_(Vertex::ID_IN_SUBSET)       = numAndNodes_;
                    node.idx_(Vertex::ID_IN_INSTANCE_SET) =
                        curNode.idx_(Vertex::ID_IN_INSTANCE_SET);
                    node.split_                         = VER_SPLIT;
                    node.idx_(Vertex::SPLIT_STEP1)        = leftWd;
                    node.idx_(Vertex::SPLIT_STEP2)        = curWd - rightWd;

                    if(addNode(node)) {
                        BFS.push(node.idx_(Vertex::ID));
                    }

                    childIDs.push_back(node.idx_(Vertex::ID));

                } // for r

                if(!param_.allowGap_) {
                    continue;
                }

                rightWd = curWd - leftWd;

                numSplit = rightWd - step - 1;
                --rightWd;

                for(int r = 0; r < numSplit; ++r, --rightWd) {
                    node.init();
                    node.type_                          = AND_NODE;
                    node.idx_(Vertex::ID_IN_SUBSET)       = numAndNodes_;
                    node.idx_(Vertex::ID_IN_INSTANCE_SET) =
                        curNode.idx_(Vertex::ID_IN_INSTANCE_SET);
                    node.split_                         = VER_SPLIT;
                    node.idx_(Vertex::SPLIT_STEP1)        = leftWd;
                    node.idx_(Vertex::SPLIT_STEP2)        = curWd - rightWd;

                    if(addNode(node)) {
                        BFS.push(node.idx_(Vertex::ID));
                    }

                    childIDs.push_back(node.idx_(Vertex::ID));
                } // for i

            } //for leftWd

        } else if(curNode.type_ == AND_NODE) {
            // Add its two child Or-nodes

            if(curNode.split_ == HOR_SPLIT) {
                // 1) Add the top Or-node
                bbox = GridPrimitive(curInstance.x(), curInstance.y(), curWd,
                                     curNode.idx_(Vertex::SPLIT_STEP1));

                instance.init();
                instance.setBbox(bbox);

                node.init();
                node.type_                          = OR_NODE;
                node.idx_(Vertex::ID_IN_SUBSET)       = numOrNodes_;
                node.idx_(Vertex::ID_IN_INSTANCE_SET) = findInstance(instance);

                if(addNode(node)) {
                    BFS.push(node.idx_(Vertex::ID));
                }

                childIDs.push_back(node.idx_(Vertex::ID));

                // 2) Add the bottom Or-node
                bbox = GridPrimitive(curInstance.x(), curInstance.y() +
                                     curNode.idx_(Vertex::SPLIT_STEP2),
                                     curWd, curHt -
                                     curNode.idx_(Vertex::SPLIT_STEP2));

                instance.init();
                instance.setBbox(bbox);

                node.init();
                node.type_                          = OR_NODE;
                node.idx_(Vertex::ID_IN_SUBSET)       = numOrNodes_;
                node.idx_(Vertex::ID_IN_INSTANCE_SET) = findInstance(instance);

                if(addNode(node)) {
                    BFS.push(node.idx_(Vertex::ID));
                }

                childIDs.push_back(node.idx_(Vertex::ID));

            } else if(curNode.split_ == VER_SPLIT) {
                // 1) Add the left Or-node
                bbox = GridPrimitive(curInstance.x(), curInstance.y(),
                                     curNode.idx_(Vertex::SPLIT_STEP1), curHt);

                instance.init();
                instance.setBbox(bbox);

                node.init();
                node.type_                          = OR_NODE;
                node.idx_(Vertex::ID_IN_SUBSET)       = numOrNodes_;
                node.idx_(Vertex::ID_IN_INSTANCE_SET) = findInstance(instance);

                if(addNode(node)) {
                    BFS.push(node.idx_(Vertex::ID));
                }

                childIDs.push_back(node.idx_(Vertex::ID));

                // 2) Add the right Or-node
                bbox = GridPrimitive(curInstance.x() +
                                     curNode.idx_(Vertex::SPLIT_STEP2),
                                     curInstance.y(),
                                     curWd - curNode.idx_(Vertex::SPLIT_STEP2),
                                     curHt);

                instance.init();
                instance.setBbox(bbox);

                node.init();
                node.type_                          = OR_NODE;
                node.idx_(Vertex::ID_IN_SUBSET)       = numOrNodes_;
                node.idx_(Vertex::ID_IN_INSTANCE_SET) = findInstance(instance);

                if(addNode(node)) {
                    BFS.push(node.idx_(Vertex::ID));
                }

                childIDs.push_back(node.idx_(Vertex::ID));
            }
        }

        // unique the childids
        uniqueVector_<int>(childIDs);

        nodeSet_[curID].childIDs_ = childIDs;

    }// while

    assignParentIDs();

    vector<int> visited(nodeSet_.size(), 0);
    traceDFS(0, visited);

    visited.assign(nodeSet_.size(), 0);
    traceBFS(0, visited);

    if(param_.countConfig_) {
        countConfigurations();
    }
}

void AOGrid::parse() {
    Scalar improvement = (1 - param().betaImprovement_);

        vector<std::pair<int, Scalar> > goodness;

    // using DFS to compute scores of AND nodes and OR nodes
    for(int i = 0; i < DFSqueue().size(); ++i) {
        Vertex & node(getNodeSet()[DFSqueue()[i]]);

        switch(node.type_) {
            case T_NODE: {
                node.computeGoodness(param().betaRule_, pscores());
                break;
            }
            case AND_NODE: {
                node.pscores_ = nodeSet()[node.childIDs_[0]].scores(*this, true);
                node.nscores_ = nodeSet()[node.childIDs_[0]].scores(*this, false);

                for(int c = 1; c < node.childIDs_.size(); ++c) {
                    std::transform(node.pscores_.begin(), node.pscores_.end(),
                                   nodeSet()[node.childIDs_[c]].scores(*this, true).begin(),
                                   node.pscores_.begin(), std::plus<Scalar>());

                    if(node.nscores_.size() > 0)
                        std::transform(node.nscores_.begin(), node.nscores_.end(),
                                       nodeSet()[node.childIDs_[c]].scores(*this, false).begin(),
                                       node.nscores_.begin(), std::plus<Scalar>());
                }

                node.computeGoodness(param().betaRule_, pscores());

                break;
            }
            case OR_NODE: {
                int best = -1;
                Scalar v = std::numeric_limits<Scalar>::infinity();

                for(int c = 0; c < node.childIDs_.size(); ++c) {
                    if(node.childIDs_[c] == rootTermNodeId_)
                        continue;

                    Scalar tmp = nodeSet()[node.childIDs_[c]].goodness_;
                    if(nodeSet()[node.childIDs_[c]].type_ == T_NODE &&
                            node.childIDs_.size() > 1) {
                        tmp *= improvement;
                    }

                    if(tmp < v) {
                        v = tmp;
                        best = c;
                    }

//                    if ( (tmp == v) && (node.idx_(Vertex::ID_IN_SUBSET) == 0) && (rand() > RAND_MAX / 2) ) {
//                        best = c;
//                    }
                }

                node.idx_(Vertex::ID_BEST_CHILD) = best;

                node.goodness_ = nodeSet()[node.childIDs_[best]].goodness_;

                break;
            }
        } // switch

                goodness.push_back(std::make_pair(node.type_, node.goodness_));

    } // for i

//        std::cout << "Goodness: " ;
//        for ( int i = 0; i < goodness.size(); ++i ) {
//            std::cout << "(" << goodness[i].first << ", " << goodness[i].second
//                      << ") ";
//        }
//        std::cout << std::endl;
}

vector<int> AOGrid::parseTreeBFS() {
    vector<int> BFS;
    BFS.push_back(BFSqueue()[0]);
    int head = 0;

    vector<std::pair<int, Scalar> > goodness;

    while(head < BFS.size()) {
        Vertex & node(getNodeSet()[BFS[head]]);
        head++;

        goodness.push_back(std::make_pair(node.type_, node.goodness_));

        switch(node.type_) {
            case AND_NODE: {
                for(int i = 0; i < node.childIDs_.size(); ++i) {
                    BFS.push_back(node.childIDs_[i]);
                }
                break;
            }
            case OR_NODE: {
                BFS.push_back(node.childIDs_[node.idx_(Vertex::ID_BEST_CHILD)]);
                break;
            }
        } // switch
    } // while

//    std::cout << "Goodness: " ;
//    for(int i = 0; i < goodness.size(); ++i) {
//        std::cout << "(" << goodness[i].first << ", " << goodness[i].second
//                  << ") ";
//    }
//    std::cout << std::endl;

    turnOnOff(false);

    for(int i = 0; i < BFS.size(); ++i) {
        getNodeSet()[BFS[i]].idx_(Vertex::ID_ON_OFF) = 1;
    }

    return BFS;
}

bool AOGrid::findEqualBestChildForRootOr(Scalar goodnessThr) {
    bool found = false;

    Vertex & root(getNodeSet()[BFSqueue_[0]]);

    int best = -1;
    Scalar minDiff = std::numeric_limits<Scalar>::infinity();

    for(int i = 0; i < root.childIDs_.size(); ++i) {
        Vertex & node(getNodeSet()[root.childIDs_[i]]);

        if(i == root.idx_(AOGrid::Vertex::ID_BEST_CHILD) ||
                root.childIDs_[i] == rootTermNodeId_) {
            node.goodness_ = Inf;
            continue;
        }
        Scalar diff = std::abs(node.goodness_ - goodnessThr);
        if(diff < 0.01F) {
            found = true;
            //root.idx_(AOGrid::Vertex::ID_BEST_CHILD) = i;
            //break;
            if ( diff < minDiff ) {
                minDiff = diff;
                best = i;
            }
        }
    }

    if ( found ) {
        root.idx_(AOGrid::Vertex::ID_BEST_CHILD) = best;
    }

    return found;
}

int AOGrid::findInstance(GridPrimitiveInstance & instance) {
    for(int i = 0; i < instanceSet_.size(); ++i) {
        if(static_cast<GridPrimitive>(instanceSet_[i]) ==
                static_cast<GridPrimitive>(instance)) {
            return i;
        }
    }

    instance.dictIdx_ = updateDict(instance);

    instanceSet_.push_back(instance);

    return instanceSet_.size() - 1;
}

bool AOGrid::isSameNode(const Vertex & node1, const Vertex & node2) {
    return (node1.type_ == node2.type_ &&
            node1.idx_(Vertex::ID_IN_INSTANCE_SET) ==
            node2.idx_(Vertex::ID_IN_INSTANCE_SET) &&
            node1.split_ == node2.split_ &&
            node1.idx_(Vertex::SPLIT_STEP1) == node2.idx_(Vertex::SPLIT_STEP1) &&
            node1.idx_(Vertex::SPLIT_STEP2) == node2.idx_(Vertex::SPLIT_STEP2));
}

bool AOGrid::addNode(Vertex & node) {
    int numNodes = nodeSet_.size();

    int idNode = nodeSet_.size();

    for(int i = 0; i < numNodes; ++i) {
        if(isSameNode(node, nodeSet_[i])) {
            node = nodeSet_[i];
            return false;
        }
    }

    node.idx_(Vertex::ID) = idNode;
    switch(node.type_) {
        case AND_NODE:
            numAndNodes_++;
            break;
        case OR_NODE:
            numOrNodes_++;
            break;
        case T_NODE:
            numTermNodes_++;
            break;
        default:
            std::cerr << " wrong node type" << std::endl;
            return false;
    }

    nodeSet_.push_back(node);

    return true;
}

int AOGrid::updateDict(GridPrimitiveInstance & instance) {
    for(int i = 0; i < dict_.size(); ++i) {
        if(dict_[i].isSameType(static_cast<GridPrimitive>(instance))) {
            return i;
        }
    }

    dict_.push_back(GridPrimitive(instance.width(), instance.height()));

    return dict_.size() - 1;
}

void AOGrid::assignParentIDs() {
    for(int i = 0; i < nodeSet_.size(); ++i) {
        Vertex & node(nodeSet_[i]);
        for(int j = 0; j < node.childIDs_.size(); ++j) {
            nodeSet_[node.childIDs_[j]].parentIDs_.push_back(
                node.idx_(Vertex::ID));
        }
    }
}

void AOGrid::traceDFS(int nodeID, vector<int> & visited) {
    if(visited[nodeID] == 1) {
        RGM_LOG(error, "Cycle detected in AOGrid!");
        return;
    }

    visited[nodeID] = 1;

    const Vertex & node = nodeSet_[nodeID];

    int numChild = node.childIDs_.size();

    for(int i = 0; i < numChild; ++i) {
        if(visited[node.childIDs_[i]] < 2) {
            traceDFS(nodeSet_[node.childIDs_[i]].idx_(Vertex::ID), visited);
        }
    }

    DFSqueue_.push_back(nodeID);
    visited[nodeID] = 2;
}

void AOGrid::traceBFS(int nodeID, vector<int> & visited) {
    if(visited[nodeID] == 1) {
        RGM_LOG(error, "Cycle detected in AOGrid!");
        return;
    }

    BFSqueue_.push_back(nodeID);

    visited[nodeID] = 1;

    const Vertex & node = nodeSet_[nodeID];

    int numChild = node.childIDs_.size();

    for(int i = 0; i < numChild; ++i) {
        if(visited[node.childIDs_[i]] < 2) {
            traceBFS(nodeSet_[node.childIDs_[i]].idx_(Vertex::ID), visited);
        }
    }

    visited[nodeID] = 2;
}

long int AOGrid::countDoubleCountings(
    vector<vector<int>::const_iterator > & combination) {
    // The first And-nodes
    int idx = *(combination[0]);
    vector<int> & childIDs(nodeSet_[idx].childIDs_);

    // Get primitive bboxes split from the two children
    GridPrimitiveInstance first  =
        instanceSet_[ nodeSet_[childIDs[0]].idx_(Vertex::ID_IN_INSTANCE_SET)];
    GridPrimitiveInstance second =
        instanceSet_[ nodeSet_[childIDs[1]].idx_(Vertex::ID_IN_INSTANCE_SET)];

    vector<GridPrimitive> bb = first.partition(second);

    for(int i = 1; i < combination.size(); ++i) {
        idx = *(combination[i]);
        vector<int> & childIDs1(nodeSet_[idx].childIDs_);

        first  = instanceSet_[
                     nodeSet_[childIDs1[0]].idx_(Vertex::ID_IN_INSTANCE_SET) ];
        second = instanceSet_[
                     nodeSet_[childIDs1[1]].idx_(Vertex::ID_IN_INSTANCE_SET) ];

        vector<Rectangle> bb1 = first.partition(second);

        bb = getOverlappedRectangles(bb, bb1);
    }

    if(bb.size() == 0) {
        return 0;
    }

    // Get double-countings
    double count = 1;

    for(int i = 0; i < bb.size(); ++i) {
        // Get bounding box label
        int idxInstance = -1;
        for(int j = 0; j < instanceSet_.size(); ++j) {
            if(bb[i] == instanceSet_[j]) {
                idxInstance = j;
                break;
            }
        }

        // count
        for(int j = 0; j < nodeSet_.size(); ++j) {
            if(nodeSet_[j].type_ == OR_NODE &&
                    nodeSet_[j].idx_(Vertex::ID_IN_INSTANCE_SET) == idxInstance) {
                count *= nodeSet_[j].numConfigurations_;
                RGM_CHECK_GT(count, 0);
                break;
            }
        }
    }

    return count;
}

long int AOGrid::getDoubleCountedConfig(int nodeID) {
    typedef vector<int>::const_iterator Fci;

    long int numConfig = 0;

    const Vertex & node(nodeSet_[nodeID]);
    if(node.type_ != OR_NODE) {
        return numConfig;
    }

    // Get indices of child And-nodes
    vector<int> AndIDs;
    for(int i = 0; i < node.childIDs_.size(); ++i) {
        int idx = node.childIDs_[i];
        if(nodeSet_[idx].type_ == AND_NODE) {
            AndIDs.push_back(idx);
        }
    }

    Fci istart(AndIDs.begin());
    Fci iend(AndIDs.end());

    // Using inclusion-exclusion
    int factor = 1;
    for(int len = 2; len <= AndIDs.size(); ++len) {
        vector<vector<Fci > > combinations(
            enumerateCombinations_<Fci >(istart, iend, len));

        for(int i = 0; i < combinations.size(); ++i) {
            vector<Fci > & curComb(combinations[i]);
            double curDblcnt = countDoubleCountings(curComb);
            numConfig += factor * curDblcnt;
        }

        factor = -factor;
    }

    return numConfig;
}

void AOGrid::countConfigurations() {
    numConfigurations_ = 0;

    if(param_.allowOverlap_ || param_.allowGap_) {
        RGM_LOG(error, "can not count the configurations for AOGrid"
                " with overlapped nodes");
        return;
    }

    for(int i = 0; i < DFSqueue_.size(); ++i) {
        int idx = DFSqueue_[i];
        Vertex & node(nodeSet_[idx]);

        switch(node.type_) {
            case OR_NODE:
                node.numConfigurations_ = 0;
                for(int j = 0; j < node.childIDs_.size(); ++j) {
                    node.numConfigurations_ += nodeSet_[
                                                   node.childIDs_[j]].numConfigurations_;
                }
                // Double-counting between its child And-nodes:
                // like inclusion-exclusion operator in prob.
                node.numConfigurations_ -= getDoubleCountedConfig(idx);
                break;
            case AND_NODE:
                node.numConfigurations_ = nodeSet_[
                                              node.childIDs_[0]].numConfigurations_;
                for(int j = 1; j < node.childIDs_.size(); ++j) {
                    node.numConfigurations_ *= nodeSet_[
                                                   node.childIDs_[j]].numConfigurations_;
                }
                break;
            case T_NODE:
                node.numConfigurations_ = 1;
                break;
        }
    }

    // The root Or-node
    numConfigurations_ = nodeSet_[0].numConfigurations_;
}

string AOGrid::visualize(string & saveDir, cv::Size sz,
                         int edgeDirection/*=0*/, int nodeLabel/*=0*/) {
    FileUtil::VerifyTheLastFileSep(saveDir);
    FileUtil::VerifyDirectoryExists(saveDir);

    string AOGDir = saveDir + "AOGrid_" +
                    NumToString_<int>(param_.gridWidth_) + "_" +
                    NumToString_<int>(param_.gridHeight_) + FILESEP;

    FileUtil::CreateDir(AOGDir);

    pictureNodes(AOGDir);
    writeGraphVizDotFile(AOGDir, sz, edgeDirection, nodeLabel);

    return AOGDir;
}

void AOGrid::pictureNodes(string & saveDir) {

    int nCol = param_.gridWidth_;
    int nRow = param_.gridWidth_;

    const int interval  = 10;
    const int lineWidth = 3;

    int intervalWd = 0, intervalHt = 0;

    if(param_.inputWindow_.width() > param_.inputWindow_.height()) {
        intervalHt = interval;
        intervalWd = interval * ((float)param_.inputWindow_.width() /
                                 param_.inputWindow_.height());
    } else {
        intervalWd = interval;
        intervalHt = interval * ((float)param_.inputWindow_.height() /
                                 param_.inputWindow_.width());
    }

    int wd = nCol * (intervalWd + lineWidth) + lineWidth;
    int ht = nRow * (intervalHt + lineWidth) + lineWidth;

    // Image for the whole grid
    cv::Mat gridImg(ht, wd, CV_8UC3, cv::Scalar::all(255));

    int xx = 0;
    for(int x = 0; x <= nCol; ++x) {
        gridImg.colRange(xx, xx + lineWidth) = cv::Scalar::all(0);
        xx += intervalWd + lineWidth;
    }

    int yy = 0;
    for(int y = 0; y <= nRow; ++y) {
        gridImg.rowRange(yy, yy + lineWidth) = cv::Scalar::all(0);
        yy += intervalHt + lineWidth;
    }

    const string extName(".png");

    string saveName = saveDir + "grid" + extName;
    cv::imwrite(saveName, gridImg);

    // Images for Term-nodes
    for(int i = 0; i < nodeSet_.size(); ++i) {
        const Vertex & node(nodeSet_[i]);
        if(node.type_ != T_NODE) {
            continue;
        }

        const GridPrimitiveInstance & instance(
            instanceSet_[node.idx_(Vertex::ID_IN_INSTANCE_SET)]);

        int y1 = instance.top()  * (intervalHt + lineWidth) + lineWidth;
        int y2 = (instance.bottom() + 1) * (intervalHt + lineWidth);
        int x1 = instance.left() * (intervalWd + lineWidth) + lineWidth;
        int x2 = (instance.right() + 1) * (intervalWd + lineWidth);

        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

        cv::Mat curImg;

        vector<cv::Mat> curImgs;
        cv::split(gridImg, curImgs);
        for(int ch = 0; ch < curImgs.size(); ++ch) {
            curImgs[ch](roi) -= 120;
        }

        cv::merge(curImgs, curImg);

        saveName = saveDir + "TNode_" + NumToString_<int>(
                       node.idx_(Vertex::ID_IN_INSTANCE_SET)) + extName;
        cv::imwrite(saveName, curImg);
    }

    // Image for And-nodes
    const int margin = 1;

    for(int i = 0; i < nodeSet_.size(); ++i) {
        const Vertex & node(nodeSet_[i]);
        if(node.type_ != AND_NODE) {
            continue;
        }

        const Vertex & ch1(nodeSet_[node.childIDs_[0]]);
        const GridPrimitiveInstance & instance1(
            instanceSet_[ch1.idx_(Vertex::ID_IN_INSTANCE_SET)]);

        int y1 = instance1.top()  * (intervalHt + lineWidth) + lineWidth;
        int y2 = (instance1.bottom() + 1) * (intervalHt + lineWidth);
        int x1 = instance1.left() * (intervalWd + lineWidth) + lineWidth;
        int x2 = (instance1.right() + 1) * (intervalWd + lineWidth);

        int ty1 = std::min<int>(y1 + margin, ht - 1);
        int ty2 = std::max<int>(y2 - margin, 0);
        int tx1 = std::min<int>(x1 + margin, wd - 1);
        int tx2 = std::max<int>(x2 - margin, 0);

        cv::Rect roi(tx1, ty1, tx2 - tx1 + 1, ty2 - ty1 + 1);

        cv::Mat curImg;

        vector<cv::Mat> curImgs;
        cv::split(gridImg, curImgs);
        for(int ch = 0; ch < curImgs.size(); ++ch) {
            curImgs[ch](roi) -= 100;
        }

        cv::merge(curImgs, curImg);


        const Vertex & ch2(nodeSet_[node.childIDs_[1]]);
        const GridPrimitiveInstance & instance2(
            instanceSet_[ch2.idx_(Vertex::ID_IN_INSTANCE_SET)]);

        int yy1 = instance2.top()  * (intervalHt + lineWidth) + lineWidth;
        int yy2 = (instance2.bottom() + 1) * (intervalHt + lineWidth);
        int xx1 = instance2.left() * (intervalWd + lineWidth) + lineWidth;
        int xx2 = (instance2.right() + 1) * (intervalWd + lineWidth);

        ty1 = std::min<int>(yy1 + margin, ht - 1);
        ty2 = std::max<int>(yy2 - margin, 0);
        tx1 = std::min<int>(xx1 + margin, wd - 1);
        tx2 = std::max<int>(xx2 - margin, 0);

        roi = cv::Rect(tx1, ty1, tx2 - tx1 + 1, ty2 - ty1 + 1);

        cv::split(curImg, curImgs);
        for(int ch = 0; ch < curImgs.size(); ++ch) {
            curImgs[ch](roi) -= 160;
        }

        cv::merge(curImgs, curImg);

        int val = 200;

        curImg.rowRange(y1, y2).col(x1).setTo(cv::Vec3b(0, 0, val));
        curImg.rowRange(y1, y2).col(x2).setTo(cv::Vec3b(0, 0, val));
        curImg.colRange(x1, x2).row(y1).setTo(cv::Vec3b(0, 0, val));
        curImg.colRange(x1, x2).row(y2).setTo(cv::Vec3b(0, 0, val));

        curImg.rowRange(yy1, yy2).col(xx1).setTo(cv::Vec3b(0, val, 0));
        curImg.rowRange(yy1, yy2).col(xx2).setTo(cv::Vec3b(0, val, 0));
        curImg.colRange(xx1, xx2).row(yy1).setTo(cv::Vec3b(0, val, 0));
        curImg.colRange(xx1, xx2).row(yy2).setTo(cv::Vec3b(0, val, 0));

        int ox1 = std::max<int>(x1, xx1);
        int ox2 = std::min<int>(x2, xx2);
        int oy1 = std::max<int>(y1, yy1);
        int oy2 = std::min<int>(y2, yy2);

        if(ox1 <= ox2 && oy1 <= oy2) {
            ty1 = std::min<int>(oy1 + margin, ht - 1);
            ty2 = std::max<int>(oy2 - margin, 0);
            tx1 = std::min<int>(ox1 + margin, wd - 1);
            tx2 = std::max<int>(ox2 - margin, 0);

            roi = cv::Rect(tx1, ty1, tx2 - tx1 + 1, ty2 - ty1 + 1);

            cv::split(curImg, curImgs);
            for(int ch = 0; ch < curImgs.size(); ++ch) {
                curImgs[ch](roi) -= 200;
            }

            cv::merge(curImgs, curImg);
        }

        saveName = saveDir + "AndNode_" + NumToString_<int>(
                       node.idx_(Vertex::ID_IN_SUBSET)) + extName;
        cv::imwrite(saveName, curImg);
    }
}

void AOGrid::writeGraphVizDotFile(string & saveDir, cv::Size sz,
                                  int edgeDirection, int nodeLabel) {
    string dotfile = saveDir + "AOGrid.dot";
    FILE * f = fopen(dotfile.c_str(), "w");
    if(f == NULL) {
        RGM_LOG(error, "can not write to file " + dotfile);
        return;
    }

    fprintf(f, "digraph AOG {\n");

    if(sz.width > 0 && sz.height > 0) {
        fprintf(f, "graph [autosize=false, size=\"%f,%f!\", resolution=100];\n",
                sz.width / 100.F, sz.height / 100.F);
    }

    const string extName(".png");
    const int zeroPaddingLen = 5;

    // Draw nodes
    for(int i = 0; i < nodeSet_.size(); ++i) {
        const Vertex & node(nodeSet_[i]);
        int ID = node.idx_(Vertex::ID);
        int instanceIdx = node.idx_(Vertex::ID_IN_INSTANCE_SET);
        string textlabel;

        switch(nodeLabel) {
            case 1:
                if(numConfigurations_ > 0) {
                    textlabel = NumToString_<double>(node.numConfigurations_);
                } else {
                    textlabel = "\"\"";
                }
                break;
            case 2:
                textlabel = NumToString_<int>(DFSqueue_[i]);
                break;
            case 3:
                textlabel = NumToString_<int>(BFSqueue_[i]);
                break;
            case 4:
                textlabel = NumToString_<Scalar>(node.goodness_);
                break;
            default:
                textlabel = "\"\"";
                break;
        }

        switch(node.type_) {
            case OR_NODE:
                if(ID == 0) {
                    fprintf(f,
                            "node%d [shape=ellipse, style=bold, color=green, label=\"%s\", image=\"%sgrid%s\"];\n",
                            ID, textlabel.c_str(), saveDir.c_str(), extName.c_str());
                } else {
                    fprintf(f,
                            "node%d [shape=ellipse, style=bold, color=green, label=\"%s\", image=\"%sTNode_%d%s\"];\n",
                            ID, textlabel.c_str(), saveDir.c_str(), instanceIdx, extName.c_str());
                }
                break;

            case AND_NODE:
                fprintf(f,
                        "node%d [shape=ellipse, style=filled, color=blue, label=\"%s\", image=\"%sAndNode_%d%s\"];\n",
                        ID, textlabel.c_str(), saveDir.c_str(), node.idx_(Vertex::ID_IN_SUBSET),
                        extName.c_str());
                break;

            case T_NODE:
                fprintf(f,
                        "node%d [shape=box, style=bold, color=red, label=\"%s\", image=\"%sTNode_%d%s\"];\n",
                        ID, textlabel.c_str(), saveDir.c_str(), instanceIdx, extName.c_str());
                break;
        }
    }

    // Draw edges
    for(int i = 0; i < nodeSet_.size(); ++i) {
        const Vertex & node(nodeSet_[i]);
        switch(node.type_) {
            case OR_NODE:
                fprintf(f, "edge [style=bold, color=green];\n");
                break;

            case AND_NODE:
                fprintf(f, "edge [style=bold, color=blue];\n");
                break;

            case T_NODE:
                fprintf(f, "edge [style=bold, color=red];\n");
                break;
        }

        int ID = node.idx_(Vertex::ID);
        if(edgeDirection == 0) {
            const vector<int> & childIDs(node.childIDs_);
            for(int j = 0; j < childIDs.size(); ++j) {
                fprintf(f, "node%d -> node%d;\n", ID, childIDs[j]);
            }
        } else {
            const vector<int> & parentIDs(node.parentIDs_);
            for(int j = 0; j < parentIDs.size(); ++j) {
                fprintf(f, "node%d -> node%d;\n", ID, parentIDs[j]);
            }
        }
    }

    fprintf(f, "}");

    fclose(f);

    /// Use GraphViz
    string cmd = "dot -Tpdf " + dotfile + " -o " + saveDir + "AOGrid.pdf";
    //std::system(cmd.c_str());
    system(cmd.c_str());

    cmd = "dot -Tpng " + dotfile + " -o " + saveDir + "AOGrid.png";
    //std::system(cmd.c_str());
    system(cmd.c_str());

}

Rectangle AOGrid::instanceBbox(int idx) const {
    //RGM_CHECK_GE(idx, 0);
    //RGM_CHECK_LT(idx, instanceSet().size());

    const GridPrimitiveInstance & inst(instanceSet()[idx]);

    int x1 = inst.x() * param().cellWidth_;
    int y1 = inst.y() * param().cellHeight_;

    int wd = 0;
    if(inst.right() + 1 == param().gridWidth_) {
        wd = (inst.width() - 1) * param().cellWidth_;
        wd += param().cellWidthLast_;
    } else {
        wd = inst.width() * param().cellWidth_;
    }

    int ht = 0;
    if(inst.bottom() + 1 == param().gridHeight_) {
        ht = (inst.height() - 1) * param().cellHeight_;
        ht += param().cellHeightLast_;
    } else {
        ht = inst.height() * param().cellHeight_;
    }

    return Rectangle(x1, y1, wd, ht);
}

void AOGrid::turnOnOff(bool isOn) {
    for(int i = 0; i < nodeSet().size(); ++i) {
        Vertex & n(getNodeSet()[i]);
        n.idx_(Vertex::ID_ON_OFF) = static_cast<int>(isOn);
    }
}

void AOGrid::save(const string & modelFile, int archiveType) {
    RGM_SAVE(modelFile, archiveType);
}

bool AOGrid::read(const string & modelFile, int archiveType) {
    RGM_READ(modelFile, archiveType);
    return true;
}

template<class Archive>
void AOGrid::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(param_);
    ar & BOOST_SERIALIZATION_NVP(nodeSet_);
    ar & BOOST_SERIALIZATION_NVP(rootTermNodeId_);
    ar & BOOST_SERIALIZATION_NVP(numAndNodes_);
    ar & BOOST_SERIALIZATION_NVP(numOrNodes_);
    ar & BOOST_SERIALIZATION_NVP(numTermNodes_);
    ar & BOOST_SERIALIZATION_NVP(DFSqueue_);
    ar & BOOST_SERIALIZATION_NVP(BFSqueue_);
    ar & BOOST_SERIALIZATION_NVP(dict_);
    ar & BOOST_SERIALIZATION_NVP(instanceSet_);
    ar & BOOST_SERIALIZATION_NVP(numConfigurations_);
}

INSTANTIATE_BOOST_SERIALIZATION(AOGrid);

} // namespace RGM
