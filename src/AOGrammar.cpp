#include <map>

#include "AOGrammar.hpp"
#include "AOGrid.hpp"
#include "parse_tree.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilString.hpp"
#include "util/UtilGeneric.hpp"

namespace RGM {

// ------ AOGrammar's Edge -------

template<int Dimension>
Edge::Edge_(const Edge & e) {
    init();

    edgeType_ = e.type();
    isLRFlip_ = e.isLRFlip();
    onOff_ = e.onOff();
    idx_ = e.idx();
}

template<int Dimension>
Edge::Edge_(edgeType t, Node * fromNode, Node * toNode) {
    RGM_CHECK_NOTEQ(t, UNKNOWN_EDGE);
    RGM_CHECK_NOTNULL(toNode);
    RGM_CHECK_NOTNULL(fromNode);
    RGM_CHECK_NOTEQ(fromNode, toNode);

    init();

    edgeType_ = t;
    fromNode_ = fromNode;
    toNode_ = toNode;
}

template<int Dimension>
void Edge::init() {
    edgeType_ = UNKNOWN_EDGE;
    fromNode_ = NULL;
    toNode_ = NULL;
    isLRFlip_ = false;
    LRMirrorEdge_ = NULL;
    onOff_ = true;
    idx_.fill(-1);
}

template<int Dimension>
void Edge::assignIdx(AOGrammar * g) {
    RGM_CHECK_NOTNULL(g);

    getIdx()(IDX_FROM) = g->idxNode(fromNode());
    getIdx()(IDX_TO) = g->idxNode(toNode());
    getIdx()(IDX_MIRROR) = g->idxEdge(lrMirrorEdge());

    RGM_CHECK_GE(idx()(IDX_FROM), 0);
    RGM_CHECK_GE(idx()(IDX_TO), 0);
}

template<int Dimension>
void Edge::assignConnections(AOGrammar * g) {
    RGM_CHECK_NOTNULL(g);

    // from Node
    getFromNode() = g->findNode(idx()(IDX_FROM));

    // to Node
    getToNode() = g->findNode(idx()(IDX_TO));

    // flip
    getLRMirrorEdge() = g->findEdge(idx()(IDX_MIRROR));
}

template<int Dimension>
template <class Archive>
void Edge::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(edgeType_);
    ar & BOOST_SERIALIZATION_NVP(isLRFlip_);
    ar & BOOST_SERIALIZATION_NVP(onOff_);
    ar & BOOST_SERIALIZATION_NVP(idx_);
}

INSTANTIATE_CLASS_(Edge_);
INSTANTIATE_BOOST_SERIALIZATION_(Edge_);




// ------ AOGrammar_Node ------

template<int Dimension>
Node::Node_(nodeType t) {
    init();
    nodeType_ = t;
}

template<int Dimension>
Node::Node_(const Node & n) {
    init();

    nodeType_ = n.type();
    isLRFlip_ = n.isLRFlip();
    detectWindow_ = n.detectWindow();
    anchor_ = n.anchor();

    idxInEdge_ = n.idxInEdge();

    idxOutEdge_ = n.idxOutEdge();

    onOff_ = n.onOff();

    idx_ = n.idx();
}

template<int Dimension>
void Node::init() {
    nodeType_ = UNKNOWN_NODE;
    isLRFlip_ = false;
    LRMirrorNode_ = NULL;
    anchor_.setZero();
    scaleprior_ = NULL;
    bias_ = NULL;
    deformation_ = NULL;
    appearance_ = NULL;
    cachedFilter_ = NULL;

    tag_ = boost::uuids::random_generator()();

    idxInEdge_.clear();
    idxOutEdge_.clear();
    onOff_ = true;
    idx_.fill(-1);
}

template<int Dimension>
Deformation::Param Node::deformationParam() const {
    if(!isLRFlip() || (lrMirrorNode() != NULL &&
                       deformation() != lrMirrorNode()->deformation())) {
        return deformation()->w();
    }

    Deformation::Param w = deformation()->w();
    w(1) *= -1.0F;

    return w;
}

template<int Dimension>
typename Node::AppParam Node::appearanceParam() const {
    if(!isLRFlip() || (lrMirrorNode() != NULL &&
                       appearance() != lrMirrorNode()->appearance())) {
        return appearance()->w();
    }

    return FeaturePyr::Flip(appearance()->w(), appearance()->type());
}

template<int Dimension>
void Node::assignIdx(AOGrammar * g) {
    RGM_CHECK_NOTNULL(g);

    int num = inEdges().size();
    getIdxInEdge().resize(num);
    for(int i = 0; i < num; ++i) {
        getIdxInEdge()[i] = g->idxEdge(inEdges()[i]);
    }

    num = outEdges().size();
    getIdxOutEdge().resize(num);
    for(int i = 0; i < num; ++i) {
        getIdxOutEdge()[i] = g->idxEdge(outEdges()[i]);
    }

    getIdx()(IDX_MIRROR) = g->idxNode(lrMirrorNode());
    getIdx()(IDX_SCALEPRIOR) = g->idxScaleprior(scaleprior());
    getIdx()(IDX_BIAS) = g->idxBias(bias());
    getIdx()(IDX_DEF) = g->idxDeformation(deformation());
    getIdx()(IDX_APP) = g->idxAppearance(appearance());
    getIdx()(IDX_MIRROR) = g->idxNode(lrMirrorNode());
    getIdx()(IDX_FILTER) = g->idxFilter(cachedFilter());    
}

template<int Dimension>
void Node::assignConnections(AOGrammar * g) {
    RGM_CHECK_NOTNULL(g);

    // Assigning inEdges
    getInEdges().resize(idxInEdge_.size(), NULL);
    for(int i = 0; i < idxInEdge_.size(); ++i) {
        getInEdges()[i] = g->findEdge(idxInEdge_[i]);
    }

    // Assigning outEdges
    getOutEdges().resize(idxOutEdge_.size(), NULL);
    for(int i = 0; i < idxOutEdge_.size(); ++i) {
        getOutEdges()[i] = g->findEdge(idxOutEdge_[i]);
    }

    // Assigning LR mirror node
    getLRMirrorNode() = g->findNode(idx()(IDX_MIRROR));

    // scale prior
    getScaleprior() = g->findScaleprior(idx()(IDX_SCALEPRIOR));

    // offset
    getBias() = g->findBias(idx()(IDX_BIAS));

    // deformation
    getDeformation() = g->findDeformation(idx()(IDX_DEF));

    // appearance
    getAppearance() = g->findAppearance(idx()(IDX_APP));

    // fft
    getCachedFilter() = g->findCachedFilter(idx()(IDX_FILTER));

}

template<int Dimension>
template<class Archive>
void Node::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(nodeType_);
    ar & BOOST_SERIALIZATION_NVP(isLRFlip_);
    ar & BOOST_SERIALIZATION_NVP(detectWindow_);
    ar & BOOST_SERIALIZATION_NVP(anchor_);
    ar & BOOST_SERIALIZATION_NVP(idxInEdge_);
    ar & BOOST_SERIALIZATION_NVP(idxOutEdge_);
    ar & BOOST_SERIALIZATION_NVP(onOff_);
    ar & BOOST_SERIALIZATION_NVP(idx_);
}

INSTANTIATE_CLASS_(Node_);
INSTANTIATE_BOOST_SERIALIZATION_(Node_);


// ------ AOGrammar ------

template<int Dimension>
AOGrammar::AOGrammar_(const string & modelFile) {
    init();
    read(modelFile);
}

template<int Dimension>
void AOGrammar::init() {
    gParam_.init();
    rootNode_ = NULL;
    maxDetectWindow_ = Rectangle(0, 0);
    minDetectWindow_ = Rectangle(0, 0);
    thresh_ = 0.0F;
    cached_ = false;    
    idxRootNode_ = -1;

    bgmu_.setZero();
    bgmu_(Dimension - 1) = 1;

}

template<int Dimension>
void AOGrammar::clear() {
    gParam_.init();

    // node set
    for(int i = 0; i < nodeSet_.size(); ++i) {
        delete nodeSet_[i];
    }
    nodeSet_.clear();

    nodeDFS_.clear();
    nodeBFS_.clear();

    compNodeDFS_.clear();
    compNodeBFS_.clear();

    // root
    rootNode_ = NULL;

    // edge set
    for(int i = 0; i < edgeSet_.size(); ++i) {
        delete edgeSet_[i];
    }
    edgeSet_.clear();

    // appearance
    for(int i = 0; i < appearanceSet_.size(); ++i) {
        delete appearanceSet_[i];
    }
    appearanceSet_.clear();

    // offset
    for(int i = 0; i < biasSet_.size(); ++i) {
        delete biasSet_[i];
    }
    biasSet_.clear();

    // deformation
    for(int i = 0; i < deformationSet_.size(); ++i) {
        delete deformationSet_[i];
    }
    deformationSet_.clear();

    // scale prior
    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        delete scalepriorSet_[i];
    }
    scalepriorSet_.clear();

    maxDetectWindow_ = Rectangle();
    minDetectWindow_ = Rectangle();

    cached_ = false;

    for(int i = 0; i < cachedFilters_.size(); ++i) {
        delete cachedFilters_[i];
    }
    cachedFilters_.clear();

    for(int i = 0; i < gridAOG_.size(); ++i) {
        if(gridAOG_[i] != NULL) {
            delete gridAOG_[i];
            gridAOG_[i] = NULL;
        }
    }
    gridAOG_.clear();

    idxRootNode_ = -1;

    bgmu_.setZero();
    bgmu_(Dimension - 1) = 1;

}


template<int Dimension>
void AOGrammar::initRoots(const vector<pair<int, int> > & rootSz) {

    Anchor anchor;
    anchor.setZero();

    // create root OR-node
    getRootNode() = addNode(OR_NODE);

    // add object subcategory AND-node and link it to root OR-node
    int numSubcategory = rootSz.size() ;
    for(int i = 0; i < numSubcategory; i++) {
        pair<Node *, Edge *> sub = addChild(getRootNode(), AND_NODE, SWITCHING);

        // bias term
        Bias bias;
        bias.reset();
        sub.first->getBias() = addBias(bias);

        // scale prior
        //if ( !isSpecific() ) {
            Scaleprior prior;
            prior.reset(false);
            sub.first->getScaleprior() = addScaleprior(prior);
        //}

        // add appearance unit
        Node * rootApp = addAppearanceUnit(rootSz[i], anchor, false, false, false);

        // set detection window
        sub.first->getDetectWindow() = rootApp->detectWindow();

        // link object subcategory AND-node to its appearance unit
        addEdge(sub.first, rootApp, COMPOSITION);

        // add LR mirror
        if(isLRFlip()) {
            Node * n = addLRMirror(sub.first, sharedLRFlip());
            RGM_CHECK_NOTNULL(n);
            Edge * e = addEdge(getRootNode(), n, sub.second->type());
            setEdgeLRFlip(sub.second, e);
        }
    }

    // max and min detection window
    int i = 0;
    int maxwd = rootSz[i].first;
    int maxht = rootSz[i].second;
    int minwd = rootSz[i].first;
    int minht = rootSz[i].second;
    for(i = 1; i < rootSz.size(); ++i) {
        maxwd = max(rootSz[i].first, maxwd);
        maxht = max(rootSz[i].second, maxht);
        minwd = min(rootSz[i].first, minwd);
        minht = min(rootSz[i].second, minht);
    }
    getMaxDetectWindow() = Rectangle(maxwd, maxht);
    getMinDetectWindow() = Rectangle(minwd, minht);

    getCachedFFTStatus() = false;

    // turn off all nodes
    setOnOff(false);

    // finalizing
    finalize(false);
}

template<int Dimension>
void AOGrammar::createDPM(int numParts, const pair<int, int> & partSz,
                          int partScale) {

    RGM_CHECK(partScale == 0 || partScale == 1, error);

    vector<Node *> sub = getSubcategoryRootAndNodes(false, false);

    for(int i = 0; i < sub.size(); ++i) {
        Node * n = sub[i];

        // Enable to learn scale prior
        if(!gParam_.isSpecific_ && (n->scaleprior() != NULL)) {
            n->getScaleprior()->reset(true);
        }

        // Get root appearance parameters
        Node * t = n;
        while(t->type() != T_NODE) {
            t = t->getOutEdges()[0]->getToNode();
        }
        AppParam rootw = t->appearanceParam();

        // Purse parts
        vector<pair<AppParam, Anchor> > parts;
        pursueParts(rootw, numParts, partSz.first, partSz.second, partScale,
                    parts);

        for(int j = 0; j < numParts; ++j) {
            Node * partOr = addAppearanceUnit(parts[j].first, parts[j].second,
                                              false, false, true);
            Edge * e = addEdge(n, partOr, COMPOSITION);

            if(isLRFlip() && sharedLRFlip()) {
                Node * partOrLR = addLRMirror(partOr, true);
                Edge * me = addEdge(n->getLRMirrorNode(), partOrLR, COMPOSITION);
                setEdgeLRFlip(e, me);

                int x = parts[j].second(0);
                int y = parts[j].second(1);
                int x2 = std::pow(2.0f, parts[j].second(2)) * rootw.cols() -
                         x - partSz.first;
                partOrLR->getAnchor() << x2, y, parts[j].second(2);
            }
        }
    }

    getCachedFFTStatus() = false;

    traceDFSandBFS();
}

template<int Dimension>
void AOGrammar::turnOnRootOnly(const vector<int> & idx) {
    // turn off all nodes
    setOnOff(false);

    getRootNode()->getOnOff() = true;

    for(int i = 0; i < rootNode()->outEdges().size(); ++i) {
        Node * n = getRootNode()->getOutEdges()[i]->getToNode();
        bool isValid = idx.empty();
        for(int j = 0; j < idx.size(); ++j) {
            if(i == idx[j]) {
                isValid = true;
                break;
            }
        }

        if(isValid) {
            getRootNode()->getOutEdges()[i]->getOnOff() = true;

            do {
                n->getOnOff() = true;
                n->getOutEdges()[0]->getOnOff() = true;
                n = n->getOutEdges()[0]->getToNode();
            } while(n->outEdges().size() > 0);

            n->getOnOff() = true;
        }
    }

    traceDFSandBFS();

    finalize(false);

    getCachedFFTStatus() = false;
}

template<int Dimension>
void AOGrammar::pursueParts(const AppParam & rootw, int numParts,
                            int partWd, int partHt, int partScale,
                            vector<pair<AppParam, Anchor> > & parts) {
    RGM_CHECK((partScale == 0 || partScale == 1), error);

    AppParam rootwx;
    if(partScale == 0) {
        rootwx = rootw;
    } else {
        // Assume the parts will be place at twice the resolution of root
        const int s = 2;
        FeaturePyr::resize(rootw, s, rootwx);
    }

    // Compute the energy of each cell
    Matrix energy(rootwx.rows(), rootwx.cols());
    for(int y = 0; y < rootwx.rows(); ++y) {
        for(int x = 0; x < rootwx.cols(); ++x) {
            rootwx(y, x) = rootwx(y, x).cwiseMax(0);
            energy(y, x) = 0;
            for(int i = 0; i < Dimension; ++i) {
                energy(y, x) += rootwx(y, x)(i) * rootwx(y, x)(i);
            }
        }
    }

    // make a copy
    Matrix energyOrig = energy;

    // Assign each part greedily to the region of maximum energy
    parts.resize(numParts);
    for(int i = 0; i < numParts; ++i) {
        double maxEnergy = 0.0;
        int argX = 0;
        int argY = 0;

        for(int y = 0; y <= energy.rows() - partHt; ++y) {
            for(int x = 0; x <= energy.cols() - partWd; ++x) {
                const double e = energy.block(y, x, partHt, partWd).sum();

                if(e > maxEnergy) {
                    maxEnergy = e;
                    argX = x;
                    argY = y;
                }
            }
        }

        // Initialize the part
        parts[i].first = rootwx.block(argY, argX, partHt, partWd);
        parts[i].second << argX, argY, partScale;

        // Set the energy of the part to zero
        energy.block(argY, argX, partHt, partWd).setZero();
    }

    // Retry 10 times from randomized starting points
    double bestCover = 0.0;
    vector<pair<AppParam, Anchor> > best;

    Eigen::VectorXi progress(numParts);

    for(int i = 0; i < 10; ++i) {
        // Try from the current one
        vector<pair<AppParam, Anchor> > tmp(parts);

        progress.fill(1);

        // Remove a part at random and look for the best place to put it
        for(int j = 0; j < 1000; ++j) {
            if(progress.sum() == 0) {
                break;
            }

            // Recover the energy
            energy = energyOrig;

            // Select a part at random
            const int part = rand() % numParts;

            // Zero out the energy covered by the other parts
            for(int k = 0; k < numParts; ++k)
                if(k != part) {
                    energy.block(tmp[k].second(1), tmp[k].second(0),
                                 partHt, partWd).setZero();
                }

            // Find the region of maximum energy
            double maxEnergy = 0.0;
            int argX = 0;
            int argY = 0;

            for(int y = 0; y <= energy.rows() - partHt; ++y) {
                for(int x = 0; x <= energy.cols() - partWd; ++x) {
                    const double e = energy.block(y, x, partHt, partWd).sum();

                    if(e > maxEnergy) {
                        maxEnergy = e;
                        argX = x;
                        argY = y;
                    }
                }
            }

            if(tmp[part].second(0) == argX && tmp[part].second(1) == argY) {
                progress(part) = 0;
                continue;
            }

            progress(part) = 1;

            // Initialize the part
            tmp[part].first = rootwx.block(argY, argX, partHt, partWd);
            tmp[part].second << argX, argY, partScale;
        }

        // Compute the energy covered by this part arrangement
        double cover = 0.0;

        // Recover the energy
        energy = energyOrig;

        for(int j = 0; j < numParts; ++j) {
            // Add the energy of the part
            cover += energy.block(tmp[j].second(1), tmp[j].second(0),
                                  partHt, partWd).squaredNorm();

            // Set the energy of the part to zero
            energy.block(tmp[j].second(1), tmp[j].second(0),
                         partHt, partWd).setZero();
        }

        if(cover > bestCover) {
            bestCover = cover;
            best = tmp;
        }
    }

    // normalize part appearance parameters
    const Scalar alpha = 0.1F;
    for(int i = 0; i < best.size(); ++i) {

        for(int r = 0; r < best[i].first.rows(); ++r)
            for(int c = 0; c < best[i].first.cols(); ++c) {
                best[i].first(r, c) = best[i].first(r, c).cwiseMax(0);
            }

        Scalar n = FeaturePyr::Map(best[i].first).norm() + EPS;
        FeaturePyr::Map(best[i].first) *= (alpha / n);
    }

    parts.swap(best);

    /*int bs = 20;
    Mat img(energy.rows()*bs, energy.cols()*bs, CV_8UC1, cv::Scalar::all(0));

    for ( int i = 0; i < parts.size(); ++i ) {
        int argX = parts[i].second(0);
        int argY = parts[i].second(1);

        cv::rectangle(img, cv::Rect(argX*bs, argY*bs, partWd*bs, partHt*bs),
                      cv::Scalar::all(255), 3);
        cv::imshow("parts", img);
        cv::waitKey(0);
    }*/
}

template<int Dimension>
void AOGrammar::createRGM(int partScale, bool useScalePrior) {
    RGM_CHECK(partScale == 0 || partScale == 1, error);

    vector<Node *> sub = getSubcategoryRootAndNodes(false, false);

    int factor = partScale == 1 ? 2 : 1;

    for(int i = 0; i < sub.size(); ++i) {
        Node * n = sub[i];
        AOGrid * grid = getGridAOG()[i];
        RGM_CHECK_EQ(n->detectWindow().width() * factor,
                     grid->param().inputWindow_.width());
        RGM_CHECK_EQ(n->detectWindow().height() * factor,
                     grid->param().inputWindow_.height());

        if(n->scaleprior() != NULL) { //!gParam_.isSpecific_ &&
            n->getScaleprior()->reset(useScalePrior);
        }

        Node * t = n;
        while(t->type() != T_NODE) {
            t = t->getOutEdges()[0]->getToNode();
        }
        AppParam w = t->appearanceParam();

        AppParam wx;
        if(partScale == 1) {
            FeaturePyr::resize(w, factor, wx);
        } else {
            wx.swap(w);
        }

        Node * ch = createRGM(grid, wx, partScale);
        Edge * e = addEdge(n, ch, COMPOSITION);

        if(isLRFlip() && sharedLRFlip()) {
            Node * mch = addLRMirror(ch, true);
            Edge * me = addEdge(n->getLRMirrorNode(), mch, COMPOSITION);
            setEdgeLRFlip(e, me);
        }
    }

    traceDFSandBFS();

    finalize(false);

    getCachedFFTStatus() = false;
}

template<int Dimension>
Node * AOGrammar::createRGM(AOGrid * grid, AppParam & w, int partScale) {
    RGM_CHECK_NOTNULL(grid);
    RGM_CHECK((partScale == 0 || partScale == 1), error);

    AOGrid & sAOG(*grid);

    Node * gridRootOrNode = NULL;

    // appearance
    const Scalar alpha = 0.1F;

    std::map<int, Appearance *> sTNode_gApp_Map;
    for(int i = 0; i < sAOG.nodeSet().size(); ++i) {
        AOGrid::Vertex & snode(sAOG.getNodeSet()[i]);
        if(snode.idx_(AOGrid::Vertex::ID_ON_OFF) < 1 ||
                snode.type_ != T_NODE)
            continue;

        getAppearanceSet().push_back(new Appearance(featType()));

        int idx = snode.idx_[AOGrid::Vertex::ID_IN_INSTANCE_SET];
        Rectangle box(sAOG.instanceBbox(idx));
        if(snode.idx_[AOGrid::Vertex::ID] == sAOG.rootTermNodeId() &&
                partScale == 1) {
            box.setWidth(box.width() / 2);
            box.setHeight(box.height() / 2);
        }

        AppParam curw = w.block(box.y(), box.x(),
                                box.height(), box.width());
        for(int r = 0; r < curw.rows(); ++r)
            for(int c = 0; c < curw.cols(); ++c) {
                curw(r, c) = curw(r, c).cwiseMax(0);
            }
        Scalar norm = FeaturePyr::Map(curw).norm() + EPS;
        FeaturePyr::Map(curw) *= (alpha / norm);

        getAppearanceSet().back()->init(box.width(), box.height());
        getAppearanceSet().back()->getW().swap(curw);

        idx = snode.idx_[AOGrid::Vertex::ID];
        sTNode_gApp_Map.insert(std::make_pair(idx, getAppearanceSet().back()));
    }

    // nodeSet and edgeSet
    std::map<int, Node *> gridNodeIdgrammarNodeMap;
    typename std::map<int, Node *>::iterator iter;

    const vector<int> & DFS(sAOG.DFSqueue());
    for(int i = 0; i < DFS.size(); ++i) {
        AOGrid::Vertex & snode(sAOG.getNodeSet()[DFS[i]]);
        int id = snode.idx_[AOGrid::Vertex::ID];
        if(snode.idx_(AOGrid::Vertex::ID_ON_OFF) < 1 ||
                id == sAOG.rootTermNodeId())
            continue;

        int idx = snode.idx_[AOGrid::Vertex::ID_IN_INSTANCE_SET];
        Rectangle box(sAOG.instanceBbox(idx));

        switch(snode.type_) {
            case T_NODE: {
                // T-Node
                Node * gTNode = addNode(T_NODE);
                gTNode->getAppearance() = sTNode_gApp_Map[id];
                RGM_CHECK_NOTNULL(gTNode->appearance());
                gTNode->getDetectWindow() = Rectangle(box.width(), box.height());

                //And-Node
                pair<Node *, Edge *> gAnd(addParent(gTNode, AND_NODE, DEFORMATION));
                gAnd.first->getDetectWindow() = gTNode->detectWindow();

                // Deformation
                Deformation def;
                def.reset(); //(gParam_.isSpecific_ ? 1 : 0)
                gAnd.first->getDeformation() = addDeformation(def);

                gridNodeIdgrammarNodeMap.insert(std::make_pair(id, gAnd.first));

                snode.idx_[AOGrid::Vertex::ID_G] = idxNode(gTNode);

                break;
            }
            case AND_NODE: {
                // And-Node
                Node * gAndNode = addNode(AND_NODE);
                gAndNode->getDetectWindow() = Rectangle(box.width(), box.height());

                bool allsame = true;
                Anchor anchor;
                int dx, dy, ds;
                for(int ch = 0; ch < snode.childIDs_.size(); ++ch) {
                    const AOGrid::Vertex & snodeCh(sAOG.nodeSet()[snode.childIDs_[ch]]);
                    RGM_CHECK_GT(snodeCh.idx_(AOGrid::Vertex::ID_ON_OFF), 0);
                    idx = snodeCh.idx_[AOGrid::Vertex::ID_IN_INSTANCE_SET];
                    Rectangle boxch(sAOG.instanceBbox(idx));
                    dx = boxch.x() - box.x();
                    dy = boxch.y() - box.y();
                    ds = 0;
                    anchor << dx, dy, ds;

                    iter = gridNodeIdgrammarNodeMap.find(
                               snodeCh.idx_[AOGrid::Vertex::ID]);
                    RGM_CHECK_NOTEQ(iter, gridNodeIdgrammarNodeMap.end());

                    // check the other anchors of the child w.r.t. other parent nodes
                    allsame = true;
                    for(int p = 0; p < snodeCh.parentIDs_.size(); ++p) {
                        const AOGrid::Vertex & parent(sAOG.nodeSet()[snodeCh.parentIDs_[p]]);
                        idx = parent.idx_[AOGrid::Vertex::ID_IN_INSTANCE_SET];
                        Rectangle boxp(sAOG.instanceBbox(idx));
                        int dx1 = boxch.x() - boxp.x();
                        int dy1 = boxch.y() - boxp.y();
                        if(dx1 != dx || dy1 != dy) {
                            allsame = false;
                            break;
                        }
                    }

                    if(allsame) {
                        iter->second->getAnchor() = anchor;
                        addEdge(gAndNode, iter->second, COMPOSITION);
                    } else {
                        pair<Node *, Edge *> gOr(addParent(iter->second,
                                                           OR_NODE, SWITCHING));
                        gOr.first->getDetectWindow() = iter->second->detectWindow();
                        gOr.first->getAnchor() = anchor;
                        addEdge(gAndNode, gOr.first, COMPOSITION);
                    }
                }

                gridNodeIdgrammarNodeMap.insert(std::make_pair(id, gAndNode));

                snode.idx_[AOGrid::Vertex::ID_G] = idxNode(gAndNode);

                break;
            }
            case OR_NODE: {
                // Or-Node
                Node * ornode  = addNode(OR_NODE);
                for(int ch = 0; ch < snode.childIDs_.size(); ++ch) {
                    const AOGrid::Vertex & snodeCh(sAOG.nodeSet()[snode.childIDs_[ch]]);
                    if(snodeCh.idx_(AOGrid::Vertex::ID_ON_OFF) < 1 ||
                            snode.childIDs_[ch] == sAOG.rootTermNodeId())
                        continue;

                    iter = gridNodeIdgrammarNodeMap.find(snode.childIDs_[ch]);
                    RGM_CHECK_NOTEQ(iter, gridNodeIdgrammarNodeMap.end());
                    addEdge(ornode, iter->second, SWITCHING);

                    // add bias if more than one children are on
                    //                Offset bias;
                    //                bias.reset();
                    //                iter->second->getBias() = addOffset(bias);
                }
                ornode->getDetectWindow() = Rectangle(box.width(), box.height());

                if(snode.parentIDs_.size() == 0) {
                    ornode->getAnchor().setZero();
                    gridRootOrNode = ornode;
                }

                gridNodeIdgrammarNodeMap.insert(std::make_pair(id, ornode));

                snode.idx_[AOGrid::Vertex::ID_G] = idxNode(ornode);

                break;
            }
        }
    }

    gridRootOrNode->getAnchor() << 0, 0, partScale;

    return gridRootOrNode;
}

template<int Dimension>
Node * AOGrammar::addNode(nodeType t) {
    getNodeSet().push_back(new Node(t));
    return getNodeSet().back();
}

template<int Dimension>
Edge * AOGrammar::addEdge(Node * from, Node * to, edgeType t) {
    RGM_CHECK_NOTNULL(from);
    RGM_CHECK_NOTNULL(to);

    bool dup = false;
    for(int i = 0; i < edgeSet().size(); ++i) {
        if(edgeSet()[i]->fromNode() == from && edgeSet()[i]->toNode() == to) {
            dup = true;
            break;
        }
    }

    if(dup) {
        RGM_LOG(error, "add duplicated edge");
    }

    getEdgeSet().push_back(new Edge(t, from, to));
    Edge * e = getEdgeSet().back();

    from->getOutEdges().push_back(e);
    to->getInEdges().push_back(e);

    return e;
}

template<int Dimension>
Bias * AOGrammar::addBias(const Bias & bias) {
    getBiasSet().push_back(new Bias(bias));
    return getBiasSet().back();
}

template<int Dimension>
Scaleprior * AOGrammar::addScaleprior(const Scaleprior & prior) {
    getScalepriorSet().push_back(new Scaleprior(prior));
    return getScalepriorSet().back();
}

template<int Dimension>
Deformation * AOGrammar::addDeformation(const Deformation & def) {
    getDeformationSet().push_back(new Deformation(def));
    return getDeformationSet().back();
}

template<int Dimension>
pair<Node *, Edge *> AOGrammar::addChild(Node * parent, nodeType chType,
                                         edgeType edgeType) {
    RGM_CHECK_NOTNULL(parent);

    Node * ch = addNode(chType);

    Edge * e = addEdge(parent, ch, edgeType);

    return std::make_pair(ch, e);
}

template<int Dimension>
pair<Node *, Edge *> AOGrammar::addParent(Node * ch, nodeType paType,
                                          edgeType edgeType) {
    RGM_CHECK_NOTNULL(ch);

    Node * pa = addNode(paType);

    Edge * e = addEdge(pa, ch, edgeType);

    return std::make_pair(pa, e);
}

template<int Dimension>
Node * AOGrammar::addLRMirror(Node * root, bool shareParam) {
    RGM_CHECK_NOTNULL(root);

    vector<Node *> dfs;

    vector<int> visited(nodeSet().size(), 0);
    traceNodeDFS(root, visited, dfs);

    for(int i = 0; i < dfs.size(); ++i) {
        Node * n = dfs[i];
        Node * mn = addNode(n->type());
        setNodeLRFlip(n, mn);

        mn->getDetectWindow() = n->detectWindow();

        if(n->bias() != NULL) {
            if(shareParam) {
                mn->getBias() = n->getBias();
            } else {
                Bias bias;
                bias.reset();
                mn->getBias() = addBias(bias);
            }
        }

        if(n->scaleprior() != NULL) {
            if(shareParam) {
                mn->getScaleprior() = n->getScaleprior();
            } else {
                Scaleprior prior;
                prior.reset(false);
                mn->getScaleprior() = addScaleprior(prior);
            }
        }

        if(n->deformation() != NULL) {
            if(shareParam) {
                mn->getDeformation() = n->getDeformation();
            } else {
                Deformation def;
                def.reset();
                mn->getDeformation() = addDeformation(def);
            }
        }

        if(n->appearance() != NULL) {
            if(shareParam) {
                mn->getAppearance() = n->getAppearance();
            } else {
                getAppearanceSet().push_back(new Appearance(featType()));
                mn->getAppearance() = getAppearanceSet().back();
                mn->getAppearance()->init(n->appearance()->w().cols(),
                                          n->appearance()->w().rows());
            }
        }

        const Anchor & anchor(n->anchor());
        mn->getAnchor() = anchor;
        if(n->inEdges().size() == 1) {
            Node * parent = n->getInEdges()[0]->getFromNode();
            RGM_CHECK_NOTNULL(parent);
            if(parent != rootNode()) {
                int wd = parent->detectWindow().width();
                RGM_CHECK_GT(wd, 0);
                mn->getAnchor()(0) = std::pow(2.0f, anchor(2)) * wd -
                                     anchor(0) - mn->detectWindow().width();
            }
        }

        for(int j = 0; j < n->outEdges().size(); ++j) {
            Edge * e = n->getOutEdges()[j];
            Node * nch = e->getToNode();
            Edge * me = addEdge(mn, nch->getLRMirrorNode(), e->type());
            setEdgeLRFlip(e, me);
        }
    }

    return root->getLRMirrorNode();
}

template<int Dimension>
void AOGrammar::setNodeLRFlip(Node * n, Node * nFlip) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK_NOTNULL(nFlip);

    n->getIsLRFlip() = false;
    n->getLRMirrorNode() = nFlip;

    nFlip->getIsLRFlip() = true;
    nFlip->getLRMirrorNode() = n;
}

template<int Dimension>
void AOGrammar::setEdgeLRFlip(Edge * e, Edge * eFlip) {
    RGM_CHECK_NOTNULL(e);
    RGM_CHECK_NOTNULL(eFlip);

    e->getIsLRFlip() = false;
    e->getLRMirrorEdge() = eFlip;

    eFlip->getIsLRFlip() = true;
    eFlip->getLRMirrorEdge() = e;
}

template<int Dimension>
Node * AOGrammar::addAppearanceUnit(const pair<int, int> & sz,
                                    const Anchor & anchor,
                                    bool hasBias, bool hasScaleprior, bool hasDef) {
    // appearance parameters
    getAppearanceSet().push_back(new Appearance(featType()));
    Appearance * app = getAppearanceSet().back();
    app->init(sz.first, sz.second);

    // T-node
    Node * tNode = addNode(T_NODE);
    tNode->getAppearance() = app;
    tNode->getDetectWindow() = Rectangle(sz.first, sz.second);

    // AND-node
    pair<Node *, Edge *>  andnode(addParent(tNode, AND_NODE,
                                            hasDef ? DEFORMATION : TERMINATION));
    andnode.first->getDetectWindow() = tNode->detectWindow();
    if(hasBias) {
        Bias bias;
        bias.reset();
        andnode.first->getBias() = addBias(bias);
    }

    if(hasScaleprior) {
        Scaleprior prior;
        prior.reset(false);
        andnode.first->getScaleprior() = addScaleprior(prior);
    }

    if(hasDef) {
        Deformation def;
        def.reset();
        andnode.first->getDeformation() = addDeformation(def);
    }

    // OR-node
    pair<Node *, Edge *> ornode(addParent(andnode.first, OR_NODE, SWITCHING));
    ornode.first->getDetectWindow() = tNode->detectWindow();
    ornode.first->getAnchor() = anchor;

    return ornode.first;
}

template<int Dimension>
Node * AOGrammar::addAppearanceUnit(AppParam & w, const Anchor & anchor,
                                    bool hasBias, bool hasScaleprior, bool hasDef) {
    // appearance parameters
    getAppearanceSet().push_back(new Appearance(featType()));
    Appearance * app = getAppearanceSet().back();
    app->init(w.cols(), w.rows());
    app->getW().swap(w);

    // T-node
    Node * tNode = addNode(T_NODE);
    tNode->getAppearance() = app;
    tNode->getDetectWindow() = Rectangle(w.cols(), w.rows());

    // AND-node
    pair<Node *, Edge *>  andnode(addParent(tNode, AND_NODE,
                                            hasDef ? DEFORMATION : TERMINATION));
    andnode.first->getDetectWindow() = tNode->detectWindow();
    if(hasBias) {
        Bias bias;
        bias.reset();
        andnode.first->getBias() = addBias(bias);
    }

    if(hasScaleprior) {
        Scaleprior prior;
        prior.reset(false);
        andnode.first->getScaleprior() = addScaleprior(prior);
    }

    if(hasDef) {
        Deformation def;
        def.reset();
        andnode.first->getDeformation() = addDeformation(def);
    }

    // OR-node
    pair<Node *, Edge *> ornode(addParent(andnode.first, OR_NODE, SWITCHING));
    ornode.first->getDetectWindow() = tNode->detectWindow();
    ornode.first->getAnchor() = anchor;

    return ornode.first;
}

template<int Dimension>
void AOGrammar::initAppFromRoot(int ds, int target, bool doNormalization) {
    RGM_CHECK((ds == 0 || ds == 1), error);
    if(gridAOG().size() == 0)
        return;

    vector<Node *> sub = getSubcategoryRootAndNodes(false, false);
    RGM_CHECK_EQ(sub.size(), gridAOG().size());

    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar alpha = 0.1F;

    for(int g = 0; g < gridAOG().size(); ++g) {
        // Get root appearance parameters
        Node * t = sub[g];
        while(t->type() != T_NODE) {
            t = t->getOutEdges()[0]->getToNode();
        }
        AppParam root = t->appearanceParam();

        AppParam rootx;
        if(ds == 0) {
            rootx.swap(root);
        } else {
            FeaturePyr::resize(root, 2, rootx);
        }

        for(int r = 0; r < rootx.rows(); ++r) {
            for(int c = 0; c < rootx.cols(); ++c) {
                rootx(r, c) = rootx(r, c).cwiseMax(0);
            }
        }

        for(int i = 0; i < gridAOG()[g]->nodeSet().size(); ++i) {
            const AOGrid::Vertex & snode(gridAOG()[g]->nodeSet()[i]);
            if(snode.type_ != T_NODE ||
                    snode.idx_(AOGrid::Vertex::ID) == gridAOG()[g]->rootTermNodeId())
                continue;

            int idx = snode.idx_(AOGrid::Vertex::ID_G);
            Node * gNode = findNode(idx);
            RGM_CHECK_NOTNULL(gNode);

            if(target < 0 && gNode->onOff())
                continue;

            if(target > 0 && !gNode->onOff())
                continue;

            idx = snode.idx_(AOGrid::Vertex::ID_IN_INSTANCE_SET);
            Rectangle box(gridAOG()[g]->instanceBbox(idx));

            AppParam &w(gNode->getAppearance()->getW());

            RGM_CHECK_EQ(w.cols(), box.width());
            RGM_CHECK_EQ(w.rows(), box.height());

            w = rootx.block(box.y(), box.x(), box.height(), box.width());

            // normalize it
            if(doNormalization) {
                Scalar n = FeaturePyr::Map(w).norm() + eps;
                FeaturePyr::Map(w) *= (alpha / n);
            }
        }
    }
}

template<int Dimension>
void AOGrammar::initAppFromRoot(int ds, const std::vector<int> &idxNotUpdate,
                                bool doNormalization) {
    RGM_CHECK((ds == 0 || ds == 1), error);
    if(gridAOG().size() == 0)
        return;

    vector<Node *> sub = getSubcategoryRootAndNodes(false, false);
    RGM_CHECK_EQ(sub.size(), gridAOG().size());

    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar alpha = 0.1F;

    for(int g = 0; g < gridAOG().size(); ++g) {
        // Get root appearance parameters
        Node * t = sub[g];
        while(t->type() != T_NODE) {
            t = t->getOutEdges()[0]->getToNode();
        }
        AppParam root = t->appearanceParam();

        AppParam rootx;
        if(ds == 0) {
            rootx.swap(root);
        } else {
            FeaturePyr::resize(root, 2, rootx);
        }

        for(int r = 0; r < rootx.rows(); ++r) {
            for(int c = 0; c < rootx.cols(); ++c) {
                rootx(r, c) = rootx(r, c).cwiseMax(0);
            }
        }

        for(int i = 0; i < gridAOG()[g]->nodeSet().size(); ++i) {
            const AOGrid::Vertex & snode(gridAOG()[g]->nodeSet()[i]);
            if(snode.type_ != T_NODE ||
                    snode.idx_(AOGrid::Vertex::ID) == gridAOG()[g]->rootTermNodeId())
                continue;

            int idx = snode.idx_(AOGrid::Vertex::ID_G);
            Node * gNode = findNode(idx);
            RGM_CHECK_NOTNULL(gNode);

            bool found = false;
            for(int j = 0; j < idxNotUpdate.size(); ++j) {
                if(idx == idxNotUpdate[j]) {
                    found = true;
                    break;
                }
            }
            if(found)
                continue;

            idx = snode.idx_(AOGrid::Vertex::ID_IN_INSTANCE_SET);
            Rectangle box(gridAOG()[g]->instanceBbox(idx));

            AppParam &w(gNode->getAppearance()->getW());

            RGM_CHECK_EQ(w.cols(), box.width());
            RGM_CHECK_EQ(w.rows(), box.height());

            w = rootx.block(box.y(), box.x(), box.height(), box.width());

            // normalize it
            if(doNormalization) {
                Scalar n = FeaturePyr::Map(w).norm() + eps;
                FeaturePyr::Map(w) *= (alpha / n);
            }
        }
    }
}

template<int Dimension>
void AOGrammar::traceNodeDFS(Node * curNode, vector<int> & visited,
                             vector<Node *> & nodeDFS, bool onlyOnNodes) {
    int idx = idxNode(curNode);
    if(visited[idx] == 1) {
        RGM_LOG(error, "DFS: Cycle detected in grammar!");
        return;
    }
    if(onlyOnNodes && !curNode->onOff()) {
        RGM_LOG(error, "DFS: Meet a node turned off");
        return;
    }

    visited[idx] = 1;

    int numChild = curNode->outEdges().size();

    for(int i = 0; i < numChild; ++i) {
        if(onlyOnNodes && !curNode->outEdges()[i]->onOff())
            continue;
        Node * ch = curNode->getOutEdges()[i]->getToNode();
        if(onlyOnNodes && !ch->onOff())
            continue;

        int chIdx = idxNode(ch);
        if(visited[chIdx] < 2) {
            traceNodeDFS(ch, visited, nodeDFS, onlyOnNodes);
        }
    }

    nodeDFS.push_back(curNode);
    visited[idx] = 2;
}

template<int Dimension>
void AOGrammar::traceNodeBFS(Node * curNode, vector<int> & visited,
                             vector<Node *> & nodeBFS, bool onlyOnNodes) {
    // assume visited.size() == nodeSet.size()

    int idx = idxNode(curNode);
    if(visited[idx] == 1) {
        RGM_LOG(error, "BFS: Cycle detected in grammar!");
        return;
    }
    if(onlyOnNodes && !curNode->onOff()) {
        RGM_LOG(error, "BFS: Meet a node turned off");
        return;
    }

    nodeBFS.push_back(curNode);

    visited[idx] = 1;

    int numChild = curNode->outEdges().size();

    for(int i = 0; i < numChild; ++i) {
        if(onlyOnNodes && !curNode->outEdges()[i]->onOff())
            continue;
        Node * ch = curNode->getOutEdges()[i]->getToNode();
        if(onlyOnNodes && !ch->onOff())
            continue;
        int chIdx = idxNode(ch);
        if(visited[chIdx] < 2) {
            traceNodeBFS(ch, visited, nodeBFS, onlyOnNodes);
        }
    }

    visited[idx] = 2;
}

template<int Dimension>
void AOGrammar::traceDFSandBFS(bool onlyOnNodes) {
    getNodeDFS().clear();
    getNodeBFS().clear();

    getCompNodeDFS().clear();
    getCompNodeBFS().clear();

    if(onlyOnNodes && !rootNode()->onOff())
        return;

    vector<int> visitedDFS(nodeSet().size(), 0);
    traceNodeDFS(getRootNode(), visitedDFS, getNodeDFS(), onlyOnNodes);

    vector<int> visitedBFS(nodeSet().size(), 0);
    traceNodeBFS(getRootNode(), visitedBFS, getNodeBFS(), onlyOnNodes);

    traceCompNodeDFSandBFS(onlyOnNodes);
}

template<int Dimension>
void AOGrammar::traceCompNodeDFSandBFS(bool onlyOnNodes) {
    vector<Node *> sub = getSubcategoryRootAndNodes(onlyOnNodes, true);

    getCompNodeDFS().clear();
    getCompNodeBFS().clear();

    getCompNodeDFS().resize(sub.size());
    getCompNodeBFS().resize(sub.size());

    for(int i = 0; i < sub.size(); ++i) {
        Node * node = sub[i];

        getCompNodeDFS()[i].clear();

        vector<int> visited(nodeSet().size(), 0);
        traceNodeDFS(node, visited, getCompNodeDFS()[i], onlyOnNodes);

        getCompNodeBFS()[i].clear();
        visited.assign(visited.size(), 0);
        traceNodeBFS(node, visited, getCompNodeBFS()[i], onlyOnNodes);
    }
}

template<int Dimension>
int AOGrammar::nbSubcategories(bool isOnOnly, bool withLRFlip) {
    int num = 0;
    for(int i = 0; i < rootNode()->outEdges().size(); ++i) {
        if(isOnOnly && (!rootNode()->outEdges()[i]->onOff() ||
                        !rootNode()->outEdges()[i]->toNode()->onOff()))
            continue;
        if(isLRFlip() && sharedLRFlip() && !withLRFlip &&
                (rootNode()->outEdges()[i]->isLRFlip() ||
                 rootNode()->outEdges()[i]->toNode()->isLRFlip()))
            continue;

        ++num;
    }

    return num;
}

template<int Dimension>
vector<Node *> AOGrammar::getSubcategoryRootAndNodes(bool isOnOnly,
                                                     bool withLRFlip) {
    vector<Node *> sub;

    for(int i = 0; i < rootNode()->outEdges().size(); ++i) {
        if(isOnOnly && (!rootNode()->outEdges()[i]->onOff() ||
                        !rootNode()->outEdges()[i]->toNode()->onOff()))
            continue;
        if(isLRFlip() && sharedLRFlip() && !withLRFlip &&
                (rootNode()->outEdges()[i]->isLRFlip() ||
                 rootNode()->outEdges()[i]->toNode()->isLRFlip()))
            continue;

        sub.push_back(getRootNode()->getOutEdges()[i]->getToNode());
    }

    return sub;
}

template<int Dimension>
void AOGrammar::cachingFilters() {
    if(cachedFFTStatus()) {
        return;
    }

    // release old ones
    for(int i = 0; i < cachedFilters().size(); ++i) {
        delete getCachedFilters()[i];
    }
    getCachedFilters().clear();

    vector<AppParam> w;
    for(int i = 0; i < nodeSet().size(); ++i) {
        if(nodeSet()[i]->type() == T_NODE &&
                nodeSet()[i]->onOff()) {
            w.push_back(nodeSet()[i]->appearanceParam());
        }
    }

    getCachedFilters().resize(w.size(), NULL);
    for(int i = 0; i < w.size(); ++i) {
        getCachedFilters()[i] = new Filter();
    }

    #pragma omp parallel for
    for(int i = 0; i < w.size(); ++i) {
        Patchwork::TransformFilter(w[i], *(getCachedFilters()[i]));
    }

    getCachedFFTStatus() = true;
}

template<int Dimension>
int AOGrammar::dim() const {
    int d = 0;

    // Count all the parameters using DFS
    const vector<Node * > & DFS = nodeDFS();

    for(int i = 0; i < DFS.size(); ++i) {
        const Node * n = DFS[i];
        if(n->isLRFlip() && isLRFlip() && sharedLRFlip()) {
            continue;
        }

        nodeType t = n->type();
        switch(t) {
            case T_NODE: {
                const AppParam & w = n->appearance()->w();
                d += static_cast<int>(w.size()) * Dimension;
                break;
            }
            case AND_NODE: {
                if(n->deformation() != NULL) {
                    d += 4;
                }

                if(n->scaleprior() != NULL) {
                    d += 3;
                }

                if(n->bias() != NULL) {
                    d++;
                }

                break;
            }
        } // switch
    } // for i

    return d;
}

template<int Dimension>
template<class Archive>
void AOGrammar::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(gParam_);

    ar.register_type(static_cast<Node *>(NULL));
    ar.register_type(static_cast<Edge *>(NULL));
    ar.register_type(static_cast<ParamUtil *>(NULL));
    ar.register_type(static_cast<Appearance *>(NULL));
    ar.register_type(static_cast<Bias *>(NULL));
    ar.register_type(static_cast<Deformation *>(NULL));
    ar.register_type(static_cast<Scaleprior *>(NULL));
    ar.register_type(static_cast<AOGrid *>(NULL));

    ar.template register_type<Node >();
    ar.template register_type<Edge >();
    ar.template register_type<ParamUtil>();
    ar.template register_type<Appearance >();
    ar.template register_type<Bias>();
    ar.template register_type<Deformation>();
    ar.template register_type<Scaleprior>();
    ar.template register_type<AOGrid>();

    ar & BOOST_SERIALIZATION_NVP(nodeSet_);
    ar & BOOST_SERIALIZATION_NVP(edgeSet_);

    ar & BOOST_SERIALIZATION_NVP(appearanceSet_);
    ar & BOOST_SERIALIZATION_NVP(biasSet_);
    ar & BOOST_SERIALIZATION_NVP(deformationSet_);
    ar & BOOST_SERIALIZATION_NVP(scalepriorSet_);

    ar & BOOST_SERIALIZATION_NVP(maxDetectWindow_);
    ar & BOOST_SERIALIZATION_NVP(minDetectWindow_);

    ar & BOOST_SERIALIZATION_NVP(featureParam_);
    ar & BOOST_SERIALIZATION_NVP(bgmu_);

    ar & BOOST_SERIALIZATION_NVP(thresh_);

    ar & BOOST_SERIALIZATION_NVP(bboxPred_);

    ar & BOOST_SERIALIZATION_NVP(gridAOG_);

    ar & BOOST_SERIALIZATION_NVP(idxRootNode_);

}

template<int Dimension>
void AOGrammar::save(const string & modelFile, int archiveType) {
    finalize(false);

    RGM_SAVE(modelFile, archiveType);
}

template<int Dimension>
bool AOGrammar::read(const string & modelFile, int archiveType) {
    RGM_READ(modelFile, archiveType);

    finalize(true);

    getCachedFFTStatus() = false;

    return true;
}

template<int Dimension>
void AOGrammar::finalize(bool hasIdx) {
    if(hasIdx) {
        for(int i = 0; i < nodeSet().size(); ++i) {
            getNodeSet()[i]->assignConnections(this);
        }

        for(int i = 0; i < edgeSet().size(); ++i) {
            getEdgeSet()[i]->assignConnections(this);
        }

        getRootNode() = findNode(idxRootNode_);

    } else {
        for(int i = 0; i < nodeSet().size(); ++i) {
            getNodeSet()[i]->assignIdx(this);
        }

        for(int i = 0; i < edgeSet().size(); ++i) {
            getEdgeSet()[i]->assignIdx(this);
        }

        RGM_CHECK_NOTNULL(rootNode());
        idxRootNode_ = idxNode(rootNode());

    }

    traceDFSandBFS();
}

template<int Dimension>
void save(const string & modelFile, const vector<AOGrammar > & models,
          int archiveType) {
    DEFINE_RGM_LOGGER;

    std::ofstream out;
    out.open(modelFile.c_str(), std::ios::out);

    if(!out.is_open()) {
        RGM_LOG(error, "Can not write models to " + modelFile);
        return;
    }

    switch(archiveType) {
        case 1: {
            boost::archive::text_oarchive oa(out);
            oa << models;
            break;
        }
        case 0:
        default: {
            boost::archive::binary_oarchive oa(out);
            oa << models;
            break;
        }
    } // switch

    out.close();
}

template<int Dimension>
bool load(const string & modelFile, vector<AOGrammar > & models,
          int archiveType) {
    std::ifstream in;
    in.open(modelFile.c_str(), std::ios::in);

    if(!in.is_open()) {
        return false;
    }

    models.clear();

    switch(archiveType) {
        case 1: {
            boost::archive::text_iarchive ia(in);
            ia >> models;
            break;
        }
        case 0:
        default: {
            boost::archive::binary_iarchive ia(in);
            ia >> models;
            break;
        }
    } // switch

    in.close();

    for(int i = 0; i < models.size(); ++i) {
        models[i].finalize(true);
    }

    return true;
}

template<int Dimension>
void AOGrammar::setOnOff(bool onoff) {
    for(int i = 0; i < nodeSet().size(); ++i) {
        getNodeSet()[i]->getOnOff() = onoff;
    }

    for(int i = 0; i < edgeSet().size(); ++i) {
        getEdgeSet()[i]->getOnOff() = onoff;
    }
}

template<int Dimension>
void AOGrammar::setOnOff(const int gridIdx, const vector<int> &ptBFS) {
    RGM_CHECK_NOTNULL(gridAOG_[gridIdx]);

    AOGrid & sAOG(*gridAOG_[gridIdx]);

    for(int i = 0; i < ptBFS.size(); ++i) {
        int idx = sAOG.nodeSet()[ptBFS[i]].idx_(AOGrid::Vertex::ID_G);
        Node * gNode = findNode(idx);
        RGM_CHECK_NOTNULL(gNode);

        gNode->getOnOff() = true;

        if(gNode->inEdges().size() == 1) {
            gNode->getInEdges()[0]->getOnOff() = true;
            gNode->getInEdges()[0]->getFromNode()->getOnOff() = true;
        }

        switch(gNode->type()) {
            case AND_NODE: {
                for(int j = 0; j < gNode->outEdges().size(); ++j) {
                    gNode->getOutEdges()[j]->getOnOff() = true;
                    Node * ch = gNode->getOutEdges()[j]->getToNode();
                    ch->getOnOff() = true;
                    if(ch->type() == OR_NODE &&
                            ch->outEdges().size() == 1) {
                        ch->getOutEdges()[0]->getOnOff() = true;
                        ch->getOutEdges()[0]->getToNode()->getOnOff() = true;
                    }
                }
                break;
            }
            case OR_NODE: {
                if(gNode->outEdges().size() == 1) {
                    gNode->getOutEdges()[0]->getOnOff() = true;
                    gNode->getOutEdges()[0]->getToNode()->getOnOff() = true;
                }
                break;
            }
            case T_NODE: {
                // and-node
                gNode->getInEdges()[0]->getOnOff() = true;

                Node * node = gNode->getInEdges()[0]->getFromNode();
                node->getOnOff() = true;

                // or-node
                node->getInEdges()[0]->getOnOff() = true;

                node = node->getInEdges()[0]->getFromNode();
                node->getOnOff() = true;

                if(node->getInEdges().size() == 1) {
                    node->getInEdges()[0]->getOnOff() = true;
                    node->getInEdges()[0]->getFromNode()->getOnOff() = true;
                }
                break;
            }
        }
    }
}

template<int Dimension>
void AOGrammar::getOnOff(vector<bool> & statusNodes) {
    statusNodes.resize(nodeSet().size());
    for(int i = 0; i < nodeSet().size(); ++i) {
        statusNodes[i] = nodeSet()[i]->onOff();
    }
}

template<int Dimension>
void AOGrammar::getOnOff(vector<bool> & statusNodes,
                         vector<bool> & statusEdges) {
    if(statusNodes.size() != nodeSet().size()) {
        statusNodes.assign(nodeSet().size(), false);
    }

    if(statusEdges.size() != edgeSet().size()) {
        statusEdges.assign(edgeSet().size(), false);
    }

    for(int i = 0; i < nodeSet().size(); ++i) {
        statusNodes[i] = statusNodes[i] || nodeSet()[i]->onOff();
    }

    for(int i = 0; i < edgeSet().size(); ++i) {
        statusEdges[i] = statusEdges[i] || edgeSet()[i]->onOff();
    }
}

template<int Dimension>
void AOGrammar::turnOff(vector<int> & turnedOffTnodeIdx) {
    if(turnedOffTnodeIdx.size() == 0)
        return;

    for(int i = 0; i < turnedOffTnodeIdx.size(); ++i) {
        int idx = turnedOffTnodeIdx[i];

        Node * tnode = findNode(idx);
        tnode->getOnOff() = false;
        tnode->getInEdges()[0]->getOnOff() = false;

        Node * anode = tnode->getInEdges()[0]->getFromNode();
        anode->getOnOff() = false;
        anode->getInEdges()[0]->getOnOff() = false;

        Node * onode = anode->getInEdges()[0]->getFromNode();
        onode->getOnOff() = false;
        for(int j = 0; j < onode->inEdges().size(); ++j) {
            onode->getInEdges()[j]->getOnOff() = false;
        }
    }

    bool onOff;
    bool changed = true;
    while(changed) {
        vector<Node *> BFS;
        BFS.push_back(getRootNode());
        int head = 0;

        int num = 0;
        while(head < BFS.size()) {
            Node * node = BFS[head++];
            nodeType t = node->type();

            onOff = false;
            switch(t) {
                case AND_NODE:
                case OR_NODE:
                    for(int i = 0; i < node->outEdges().size(); ++i) {
                        if(node->outEdges()[i]->onOff() &&
                                node->outEdges()[i]->toNode()->onOff()) {
                            BFS.push_back(node->getOutEdges()[i]->getToNode());
                            onOff = true;
                        }
                    }
                    if(!onOff) {
                        node->getOnOff() = false;
                        for(int i = 0; i < node->inEdges().size(); ++i) {
                            node->getInEdges()[i]->getOnOff() = false;
                        }
                        for(int i = 0; i < node->outEdges().size(); ++i) {
                            node->getOutEdges()[i]->getOnOff() = false;
                        }
                        num++;
                    }
                    break;
            }
        }

        changed = num > 0;
    }

    traceDFSandBFS();

    getCachedFFTStatus() = false;
}

template<int Dimension>
bool AOGrammar::isDAG() const {

    const vector<Node * > & DFS(nodeDFS());
    for(int i = 0; i < DFS.size(); ++i) {
        const Node * n = DFS[i];
        const vector<Edge *> &inEdges(n->inEdges());
        int numOn = 0;
        for ( int j = 0; j < inEdges.size(); ++j ) {
            if ( inEdges[j]->onOff() ) {
                ++numOn;
            }
            if ( numOn > 1) {
                return true;
            }
        }
    }

    return false;
}

template<int Dimension>
void AOGrammar::assignParameters(const double * x) {
    RGM_CHECK_NOTNULL(x);

    // Assign all the parameters using DFS
    vector<Node * > & DFS(getNodeDFS());
    int idx = 0;
    for(int i = 0; i < DFS.size(); ++i) {
        Node * n = DFS[i];
        if(isLRFlip() && sharedLRFlip() && n->isLRFlip()) {
            continue;
        }

        nodeType t = n->type();
        switch(t) {
            case T_NODE: {
                AppParam & w = n->getAppearance()->getW();
                const int dim = static_cast<int>(w.size()) *  Dimension;
                std::copy(x + idx, x + idx + dim, w.data()->data());
                idx += dim;
                break;
            }
            case AND_NODE: {
                if(n->deformation() != NULL) {
                    Deformation::Param & w = n->getDeformation()->getW();
                    const int dim = static_cast<int>(w.size());
                    RGM_CHECK_EQ(dim, 4);
                    std::copy(x + idx, x + idx + dim, w.data());
                    idx += dim;
                }
                if(n->scaleprior() != NULL) {
                    Scaleprior::Param & w = n->getScaleprior()->getW();
                    const int dim = static_cast<int>(w.size());
                    RGM_CHECK_EQ(dim, 3);
                    std::copy(x + idx, x + idx + dim, w.data());
                    idx += dim;
                }
                if(n->bias() != NULL) {
                    Scalar & w = n->getBias()->getW();
                    w = x[idx++];
                }
                break;
            }
        } // switch
    } // for i
}

template<int Dimension>
void AOGrammar::getParameters(double * x, int which) {
    // Concatenates all the parameters using DFS
    const vector<Node * > & DFS(nodeDFS());
    int idx = 0;
    for(int i = 0; i < DFS.size(); ++i) {
        const Node * n = DFS[i];
        if(isLRFlip() && sharedLRFlip() && n->isLRFlip()) {
            continue;
        }

        nodeType t = n->type();
        switch(t) {
            case T_NODE: {
                switch(which) {
                    case 0:
                    case 1: {
                        const AppParam & w = (which == 0) ?
                                             n->appearance()->w() :
                                             n->appearance()->lowerBound();
                        const int dim = static_cast<int>(w.size()) * Dimension;
                        std::copy(w.data()->data(), w.data()->data() + dim, x + idx);
                        idx += dim;
                        break;
                    }
                    case 2: {
                        const typename Appearance::dParam & w = n->appearance()->gradient();
                        const int dim = static_cast<int>(w.size()) * Dimension;
                        std::copy(w.data()->data(), w.data()->data() + dim, x + idx);
                        idx += dim;
                        break;
                    }
                    default: {
                        RGM_LOG(error, "The grammar doesnt have what you want.");
                        break;
                    }
                }
                break;
            }
            case AND_NODE: {
                if(n->deformation() != NULL) {
                    switch(which) {
                        case 0:
                        case 1: {
                            const Deformation::Param & w = ((which == 0) ?
                                                            n->deformation()->w() :
                                                            n->deformation()->lowerBound());
                            const int dim = static_cast<int>(w.size());
                            RGM_CHECK_EQ(dim, 4);
                            std::copy(w.data(), w.data() + dim, x + idx);
                            idx += dim;
                            break;
                        }
                        case 2: {
                            const Deformation::dParam & w = n->deformation()->gradient();
                            const int dim = static_cast<int>(w.size());
                            RGM_CHECK_EQ(dim, 4);
                            std::copy(w.data(), w.data() + dim, x + idx);
                            idx += dim;
                            break;
                        }
                        default: {
                            RGM_LOG(error, "The grammar doesnt have what you want.");
                            break;
                        }
                    }
                }

                if(n->scaleprior() != NULL) {
                    switch(which) {
                        case 0:
                        case 1: {
                            const Scaleprior::Param & w = ((which == 0) ?
                                                           n->scaleprior()->w() :
                                                           n->scaleprior()->lowerBound());
                            const int dim = static_cast<int>(w.size());
                            RGM_CHECK_EQ(dim, 3);
                            std::copy(w.data(), w.data() + dim, x + idx);
                            idx += dim;
                            break;
                        }
                        case 2: {
                            const Scaleprior::dParam & w = n->scaleprior()->gradient();
                            const int dim = static_cast<int>(w.size());
                            RGM_CHECK_EQ(dim, 3);
                            std::copy(w.data(), w.data() + dim, x + idx);
                            idx += dim;
                            break;
                        }
                        default: {
                            RGM_LOG(error, "The grammar doesnt have what you want.");
                            break;
                        }
                    }
                }

                if(n->bias() != NULL) {
                    const double  w = ((which == 0) ? n->bias()->w() :
                                       (which == 1 ?
                                        n->bias()->lowerBound() :
                                        n->bias()->gradient()));
                    x[idx++] =  w;
                }

                break;
            }
        } // switch
    } // for i
}

template<int Dimension>
void AOGrammar::initGradient() {
    vector<Node *> & DFS(getNodeDFS());

    for(int i = 0; i < DFS.size(); ++i) {
        Node * n = DFS[i];
        if(isLRFlip() && sharedLRFlip() && n->isLRFlip())
            continue;

        if(n->appearance() != NULL) {
            int wd = n->appearance()->w().cols();
            int ht = n->appearance()->w().rows();
            n->getAppearance()->getGradient() =
                Appearance::dParam::Constant(ht, wd, dCell::Zero());
        }

        if(n->bias() != NULL) {
            n->getBias()->getGradient() = 0;
        }

        if(n->deformation() != NULL) {
            n->getDeformation()->getGradient().setZero();
        }

        if(n->scaleprior() != NULL) {
            n->getScaleprior()->getGradient().setZero();
        }
    }

    //    for ( int i = 0; i < appearanceSet().size(); ++i ) {
    //        int wd = appearanceSet()[i]->w().cols();
    //        int ht = appearanceSet()[i]->w().rows();
    //        getAppearanceSet()[i]->getGradient() =
    //                Appearance::dParam::Constant(ht, wd, Appearance::dCell::Zero());
    //    }

    //    for ( int i = 0; i < biasSet().size(); ++i ) {
    //        getBiasSet()[i]->getGradient() = 0;
    //    }

    //    for ( int i = 0; i < deformationSet().size(); ++i ) {
    //        getDeformationSet()[i]->getGradient().setZero();
    //    }

    //    for ( int i = 0; i < scalepriorSet().size(); ++i ) {
    //        getScalepriorSet()[i]->getGradient().setZero();
    //    }
}

template<int Dimension>
void AOGrammar::updateGradient(const ParseTree & pt, double mult) {
    if(pt.empty()) {
        return;
    }

    RGM_CHECK_EQ(pt.grammar(), this);

    for(int i = 0; i < pt.nodeSet().size(); ++i) {
        int idxG = pt.nodeSet()[i]->idx()[PtNode::IDX_G];
        Node * node = findNode(idxG);
        RGM_CHECK_NOTNULL(node);

        int idx = pt.nodeSet()[i]->idx()[PtNode::IDX_BIAS];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->bias());
            if(node->bias()->learningStatus() != 0) {
                node->getBias()->getGradient() += mult * pt.biasSet()[idx];
            }
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_DEF];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->deformation());
            if(node->deformation()->learningStatus() != 0) {
                node->getDeformation()->getGradient() +=
                    mult *
                    (*(pt.deformationSet()[idx])).template cast<double>();
            }
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_SCALEPRIOR];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->scaleprior());
            if(node->scaleprior()->learningStatus() != 0) {
                node->getScaleprior()->getGradient() +=
                    mult *
                    (*(pt.scalepriorSet()[idx])).template cast<double>();
            }
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_APP];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->appearance());
            if(node->appearance()->learningStatus() != 0) {
                FeaturePyr::dMap(node->getAppearance()->getGradient()) +=
                    mult *
                    FeaturePyr::Map(
                        *(pt.appearanceSet()[idx])).template cast<double>();
            }
        }

    } // for i
}

template<int Dimension>
double AOGrammar::dot(const ParseTree & pt) {
    if(pt.empty()) {
        return 0;
    }

    RGM_CHECK_EQ(pt.grammar(), this);

    double score = 0;
    for(int i = 0; i < pt.nodeSet().size(); ++i) {
        int idxG = pt.nodeSet()[i]->idx()[PtNode::IDX_G];
        const Node * node = findNode(idxG);
        RGM_CHECK_NOTNULL(node);
        if ( !node->onOff() ) {
            int debug = 1;
        }
        RGM_CHECK(node->onOff(), error);

        int idx = pt.nodeSet()[i]->idx()[PtNode::IDX_BIAS];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->bias());
            score += node->bias()->w() * pt.biasSet()[idx];
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_DEF];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->deformation());
            score += node->deformation()->w().cwiseProduct(
                         *pt.deformationSet()[idx]).sum();
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_SCALEPRIOR];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->scaleprior());
            score += node->scaleprior()->w().cwiseProduct(
                         *pt.scalepriorSet()[idx]).sum();
        }

        idx = pt.nodeSet()[i]->idx()[PtNode::IDX_APP];
        if(idx != -1) {
            RGM_CHECK_NOTNULL(node->appearance());
            score += FeaturePyr::Map(node->appearance()->w()).cwiseProduct(
                         FeaturePyr::Map(*pt.appearanceSet()[idx])).sum();
        }
    } // for i

    return score;
}

template<int Dimension>
double AOGrammar::computeNorm(bool hasGrad) {
    if(regMethod() == REG_L2) {
        return computeL2Norm(hasGrad);
    } else if(regMethod() == REG_MAX &&
              type() == STARMIXTURE) {
        return computeMaxNorm(hasGrad);
    } else {
        RGM_LOG(error, "Wrong type of regularization");
        return std::numeric_limits<double>::quiet_NaN();
    }
}

template<int Dimension>
double AOGrammar::computeL2Norm(bool hasGrad) {
    double n = 0;

    /*
    for ( int i = 0; i < appearanceSet().size(); ++i ) {
        Scalar regMult    = appearanceSet()[i]->getRegularizationCost();
        Scalar learnMult  = appearanceSet()[i]->getLearningStatus() == 0 ? 0 : 1;

        if ( regMult != 0 ) {
            n += (FeaturePyr::Map(appearanceSet()[i]->w())).squaredNorm() * regMult;

            if ( learnMult != 0 && hasGrad) {
                FeaturePyr::dMap( getAppearanceSet()[i]->getGradient() ) +=
                        FeaturePyr::Map( appearanceSet()[i]->w() ).template cast<double>() * regMult * learnMult;
            }
        }
    }

    for ( int i = 0; i < biasSet().size(); ++i ) {
        Scalar regMult    = biasSet()[i]->getRegularizationCost();
        Scalar learnMult  = biasSet()[i]->getLearningStatus() == 0 ? 0 : 1;

        if ( regMult != 0 ) {
            n += biasSet()[i]->w() * biasSet()[i]->w() *  regMult;

            if ( learnMult != 0  && hasGrad) {
                getBiasSet()[i]->getGradient() += biasSet()[i]->w() * regMult * learnMult;
            }
        }
    }

    for ( int i = 0; i < deformationSet().size(); ++i ) {
        Scalar regMult    = deformationSet()[i]->getRegularizationCost();
        Scalar learnMult  = deformationSet()[i]->getLearningStatus() == 0 ? 0 : 1;

        if ( regMult != 0 ) {
            n += deformationSet()[i]->w().squaredNorm() * regMult;

            if ( learnMult != 0  && hasGrad) {
                getDeformationSet()[i]->getGradient() +=
                        deformationSet()[i]->w().template cast<double>() * regMult * learnMult;
            }
        }
    }

    for ( int i = 0; i < scalepriorSet().size(); ++i ) {
        Scalar regMult    = scalepriorSet()[i]->getRegularizationCost();
        Scalar learnMult  = scalepriorSet()[i]->getLearningStatus() == 0 ? 0 : 1;

        if ( regMult != 0 ) {
            n += scalepriorSet()[i]->w().squaredNorm() * regMult;

            if ( learnMult != 0  && hasGrad) {
                getScalepriorSet()[i]->getGradient() +=
                        scalepriorSet()[i]->w().template cast<double>() * regMult * learnMult;
            }
        }
    }
    */
    int numComp = getCompNodeDFS().size();

    for(int c = 0; c < numComp; ++c) {
        vector<Node * > & DFS = getCompNodeDFS()[c];

        Scalar val = 0;

        for(int i = 0; i < DFS.size(); ++i) {
            Node * node = DFS[i];

            if(isLRFlip() && sharedLRFlip() && node->isLRFlip())
                continue;

            nodeType t = node->type();
            switch(t) {
                case T_NODE: {
                    const AppParam & w = node->appearance()->w();
                    Scalar regMult = node->appearance()->regularizationCost();

                    if(regMult != 0) {
                        val += (FeaturePyr::Map(w)).squaredNorm() * regMult;

                        Scalar learnMult = node->appearance()->learningStatus() == 0
                                           ? 0 : 1;
                        if(learnMult != 0 && hasGrad) {
                            FeaturePyr::dMap(node->getAppearance()->getGradient())
                            += FeaturePyr::Map(w).template cast<double>()
                               * regMult * learnMult;
                        }
                    }

                    break;
                }
                case AND_NODE: {
                    if(node->deformation() != NULL) {
                        const Deformation::Param & w = node->deformation()->w();
                        Scalar regMult = node->deformation()->regularizationCost();

                        if(regMult != 0) {
                            val += w.squaredNorm() * regMult;

                            Scalar learnMult =
                                node->deformation()->learningStatus() == 0 ?
                                0 : 1;
                            if(learnMult != 0  && hasGrad) {
                                node->getDeformation()->getGradient() +=
                                    w.template cast<double>()
                                    * regMult * learnMult;
                            }
                        }
                    }

                    if(node->scaleprior() != NULL) {
                        const Scaleprior::Param & w = node->scaleprior()->w();
                        Scalar regMult = node->scaleprior()->regularizationCost();

                        if(regMult != 0) {
                            val += w.squaredNorm() * regMult;
                            Scalar learnMult = node->scaleprior()->learningStatus() == 0 ?
                                               0 : 1;
                            if(learnMult != 0  && hasGrad) {
                                node->getScaleprior()->getGradient() +=
                                    w.template cast<double>()
                                    * regMult * learnMult;
                            }
                        }
                    }

                    if(node->bias() != NULL) {
                        const Scalar & w = node->bias()->w();
                        Scalar regMult = node->bias()->regularizationCost();

                        if(regMult != 0) {
                            val +=  w * regMult;
                            Scalar learnMult = node->bias()->learningStatus() == 0 ?
                                               0 : 1;
                            if(learnMult != 0  && hasGrad) {
                                node->getBias()->getGradient() +=
                                    w * regMult * learnMult;
                            }
                        }
                    }

                    break;
                }
            } // switch

            n += val;
        } // for i
    }

    return n * 0.5;
}

template<int Dimension>
double AOGrammar::computeMaxNorm(bool hasGrad) {
    /// Softmax parameters
    /// softmax(x_1,...,x_i) = 1/beta * log[sum_i[exp(beta*x_i)]]
    const Scalar beta = 1000.0F;
    const Scalar inv_beta = 1.0F / beta;

    int numComp = getCompNodeDFS().size();
    vector<bool> isValid(numComp, false);

    double maxHnrms2 = -std::numeric_limits<double>::infinity();

    vector<double> hnrms2(numComp, 0);

    for(int c = 0; c < numComp; ++c) {
        const vector<Node * > & DFS = compNodeDFS()[c];
        if(isLRFlip() && sharedLRFlip() && DFS[0]->isLRFlip())
            continue;

        isValid[c] = true;
        Scalar val = 0;
        for(int i = 0; i < DFS.size(); ++i) {
            const Node * n = DFS[i];
            nodeType t = n->type();
            switch(t) {
                case T_NODE: {
                    const AppParam & w = n->appearance()->w();
                    Scalar regMult = n->appearance()->regularizationCost();

                    if(regMult != 0) {
                        val += (FeaturePyr::Map(w)).squaredNorm() * regMult;
                    }

                    break;
                }
                case AND_NODE: {
                    if(n->deformation() != NULL) {
                        const Deformation::Param & w = n->deformation()->w();
                        Scalar regMult = n->deformation()->regularizationCost();

                        if(regMult != 0) {
                            val += w.squaredNorm() * regMult;
                        }
                    }

                    if(n->scaleprior() != NULL) {
                        const Scaleprior::Param & w = n->scaleprior()->w();
                        Scalar regMult = n->scaleprior()->regularizationCost();

                        if(regMult != 0) {
                            val += w.squaredNorm() * regMult;
                        }
                    }

                    if(n->bias() != NULL) {
                        const Scalar & w = n->bias()->w();
                        Scalar regMult = n->bias()->regularizationCost();

                        if(regMult != 0) {
                            val +=  w * regMult;
                        }
                    }

                    break;
                }
            } // switch
        } // for i

        val *= 0.5;
        hnrms2[c] = val;
        if(val > maxHnrms2) {
            maxHnrms2 = val;
        }
    } // for c


    vector<double> gradientRegCost(numComp, 0);

    double Z = 0;
    for(int c = 0; c < numComp; c++) {
        if(!isValid[c])            continue;

        double a = exp(beta * (hnrms2[c] - maxHnrms2));
        gradientRegCost[c] = a;
        Z += a;
    }
    double inv_Z = 1.0 / Z;

    if(hasGrad) {
        for(int c = 0; c < numComp; ++c) {
            if(!isValid[c])
                continue;

            vector<Node *> & DFS = getCompNodeDFS()[c];

            double mult =  gradientRegCost[c] * inv_Z;
            if(mult == 0.0) {
                continue;
            }

            for(int i = 0; i < DFS.size(); ++i) {
                Node * n = DFS[i];
                nodeType t = n->type();

                switch(t) {
                    case AND_NODE: {
                        if(n->bias() != NULL) {
                            Scalar regMult    = n->bias()->regularizationCost();
                            Scalar learnMult  = n->bias()->learningStatus();

                            if(regMult != 0 && learnMult != 0) {
                                n->getBias()->getGradient() += n->bias()->w() *
                                                               regMult * mult;
                            }
                        }

                        if(n->deformation() != NULL) {
                            Scalar regMult    = n->deformation()->regularizationCost();
                            Scalar learnMult  = n->deformation()->learningStatus();

                            if(regMult != 0 && learnMult != 0) {
                                n->getDeformation()->getGradient() +=
                                    n->deformation()->w().template cast<double>()
                                    * regMult * mult;
                            }
                        }

                        if(n->scaleprior() != NULL) {
                            Scalar regMult    = n->scaleprior()->regularizationCost();
                            Scalar learnMult  = n->scaleprior()->learningStatus();

                            if(regMult != 0 && learnMult != 0) {
                                n->getScaleprior()->getGradient() +=
                                    n->scaleprior()->w().template cast<double>()
                                    * regMult * mult;
                            }
                        }

                        break;
                    }
                    case T_NODE: {
                        Scalar regMult    = n->appearance()->regularizationCost();
                        Scalar learnMult  = n->appearance()->learningStatus();

                        if(regMult != 0 && learnMult != 0) {
                            FeaturePyr::dMap(n->getAppearance()->getGradient()) +=
                                FeaturePyr::Map(n->appearance()->w()).template cast<double>() * regMult * mult;
                        }

                        break;
                    }
                }
            } // for i
        } // for c
    }

    double objVal = maxHnrms2 + inv_beta * log(Z);

    return objVal;
}

template<int Dimension>
Scalar AOGrammar::computeMaxPtMemory() {
//    traceCompNodeDFSandBFS(false);

    vector<Node *> & DFS(getNodeDFS());
    std::map<Node *, Scalar> nodeMemory;

    Scalar norm = static_cast<Scalar>(1.0F) / (1024 * 1024);
    Scalar sz = sizeof(Scalar) * norm;

    for(int i = 0; i < DFS.size(); ++i) {
        Node * n = DFS[i];
        if(isLRFlip() && sharedLRFlip() && n->isLRFlip())
            continue;

        Scalar m = 0;
        switch(n->type()) {
            case T_NODE: {
                const AppParam & w(n->appearance()->w());
                m += w.size() * Dimension * sz;
                break;
            }
            case AND_NODE: {
                if(n->deformation() != NULL) {
                    m += 4 * sz;
                }

                if(n->scaleprior() != NULL) {
                    m += 3 * sz;
                }

                if(n->bias() != NULL) {
                    m += sz;
                }
                for(int j = 0; j < n->outEdges().size(); ++j) {
                    m += nodeMemory[n->getOutEdges()[j]->getToNode()];
                }
                m += n->outEdges().size() * sizeof(PtEdge) * norm;
                break;
            }
            case OR_NODE: {
                for(int j = 0; j < n->outEdges().size(); ++j) {
                    m = std::max<Scalar>(m,
                                         nodeMemory[n->getOutEdges()[j]->getToNode()]);
                }
                m += sizeof(PtEdge) * norm;
                break;
            }
        }
        m += sizeof(PtNode) * norm;
        m += sizeof(ParseInfo) * norm;

        nodeMemory.insert(std::make_pair(n, m));
    }

    Scalar totalM = sizeof(ParseTree) * norm;
    totalM += nodeMemory[getRootNode()] + sizeof(PtStates) * norm;

//    traceCompNodeDFSandBFS();

    return totalM;
}


template<int Dimension>
int AOGrammar::idxNode(const Node * node) const {
    if(node == NULL) {
        return -1;
    }

    for(int i = 0; i < nodeSet().size(); ++i) {
        if(node == nodeSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxNodeAsChild(const Node * node) const {
    if(node == NULL || node->inEdges().size() != 1) {
        return -1;
    }

    const Node * parent = node->inEdges()[0]->fromNode();
    for(int i = 0; i < parent->outEdges().size(); ++i) {
        if(node == parent->outEdges()[i]->toNode()) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxParentAndNodeOfTermNode(const Node * tnode) const {
    if(tnode == NULL || tnode->type() != T_NODE) {
        return -1;
    }

    // goes up 3 layes: tnode -> and-node -> or-node -> obj and-node
    const Node * obj =
        tnode->inEdges()[0]->fromNode()->inEdges()[0]->fromNode()->inEdges()[0]->fromNode();

    return idxNode(obj);
}

template<int Dimension>
int AOGrammar::idxEdge(const Edge * edge) const {
    if(edge == NULL) {
        return -1;
    }

    for(int i = 0; i < edgeSet().size(); ++i) {
        if(edge == edgeSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxAppearance(const Appearance * app) const {
    if(app == NULL) {
        return -1;
    }

    for(int i = 0; i < appearanceSet().size(); ++i) {
        if(app == appearanceSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxBias(const Bias * b) const {
    if(b == NULL) {
        return -1;
    }

    for(int i = 0; i < biasSet().size(); ++i) {
        if(b == biasSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxDeformation(const Deformation * def) const {
    if(def == NULL) {
        return -1;
    }

    for(int i = 0; i < deformationSet().size(); ++i) {
        if(def == deformationSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxScaleprior(const Scaleprior * scale) const {
    if(scale == NULL) {
        return -1;
    }

    for(int i = 0; i < scalepriorSet().size(); ++i) {
        if(scale == scalepriorSet()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
int AOGrammar::idxFilter(const Filter  * filter) const {
    if(filter == NULL) {
        return -1;
    }

    for(int i = 0; i < cachedFilters().size(); ++i) {
        if(filter == cachedFilters()[i]) {
            return i;
        }
    }

    return -1;
}

template<int Dimension>
Node * AOGrammar::findNode(int idx) {
    if(idx < 0 || idx >= nodeSet().size()) {
        return NULL;
    }

    return getNodeSet()[idx];
}

template<int Dimension>
const Node * AOGrammar::findNodeConst(int idx) const {
    if(idx < 0 || idx >= nodeSet().size()) {
        return NULL;
    }

    return nodeSet()[idx];
}

template<int Dimension>
Edge * AOGrammar::findEdge(int idx) {
    if(idx < 0 || idx >= edgeSet().size()) {
        return NULL;
    }

    return getEdgeSet()[idx];
}

template<int Dimension>
Appearance * AOGrammar::findAppearance(int idx) {
    if(idx < 0 || idx >= appearanceSet().size()) {
        return NULL;
    }

    return getAppearanceSet()[idx];
}

template<int Dimension>
Bias * AOGrammar::findBias(int idx) {
    if(idx < 0 || idx >= biasSet().size()) {
        return NULL;
    }

    return getBiasSet()[idx];
}

template<int Dimension>
Deformation * AOGrammar::findDeformation(int idx) {
    if(idx < 0 || idx >= deformationSet().size()) {
        return NULL;
    }

    return getDeformationSet()[idx];
}

template<int Dimension>
Scaleprior * AOGrammar::findScaleprior(int idx) {
    if(idx < 0 || idx >= scalepriorSet().size()) {
        return NULL;
    }

    return getScalepriorSet()[idx];
}

template<int Dimension>
typename AOGrammar::Filter * AOGrammar::findCachedFilter(int idx) {
    if(idx < 0 || idx >= cachedFilters().size()) {
        return NULL;
    }

    return getCachedFilters()[idx];
}

template<int Dimension>
string AOGrammar::visualize(const string & saveDir, string saveName,
                            ParseTree * pt, bool createApp,
                            bool onlyOnNodes,
                            vector<bool> * nodeOnOffHistory,
                            vector<bool> * edgeOnOffHistory,
                            string extraImgFile) {

    string modelName = name() + "_" + year();
    boost::erase_all(modelName, ".");
    string strDir = saveDir + FILESEP + modelName + "_vis" + FILESEP;

    FileUtil::VerifyDirectoryExists(strDir);

    if(createApp) {
        // visualize all T-nodes
        pictureTNodes(strDir, onlyOnNodes);

        // deformations
        pictureDeformation(strDir, onlyOnNodes);
    }

    if(saveName.empty())
        return strDir;

    string dotfile = strDir + modelName + ".dot";
    std::ofstream ofs(dotfile.c_str(), std::ios::out);
    if(!ofs.is_open()) {
        RGM_LOG(error, "Can not write file " + dotfile);
        return strDir;
    }

    // write .dot file for graphviz
    ofs << "digraph " << modelName << "{\n "
        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
        << "<TR><TD><br/><font point-size=\"20\">"
        << modelName << "</font></TD></TR></TABLE>>;\n";

    ofs << "pack=true;\n overlap=false;\n labelloc=t;\n center=true;\n";

    // write the legend

    /*float wd = 2.0F;

    fprintf(f,  "subgraph legend {\n");

    //fprintf(f, "style=filled; color=lightgrey;\n");

    fprintf(f, "ORNode [shape=ellipse, style=bold, color=green, label=\"\"];\n");
    fprintf(f, "OR [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "OR-node", wd);

    fprintf(f, "ANDNode [shape=ellipse, style=filled, color=blue, label=\"\"];\n");
    fprintf(f, "AND [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "AND-node", wd);

    fprintf(f, "TNode [shape=box, style=bold, color=red, label=\"\"];\n");
    fprintf(f, "T [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "TERMINAL-node", wd);

    fprintf(f, "sFromNode [shape=ellipse, style=bold, color=green, label=\"\"];\n");
    fprintf(f, "sToNode [shape=ellipse, style=filled, color=blue, label=\"\"];\n");
    fprintf(f, "edge [style=bold, color=green];\n");
    fprintf(f, "sFromNode -> sToNode;\n");
    fprintf(f, "Switching [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "SWITCHING-edge", wd);

    fprintf(f, "cFromNode [shape=ellipse,  style=filled, color=blue, label=\"\"];\n");
    fprintf(f, "cToNode [shape=ellipse,style=bold, color=green, label=\"\"];\n");
    fprintf(f, "edge [style=bold, color=blue];\n");
    fprintf(f, "cFromNode -> cToNode;\n");
    fprintf(f, "Composition [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "COMPOSITION-edge", wd);

    fprintf(f, "dFromNode [shape=ellipse,  style=filled, color=blue, label=\"\"];\n");
    fprintf(f, "dToNode [shape=box,style=bold, color=red, label=\"\"];\n");
    fprintf(f, "edge [style=bold, color=red];\n");
    fprintf(f, "dFromNode -> dToNode;\n");
    fprintf(f, "Deformation [shape=plaintext, style=solid, label=\"%s\"\r, width=%.1f];\n", "DEFORMATION-edge", wd);

    fprintf(f, "{ rank=source; rankdir=LR; OR AND T}\n");
    fprintf(f, "{ rank=source; rankdir=LR; Switching Composition Deformation}\n");

    fprintf( f, "};\n"); */

    const string imgExt(".png");
    const string textlabel("\"\"");
    const int bs = 20;
    const int padding = 2;
    const int zeroPaddingNum = 5;
    int showWd = 100;
    int showHt = 100;
    string strFixed = "false";

    std::map<Node *, string> nodeToImg;

    // Write nodes using DFS
    traceDFSandBFS(!onlyOnNodes);

    for(int i = 0; i < nodeDFS().size(); ++i) {
        Node * curNode = getNodeDFS()[i];
        nodeType t = curNode->type();
        int nodeIdx = idxNode(curNode);

        int flip = static_cast<int>(curNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        bool nodeOnOff = curNode->onOff() ||
                         (nodeOnOffHistory != NULL &&
                          (*nodeOnOffHistory)[idxNode(curNode)]);

        if(onlyOnNodes && !nodeOnOff)
            continue;

        switch(t) {
            case T_NODE: {
                int appIdx = idxAppearance(curNode->appearance());
                string strApp = NumToString_<int>(appIdx, zeroPaddingNum);
                string strColor = nodeOnOff ? "red" : "grey";
                string imgFile = strDir + "AppTemplate_" + strApp + "_" +
                                 strFlip + imgExt;
                ofs << "node" << nodeIdx
                    << "[shape=box, style=bold, color=" << strColor << ", "
                    << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                    << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
                    << "\" fixedsize=\"" << strFixed << "\">"
                    << "<IMG SRC=\"" << imgFile << "\"/></TD></TR>"
                    //                << "<TR><TD><br/><font point-size=\"12\">"
                    //                << nodeIdx << "_" << strApp
                    //                << "</font></TD></TR>"
                    << "</TABLE>>"
                    << "];\n";
                nodeToImg.insert(std::make_pair(curNode, imgFile));
                break;
            }
            case AND_NODE: {
                string strColor = nodeOnOff ? "blue" : "grey";
                if(curNode->outEdges().size() == 1) {
                    if(curNode->deformation() != NULL) {
                        int defIdx = idxDeformation(curNode->deformation());
                        string strDef = NumToString_<int>(defIdx, zeroPaddingNum);
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "

                               << "label=" << textlabel


//                            << "label=<<TABLE border=\"0\" cellborder=\"0\">"
//                            << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
//                            << "\" fixedsize=\"" << strFixed << "\">"
                            //<< "<IMG SRC=\""
                            //<< strDir << "Deformation_" << strDef << "_" << strFlip << imgExt << "\"/>"
//                            << "</TD></TR>"
                            //                    << "<TR><TD><br/><font point-size=\"12\">"
                            //                    << textlabel
                            //                    << "</font></TD></TR>"
//                            << "</TABLE>>"
                            << "];\n";

                    } else {
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=" << textlabel << "];\n";
                    }
                    Node * ch = curNode->getOutEdges()[0]->getToNode();
                    if(nodeToImg.find(ch) != nodeToImg.end()) {
                        nodeToImg.insert(std::make_pair(curNode, nodeToImg[ch]));
                    }
                } else {
                    bool isValid = true;
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        if(nodeToImg.find(ch) == nodeToImg.end()) {
                            isValid = false;
                            break;
                        }
                    }
                    if(!isValid) {
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=" << textlabel << "];\n";
                        break;
                    }

                    int ds = 0;
                    vector<Mat> imgs(curNode->outEdges().size());
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        ds = std::max<int>(ds, ch->anchor()[2]);
                        imgs[j] = cv::imread(nodeToImg[ch], cv::IMREAD_UNCHANGED);
                    }

                    int factor = ds == 1 ? 2 : 1;
                    int wd = curNode->detectWindow().width() * factor;
                    int ht = curNode->detectWindow().height() * factor;
                    string strNode = NumToString_<int>(nodeIdx, zeroPaddingNum);
                    string saveName = strDir + "AndNode_" + strNode + "_" +
                                      strFlip  + imgExt;

                    Mat img(ht * bs + 2 * padding, wd * bs + 2 * padding,
                            imgs[0].type(), cv::Scalar::all(100));
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        if(ch->detectWindow().width() ==
                                curNode->detectWindow().width() &&
                                ch->detectWindow().height() ==
                                curNode->detectWindow().height()) {
                            cv::resize(imgs[j], img, cv::Size(img.cols, img.rows));
                        } else {
                            const Anchor & anchor(ch->anchor());
                            imgs[j].copyTo(img(cv::Rect(anchor(0)*bs, anchor(1)*bs,
                                                        imgs[j].cols, imgs[j].rows)));
                        }
                    }

                    cv::imwrite(saveName, img);
                    nodeToImg.insert(std::make_pair(curNode, saveName));

                    ofs << "node" << nodeIdx
                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                        << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
                        << "\" fixedsize=\"" << strFixed << "\">"
                        << "<IMG SRC=\"" << saveName << "\"/></TD></TR>"
                        //                        << "<TR><TD><br/><font point-size=\"12\">"
                        //                        << textlabel
                        //                        << "</font></TD></TR>"
                        << "</TABLE>>"
                        << "];\n";
                }
                break;

                //            if ( curNode->deformation() != NULL ) {
                //                int defIdx = idxDeformation(curNode->deformation());
                //                string strDef = NumToString_<int>(defIdx, zeroPaddingNum);
                //                ofs << "node" << nodeIdx
                //                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                    << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                //                    << "<TR><TD width=\"" << showWd <<"\" height=\"" << showHt
                //                    << "\" fixedsize=\"" << strFixed <<"\">"
                //                    << "<IMG SRC=\""
                //                    << strDir << "Deformation_" << strDef << "_" << strFlip << imgExt
                //                    << "\"/></TD></TR>"
                ////                    << "<TR><TD><br/><font point-size=\"12\">"
                ////                    << textlabel
                ////                    << "</font></TD></TR>"
                //                    << "</TABLE>>"
                //                    <<"];\n";

                //            } else if (curNode->offset() != NULL) {
                //                Mat rootApp;
                //                vector<std::pair<Mat, Anchor> > parts;

                //                for ( int j = 0; j < curNode->outEdges().size(); ++j ) {
                //                    const Node * to = curNode->outEdges()[j]->toNode();

                //                    const Node * T = to;
                //                    while ( T->type() != T_NODE) {
                //                        T = T->outEdges()[0]->toNode();
                //                    }

                //                    int appIdx = idxAppearance(T->appearance());
                //                    string strApp = NumToString_<int>(appIdx, zeroPaddingNum);
                //                    string imgFile = strDir + "AppTemplate_" + strApp + "_" + strFlip + imgExt;

                //                    if ( to->anchor()(2) == 0 ) {
                //                        rootApp = cv::imread(imgFile, cv::IMREAD_UNCHANGED);
                //                        //assert(!rootApp.empty());
                //                    } else {
                //                        parts.push_back( std::make_pair(cv::imread(imgFile, cv::IMREAD_UNCHANGED), to->anchor()) );
                //                    }
                //                } // for j

                //                if ( parts.size() > 0 ) {
                //                    Mat rootxApp(rootApp.rows * 2, rootApp.cols*2, rootApp.type());
                //                    cv::resize(rootApp, rootxApp, rootxApp.size(), 0, 0, cv::INTER_CUBIC);

                //                    for ( int j = 0; j < parts.size(); ++j ) {
                //                        int x = parts[j].second(0) * bs + padding;
                //                        int y = parts[j].second(1) * bs + padding;
                //                        parts[j].first.copyTo(rootxApp(cv::Rect(x, y, parts[j].first.cols, parts[j].first.rows)));

                //                        /*cv::imshow("debug", rootxApp);
                //                    cv::waitKey(0);*/
                //                    }

                //                    int nodeIdx = idxNode(curNode);
                //                    string strNode = NumToString_<int>(nodeIdx, zeroPaddingNum);

                //                    string saveName = strDir + "Component_" + strNode + "_" + strFlip  + imgExt;
                //                    cv::imwrite(saveName, rootxApp);
                //                    ofs << "node" << nodeIdx
                //                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                //                        << "<TR><TD width=\"" << showWd <<"\" height=\"" << showHt
                //                        << "\" fixedsize=\"" << strFixed <<"\">"
                //                        << "<IMG SRC=\""
                //                        << strDir << "Component_" << strNode << "_" << strFlip << imgExt
                //                        << "\"/></TD></TR>"
                ////                        << "<TR><TD><br/><font point-size=\"12\">"
                ////                        << textlabel
                ////                        << "</font></TD></TR>"
                //                        << "</TABLE>>"
                //                        <<"];\n";
                //                } else {
                //                    ofs << "node" << nodeIdx
                //                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                        << "label=" << textlabel <<"];\n";
                //                }
                //            } else {
                //                ofs << "node" << nodeIdx
                //                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                    << "label=" << textlabel <<"];\n";
                //            }
                //            break;
            }
            case OR_NODE: {
                string strColor = nodeOnOff ? "green" : "grey";
                if(curNode == rootNode() && FileUtil::exists(extraImgFile)) {
                    ofs << "node" << nodeIdx
                        << "[shape=ellipse, style=bold, color=" << strColor << ", "
                        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                        << "<TR><TD>"
                        << "<IMG SRC=\""
                        << extraImgFile
                        << "\"/></TD></TR>"
                        << "</TABLE>>"
                        << "];\n";

                } else {
                    ofs << "node" << nodeIdx
                        << "[shape=ellipse, style=bold, color=" << strColor << ", "
                        << "label=" << textlabel << "];\n";
                }

                if(curNode->outEdges().size() == 1) {
                    Node * ch = curNode->getOutEdges()[0]->getToNode();
                    if(nodeToImg.find(ch) != nodeToImg.end()) {
                        nodeToImg.insert(std::make_pair(curNode, nodeToImg[ch]));
                    }
                }

                break;
            }
        } // switch
    } // for i

    // Write edges using BFS
    for(int i = 0; i < nodeBFS().size(); ++i) {
        const Node * fromNode = nodeBFS()[i];
        int idxFrom = idxNode(fromNode);

        const vector<Edge *> & outEdges = fromNode->outEdges();
        for(int j = 0; j < outEdges.size(); ++j) {
            const Edge * curEdge = outEdges[j];
            edgeType t = curEdge->type();
            const Node * toNode = curEdge->toNode();
            int idxTo = idxNode(toNode);

            bool edgeOnOff = (curEdge->onOff() || (edgeOnOffHistory != NULL &&
                                                   (*edgeOnOffHistory)[idxEdge(curEdge)]));

            if(!edgeOnOff) //onlyOnNodes &&
                continue;

            switch(t) {
                case SWITCHING: {
                    string strColor =  edgeOnOff ? "green" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case COMPOSITION: {
                    string strColor = edgeOnOff ? "blue" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case DEFORMATION: {
                    string strColor = edgeOnOff ? "red" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case TERMINATION: {
                    string strColor = edgeOnOff ? "black" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
            }
            ofs << "node" << idxFrom << " -> node" << idxTo << ";\n";
        } // for j
    } // for i

    if(pt != NULL) {
        for(int i = 0; i < pt->edgeSet().size(); ++i) {
            const Edge * curEdge = findEdge(
                                       pt->edgeSet()[i]->idx()[PtEdge::IDX_G]);
            RGM_CHECK_NOTNULL(curEdge);
            RGM_CHECK(curEdge->onOff(), error);

            int idxFrom = idxNode(curEdge->fromNode());
            int idxTo = idxNode(curEdge->toNode());

            ofs << "edge [style=bold, color=red, penwidth=8., arrowsize=2., weight=6.];\n";
            ofs << "node" << idxFrom << " -> node" << idxTo << ";\n";
        }
    }

    ofs << "}\n";
    ofs.close();

    traceDFSandBFS();

    /// Use GraphViz
    if(!saveName.empty()) {
        strDir = saveDir;
        modelName = saveName;
    }
    string cmd = "dot -Tpdf " + dotfile + " -o " + strDir + modelName + ".pdf";
    std::system(cmd.c_str());

    cmd = "dot -Tpng " + dotfile + " -o " + strDir + modelName +
          ".png"; //-Gsize=15,10\! -Gdpi=100
    std::system(cmd.c_str());

    return strDir;
}

template<int Dimension>
string AOGrammar::visualize1(const string & saveDir, string saveName,
                             ParseTree * pt, std::map<int, cv::Mat> *ptScoreMaps,
                             bool createApp,
                             bool onlyOnNodes,
                             vector<bool> * nodeOnOffHistory,
                             vector<bool> * edgeOnOffHistory) {

    string modelName = name() + "_" + year();
    boost::erase_all(modelName, ".");
    string strDir = saveDir + FILESEP + modelName + "_vis" + FILESEP;
    //boost::filesystem::remove_all(boost::filesystem::path(strDir));
    FileUtil::VerifyDirectoryExists(strDir);

    if(createApp) {
        // visualize all T-nodes
        pictureTNodes(strDir, onlyOnNodes);

        // deformations
        pictureDeformation(strDir, onlyOnNodes);
    }

    if(saveName.empty())
        return strDir;

    string dotfile = strDir + modelName + ".dot";
    std::ofstream ofs(dotfile.c_str(), std::ios::out);
    if(!ofs.is_open()) {
        RGM_LOG(error, "Can not write file " + dotfile);
        return strDir;
    }

    // write .dot file for graphviz
    ofs << "digraph " << modelName << "{\n "
        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
        << "<TR><TD><br/><font point-size=\"20\">"
        << modelName << "</font></TD></TR></TABLE>>;\n";

    ofs << "pack=true;\n overlap=false;\n labelloc=t;\n center=true;\n";

    const string imgExt(".png");
    const string textlabel("\"\"");
    const int bs = 20;
    const int padding = 2;
    const int zeroPaddingNum = 5;
    int showWd = 100;
    int showHt = 100;
    string strFixed = "false";

    std::map<Node *, string> nodeToImg;

    // Write nodes using DFS
    traceDFSandBFS(false);

    for(int i = 0; i < nodeDFS().size(); ++i) {
        Node * curNode = getNodeDFS()[i];
        nodeType t = curNode->type();
        int nodeIdx = idxNode(curNode);

        int flip = static_cast<int>(curNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        bool nodeIsOn = curNode->onOff() ||
                        (nodeOnOffHistory != NULL &&
                         (*nodeOnOffHistory)[nodeIdx]);

        if(onlyOnNodes && !nodeIsOn)
            continue;

        switch(t) {
            case T_NODE: {
                int appIdx = idxAppearance(curNode->appearance());
                string strApp = NumToString_<int>(appIdx, zeroPaddingNum);
                string strColor = nodeIsOn ? "red" : "grey";
                string imgFile = strDir + "AppTemplate_" + strApp + "_" +
                                 strFlip + imgExt;
                ofs << "node" << nodeIdx
                    << "[shape=box, style=bold, color=" << strColor << ", "
                    << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                    << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
                    << "\" fixedsize=\"" << strFixed << "\">"
                    << "<IMG SRC=\"" << imgFile << "\"/></TD></TR>"
                    //                << "<TR><TD><br/><font point-size=\"12\">"
                    //                << nodeIdx << "_" << strApp
                    //                << "</font></TD></TR>"
                    << "</TABLE>>"
                    << "];\n";
                nodeToImg.insert(std::make_pair(curNode, imgFile));
                break;
            }
            case AND_NODE: {
                string strColor = nodeIsOn ? "blue" : "grey";
                if(curNode->outEdges().size() == 1) {
                    if(curNode->deformation() != NULL) {
                        int defIdx = idxDeformation(curNode->deformation());
                        string strDef = NumToString_<int>(defIdx, zeroPaddingNum);
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "

                            << "label=" << textlabel

//                            << "label=<<TABLE border=\"0\" cellborder=\"0\">"
//                            << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
//                            << "\" fixedsize=\"" << strFixed << "\">"
                            //<< "<IMG SRC=\""
                            //<< strDir << "Deformation_" << strDef << "_" << strFlip << imgExt <<"\"/>"
//                            << "</TD></TR>"
                            //                    << "<TR><TD><br/><font point-size=\"12\">"
                            //                    << textlabel
                            //                    << "</font></TD></TR>"
//                            << "</TABLE>>"
                            << "];\n";

                    } else {
//                        if(ptScoreMaps != NULL) {
//                            std::map<int, Mat>::iterator iter = ptScoreMaps->find(nodeIdx);
//                            if(iter != ptScoreMaps->end()) {
//                                string tmpName = strDir + "AndNodeScore_" +
//                                                 NumToString_<int>(nodeIdx, zeroPaddingNum) + imgExt;
//                                cv::imwrite(tmpName, iter->second);
//                                ofs << "node" << nodeIdx
//                                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
//                                    << "label=<<TABLE border=\"0\" cellborder=\"0\">"
//                                    << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
//                                    << "\" fixedsize=\"" << strFixed << "\">"
//                                    << "<IMG SRC=\""
//                                    << tmpName
//                                    << "\"/></TD></TR>"
//                                    //                    << "<TR><TD><br/><font point-size=\"12\">"
//                                    //                    << textlabel
//                                    //                    << "</font></TD></TR>"
//                                    << "</TABLE>>"
//                                    << "];\n";

//                            } else {
//                                ofs << "node" << nodeIdx
//                                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
//                                    << "label=" << textlabel << "];\n";
//                            }
//                        } else {
//                            ofs << "node" << nodeIdx
//                                << "[shape=ellipse, style=filled, color=" << strColor << ", "
//                                << "label=" << textlabel << "];\n";
//                        }

                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=" << textlabel << "];\n";
                    }
                    Node * ch = curNode->getOutEdges()[0]->getToNode();
                    if(nodeToImg.find(ch) != nodeToImg.end()) {
                        nodeToImg.insert(std::make_pair(curNode, nodeToImg[ch]));
                    }
                } else {
                    bool isValid = true;
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        if(nodeToImg.find(ch) == nodeToImg.end()) {
                            isValid = false;
                            break;
                        }
                    }
                    if(!isValid) {
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=" << textlabel << "];\n";
                        break;
                    }

                    int ds = 0;
                    vector<Mat> imgs(curNode->outEdges().size());
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        ds = std::max<int>(ds, ch->anchor()[2]);
                        imgs[j] = cv::imread(nodeToImg[ch], cv::IMREAD_UNCHANGED);
                    }

                    int factor = ds == 1 ? 2 : 1;
                    int wd = curNode->detectWindow().width() * factor;
                    int ht = curNode->detectWindow().height() * factor;
                    string strNode = NumToString_<int>(nodeIdx, zeroPaddingNum);
                    string saveName = strDir + "AndNode_" + strNode + "_" +
                                      strFlip  + imgExt;

                    Mat img(ht * bs + 2 * padding, wd * bs + 2 * padding,
                            imgs[0].type(), cv::Scalar::all(100));
                    for(int j = 0; j < curNode->outEdges().size(); ++j) {
                        Node * ch = curNode->getOutEdges()[j]->getToNode();
                        if(ch->detectWindow().width() ==
                                curNode->detectWindow().width() &&
                                ch->detectWindow().height() ==
                                curNode->detectWindow().height()) {
                            cv::resize(imgs[j], img, cv::Size(img.cols, img.rows));
                        } else {
                            const Anchor & anchor(ch->anchor());
                            imgs[j].copyTo(img(cv::Rect(anchor(0)*bs, anchor(1)*bs,
                                                        imgs[j].cols, imgs[j].rows)));
                        }
                    }

                    cv::imwrite(saveName, img);
                    nodeToImg.insert(std::make_pair(curNode, saveName));

                    ofs << "node" << nodeIdx
                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                        << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
                        << "\" fixedsize=\"" << strFixed << "\">"
                        << "<IMG SRC=\"" << saveName << "\"/></TD></TR>"
                        //                        << "<TR><TD><br/><font point-size=\"12\">"
                        //                        << textlabel
                        //                        << "</font></TD></TR>"
                        << "</TABLE>>"
                        << "];\n";
                }
                break;

                //            if ( curNode->deformation() != NULL ) {
                //                int defIdx = idxDeformation(curNode->deformation());
                //                string strDef = NumToString_<int>(defIdx, zeroPaddingNum);
                //                ofs << "node" << nodeIdx
                //                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                    << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                //                    << "<TR><TD width=\"" << showWd <<"\" height=\"" << showHt
                //                    << "\" fixedsize=\"" << strFixed <<"\">"
                //                    << "<IMG SRC=\""
                //                    << strDir << "Deformation_" << strDef << "_" << strFlip << imgExt
                //                    << "\"/></TD></TR>"
                ////                    << "<TR><TD><br/><font point-size=\"12\">"
                ////                    << textlabel
                ////                    << "</font></TD></TR>"
                //                    << "</TABLE>>"
                //                    <<"];\n";

                //            } else if (curNode->offset() != NULL) {
                //                Mat rootApp;
                //                vector<std::pair<Mat, Anchor> > parts;

                //                for ( int j = 0; j < curNode->outEdges().size(); ++j ) {
                //                    const Node * to = curNode->outEdges()[j]->toNode();

                //                    const Node * T = to;
                //                    while ( T->type() != T_NODE) {
                //                        T = T->outEdges()[0]->toNode();
                //                    }

                //                    int appIdx = idxAppearance(T->appearance());
                //                    string strApp = NumToString_<int>(appIdx, zeroPaddingNum);
                //                    string imgFile = strDir + "AppTemplate_" + strApp + "_" + strFlip + imgExt;

                //                    if ( to->anchor()(2) == 0 ) {
                //                        rootApp = cv::imread(imgFile, cv::IMREAD_UNCHANGED);
                //                        //assert(!rootApp.empty());
                //                    } else {
                //                        parts.push_back( std::make_pair(cv::imread(imgFile, cv::IMREAD_UNCHANGED), to->anchor()) );
                //                    }
                //                } // for j

                //                if ( parts.size() > 0 ) {
                //                    Mat rootxApp(rootApp.rows * 2, rootApp.cols*2, rootApp.type());
                //                    cv::resize(rootApp, rootxApp, rootxApp.size(), 0, 0, cv::INTER_CUBIC);

                //                    for ( int j = 0; j < parts.size(); ++j ) {
                //                        int x = parts[j].second(0) * bs + padding;
                //                        int y = parts[j].second(1) * bs + padding;
                //                        parts[j].first.copyTo(rootxApp(cv::Rect(x, y, parts[j].first.cols, parts[j].first.rows)));

                //                        /*cv::imshow("debug", rootxApp);
                //                    cv::waitKey(0);*/
                //                    }

                //                    int nodeIdx = idxNode(curNode);
                //                    string strNode = NumToString_<int>(nodeIdx, zeroPaddingNum);

                //                    string saveName = strDir + "Component_" + strNode + "_" + strFlip  + imgExt;
                //                    cv::imwrite(saveName, rootxApp);
                //                    ofs << "node" << nodeIdx
                //                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                        << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                //                        << "<TR><TD width=\"" << showWd <<"\" height=\"" << showHt
                //                        << "\" fixedsize=\"" << strFixed <<"\">"
                //                        << "<IMG SRC=\""
                //                        << strDir << "Component_" << strNode << "_" << strFlip << imgExt
                //                        << "\"/></TD></TR>"
                ////                        << "<TR><TD><br/><font point-size=\"12\">"
                ////                        << textlabel
                ////                        << "</font></TD></TR>"
                //                        << "</TABLE>>"
                //                        <<"];\n";
                //                } else {
                //                    ofs << "node" << nodeIdx
                //                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                        << "label=" << textlabel <<"];\n";
                //                }
                //            } else {
                //                ofs << "node" << nodeIdx
                //                    << "[shape=ellipse, style=filled, color=" << strColor << ", "
                //                    << "label=" << textlabel <<"];\n";
                //            }
                //            break;
            }
            case OR_NODE: {
                string strColor = nodeIsOn ? "green" : "grey";

                if(ptScoreMaps != NULL) {
                    std::map<int, Mat>::iterator iter = ptScoreMaps->find(nodeIdx);
                    if(iter != ptScoreMaps->end()) {
                        string tmpName = strDir + "OrNodeScore_" +
                                         NumToString_<int>(nodeIdx, zeroPaddingNum) + imgExt;
                        cv::imwrite(tmpName, iter->second);
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=<<TABLE border=\"0\" cellborder=\"0\">"
                            << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
                            << "\" fixedsize=\"" << strFixed << "\">"
                            << "<IMG SRC=\""
                            << tmpName
                            << "\"/></TD></TR>"
                            //                    << "<TR><TD><br/><font point-size=\"12\">"
                            //                    << textlabel
                            //                    << "</font></TD></TR>"
                            << "</TABLE>>"
                            << "];\n";

                    } else {
                        ofs << "node" << nodeIdx
                            << "[shape=ellipse, style=filled, color=" << strColor << ", "
                            << "label=" << textlabel << "];\n";
                    }
                } else {
                    ofs << "node" << nodeIdx
                        << "[shape=ellipse, style=filled, color=" << strColor << ", "
                        << "label=" << textlabel << "];\n";
                }

                if(curNode->outEdges().size() == 1) {
                    Node * ch = curNode->getOutEdges()[0]->getToNode();
                    if(nodeToImg.find(ch) != nodeToImg.end()) {
                        nodeToImg.insert(std::make_pair(curNode, nodeToImg[ch]));
                    }
                }

                break;
            }
        } // switch
    } // for i

    // Write edges using BFS
    for(int i = 0; i < nodeBFS().size(); ++i) {
        const Node * fromNode = nodeBFS()[i];
        int idxFrom = idxNode(fromNode);

        const vector<Edge *> & outEdges = fromNode->outEdges();
        for(int j = 0; j < outEdges.size(); ++j) {
            const Edge * curEdge = outEdges[j];
            edgeType t = curEdge->type();
            const Node * toNode = curEdge->toNode();
            int idxTo = idxNode(toNode);

            bool edgeIsOn = (curEdge->onOff() || (edgeOnOffHistory != NULL &&
                                                   (*edgeOnOffHistory)[idxEdge(curEdge)]));

            if(onlyOnNodes && !edgeIsOn) //
                continue;

            switch(t) {
                case SWITCHING: {
                    string strColor =  edgeIsOn ? "green" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case COMPOSITION: {
                    string strColor = edgeIsOn ? "blue" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case DEFORMATION: {
                    string strColor = edgeIsOn ? "red" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
                case TERMINATION: {
                    string strColor = edgeIsOn ? "black" : "grey";
                    ofs << "edge [style=bold, color=" << strColor << "];\n";
                    break;
                }
            }
            ofs << "node" << idxFrom << " -> node" << idxTo << ";\n";
        } // for j
    } // for i

    if(pt != NULL) {
        for(int i = 0; i < pt->edgeSet().size(); ++i) {
            const Edge * curEdge = findEdge(
                                       pt->edgeSet()[i]->idx()[PtEdge::IDX_G]);
            RGM_CHECK_NOTNULL(curEdge);
            RGM_CHECK(curEdge->onOff(), error);

            int idxFrom = idxNode(curEdge->fromNode());
            int idxTo = idxNode(curEdge->toNode());

            ofs << "edge [style=bold, color=red, penwidth=8., arrowsize=2., weight=6.];\n";
            ofs << "node" << idxFrom << " -> node" << idxTo << ";\n";
        }
    }

    ofs << "}\n";
    ofs.close();

    traceDFSandBFS();

    /// Use GraphViz
    if(!saveName.empty()) {
        strDir = saveDir;
        modelName = saveName;
    }
//    string cmd = "dot -Tpdf " + dotfile + " -o " + strDir + modelName + ".pdf";
//    std::system(cmd.c_str());

    string cmd = "dot -Tpng " + dotfile + " -o " + strDir + modelName +
          ".png"; //-Gsize=15,10\! -Gdpi=100
    std::system(cmd.c_str());

    return strDir;
}

template<int Dimension>
void AOGrammar::visualize(const string &saveDir, Mat img, string saveName) {
    if(img.empty() || gridAOG().size() == 0)
        return;

    string modelName = name() + "_" + year();
    boost::erase_all(modelName, ".");
    string strDir = saveDir + FILESEP + modelName + "_vis" + FILESEP;
    FileUtil::VerifyDirectoryExists(strDir);

    const string imgExt(".png");
    const int zeroPaddingNum = 5;

    Mat imgRoot, imgRootx;
    cv::Size szRoot, szRootx;

    szRoot.width = maxDetectWindow().width() * cellSize();
    szRoot.height = maxDetectWindow().height() * cellSize();
    cv::resize(img, imgRoot, szRoot, 0, 0, RGM_IMG_RESIZE);

    const AOGrid * g(gridAOG()[0]);

    szRootx.width = g->param().inputWindow_.width() * cellSize();
    szRootx.height = g->param().inputWindow_.height() * cellSize();
    cv::resize(img, imgRootx, szRootx, 0, 0, RGM_IMG_RESIZE);

    // create T-node images
    Mat imgApp;
    cv::Rect roi;
    for(int i = 0; i < g->nodeSet().size(); ++i) {
        int id = g->nodeSet()[i].idx_(AOGrid::Vertex::ID);
        int idx = g->nodeSet()[i].idx_(AOGrid::Vertex::ID_G);
        Node *gNode = findNode(idx);
        if(gNode->type() != T_NODE)
            continue;

        int appIdx = idxAppearance(gNode->appearance());
        string strApp = NumToString_<int>(appIdx, zeroPaddingNum);

        int flip = static_cast<int>(gNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        idx = g->nodeSet()[i].idx_(AOGrid::Vertex::ID_IN_INSTANCE_SET);
        Rectangle box(g->instanceBbox(idx));
        roi.x = box.x() * cellSize();
        roi.y = box.y() * cellSize();
        roi.width = box.width() * cellSize();
        roi.height = box.height() * cellSize();

        if(id == g->rootTermNodeId()) {
            imgApp = imgRoot;
        } else {
            imgApp = imgRootx(roi);
        }

        string saveName = strDir + string("AppTemplate_") +
                          strApp + "_" + strFlip  + imgExt;
        cv::imwrite(saveName, imgApp);
    }

    // root t-node
    string imgSaveName = strDir + string("AppTemplate_") +
                         NumToString_<int>(0, zeroPaddingNum) + "_0"  + imgExt;
    cv::imwrite(imgSaveName, imgRoot);

    string dotfile = strDir + modelName + ".dot";
    FILE *f = fopen(dotfile.c_str(), "w");
    if(f == NULL) {
        RGM_LOG(error, ("Can not write file " + dotfile));
        return;
    }

    // write .dot file for graphviz
    fprintf(f, "digraph %s {\n label=\"%s\";\n",
            modelName.c_str(), modelName.c_str());

    fprintf(f, "pack=true;\n overlap=false;\n labelloc=t;\n center=true;\n");

//    fprintf(f, "graph [autosize=false, size=\"%f,%f!\", resolution=100];\n",
//            300, 300);

    const string textlabel("\"\"");

    // Write nodes using DFS
    traceDFSandBFS(false);
    for(int i = 0; i < nodeDFS().size(); ++i) {
        const Node * curNode = nodeDFS()[i];
        nodeType t = curNode->type();
        int nodeIdx = idxNode(curNode);

        int flip = static_cast<int>(curNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        switch(t) {
            case T_NODE: {
                int appIdx = idxAppearance(curNode->appearance());
                string strApp = NumToString_<int>(appIdx, zeroPaddingNum);

                fprintf(f, "node%d [shape=box, style=bold, color=red, label=%s, labelloc=b, image=\"%sAppTemplate_%s_%s%s\"];\n",
                        nodeIdx, textlabel.c_str(), strDir.c_str(),
                        strApp.c_str(), strFlip.c_str(), imgExt.c_str());
                break;
            }
            case AND_NODE: {
                fprintf(f, "node%d [shape=ellipse, style=filled, color=blue, label=%s];\n",
                        nodeIdx, textlabel.c_str());
                break;
            }
            case OR_NODE: {
                fprintf(f, "node%d [shape=ellipse, style=bold, color=green, label=%s];\n",
                        nodeIdx, textlabel.c_str());
                break;
            }
        } // switch
    } // for i

    //fprintf(f, "nodeOR [shape=ellipse, style=bold, color=green, label=\"OR-node\"];\n");

    // Write edges using BFS
    for(int i = 0; i < nodeBFS().size(); ++i) {
        const Node * fromNode = nodeBFS()[i];
        int idxFrom = idxNode(fromNode);

        const std::vector<Edge *> & outEdges = fromNode->outEdges();
        for(int j = 0; j < outEdges.size(); ++j) {
//            if(!outEdges[j]->onOff())
//                continue;
            const Edge * curEdge = outEdges[j];
            edgeType t = curEdge->type();
            const Node * toNode = curEdge->toNode();
            int idxTo = idxNode(toNode);

            switch(t) {
                case SWITCHING: {
                    fprintf(f, "edge [style=bold, color=green];\n");
                    fprintf(f, "node%d -> node%d;\n", idxFrom, idxTo);
                    break;
                }
                case COMPOSITION: {
                    fprintf(f, "edge [style=bold, color=blue];\n");
                    fprintf(f, "node%d -> node%d;\n", idxFrom, idxTo);
                    break;
                }
                case DEFORMATION: {
                    fprintf(f, "edge [style=bold, color=red];\n");
                    fprintf(f, "node%d -> node%d;\n", idxFrom, idxTo);
                    break;
                }
                case TERMINATION: {
                    fprintf(f, "edge [style=bold, color=black];\n");
                    fprintf(f, "node%d -> node%d;\n", idxFrom, idxTo);
                    break;
                }
            }
        } // for j
    } // for i

    fprintf(f, "}");

    fclose(f);

    traceDFSandBFS();

    /// Use GraphViz
    if(!saveName.empty())
        modelName = saveName;

    string cmd = "dot -Tpdf " + dotfile + " -o " + saveDir + modelName + ".pdf";
    std::system(cmd.c_str());

    cmd = "dot -Tpng " + dotfile + " -o " + saveDir + modelName + ".png";
    std::system(cmd.c_str());

}

template<int Dimension>
void AOGrammar::pictureTNodes(const string & saveDir, bool onlyOnNodes,
                              std::map<int, cv::Mat> *ptScoreMaps) {
    const string imgExt(".png");

    const int bs = 20;
    const int padding = 2;
    const int zeroPaddingNum = 5;

    for(int i = 0; i < nodeSet().size(); ++i) {
        const Node * curNode = nodeSet()[i];
        if(curNode->type() != T_NODE ||
                (onlyOnNodes && !curNode->onOff())) {
            continue;
        }

        int nodeIdx = idxNode(curNode);
        int appIdx = idxAppearance(curNode->appearance());
        string strApp = NumToString_<int>(appIdx, zeroPaddingNum);

        int flip = static_cast<int>(curNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        // Return the contrast insensitive orientations
        cv::Mat_<Scalar> wFolded = FeaturePyr::fold(curNode->appearanceParam(),
                                                    featParam().type_);

        Mat img = OpencvUtil::pictureHOG(wFolded, bs);

        Mat imgPadding(img.rows + 2 * padding, img.cols + 2 * padding, img.type(),
                       cv::Scalar::all(128));
        img.copyTo(imgPadding(cv::Rect(padding, padding, img.cols, img.rows)));

        string saveName = saveDir + string("AppTemplate_") + strApp + "_" + strFlip  +
                          imgExt;

        Mat imgShow;

        //cv::normalize(img, imgShow(cv::Rect(padding, padding, img.cols, img.rows)), 255, 0.0, CV_MINMAX, CV_8UC1);

        //        cv::imshow("AppTemplate", imgShow);
        //        cv::waitKey(2);

        if(ptScoreMaps != NULL) {
            std::map<int, Mat>::iterator iter = ptScoreMaps->find(nodeIdx);
            if(iter == ptScoreMaps->end()) {
                imgPadding.convertTo(imgShow, CV_8UC1);
                cv::imwrite(saveName, imgShow);
            } else {
                Mat imgShow1;
                imgPadding.convertTo(imgShow1, CV_8UC1);
                cv::cvtColor(imgShow1, imgShow, CV_GRAY2BGR);
                int bigRows = std::max<int>(imgShow.rows, iter->second.rows);
                int bigCols = imgShow.cols + iter->second.cols;
                Mat bigShow(bigRows, bigCols, iter->second.type());
                bigShow = cv::Scalar::all(255);
                imgShow.copyTo(bigShow(cv::Rect(0, 0, imgShow.cols, imgShow.rows)));
                iter->second.copyTo(bigShow(cv::Rect(imgShow.cols, 0, iter->second.cols,
                                                     iter->second.rows)));
                cv::imwrite(saveName, bigShow);
            }
        } else {
            imgPadding.convertTo(imgShow, CV_8UC1);
            imgShow = 255 - imgShow;
            cv::imwrite(saveName, imgShow);
//            Mat cmap;
//            cv::applyColorMap(imgShow, cmap, cv::COLORMAP_JET);
//            cv::imwrite(saveName, cmap);
        }
    } // for i

    //    cv::destroyWindow("AppTemplate");
}

template<int Dimension>
void AOGrammar::pictureDeformation(const string & saveDir, bool onlyOnNodes,
                                   std::map<int, cv::Mat> *ptScoreMaps) {
    const string imgExt(".png");

    const int bs = 20;
    const int padding = 2;
    const int zeroPaddingNum = 5;
    const Scalar defScale = 500;

    for(int i = 0; i < nodeSet().size(); ++i) {
        const Node * curNode = nodeSet()[i];
        if(curNode->deformation() == NULL ||
                (onlyOnNodes && !curNode->onOff())) {
            continue;
        }

        int nodeIdx = idxNode(curNode);
        int defIdx = idxDeformation(curNode->deformation());
        string strDef = NumToString_<int>(defIdx, zeroPaddingNum);

        int flip = static_cast<int>(curNode->isLRFlip());
        string strFlip = NumToString_<int>(flip);

        const Deformation::Param w = curNode->deformationParam();

        int partHt = curNode->detectWindow().height() * bs;
        int partWd = curNode->detectWindow().width() * bs;

        cv::Mat_<Scalar> def(partHt, partWd, Scalar(0));

        int probex = partWd / 2;
        int probey = partHt / 2;

        Deformation::Param displacement;

        for(int y = 0; y < partHt; ++y) {
            Scalar py = Scalar(probey - y) / bs;
            displacement(2) = py * py;
            displacement(3) = py;

            for(int x = 0; x < partWd; ++x) {
                Scalar px = Scalar(probex - x) / bs;
                displacement(0) = px * px;
                displacement(1) = px;

                Scalar penalty = w.dot(displacement) * defScale;

                def(y, x) = penalty;
            }
        }

        Mat imgPadding(def.rows + 2 * padding, def.cols + 2 * padding, def.type(),
                       cv::Scalar::all(128));
        def.copyTo(imgPadding(cv::Rect(padding, padding, def.cols, def.rows)));

        Mat imgShow;
        //cv::normalize(img, imgShow(cv::Rect(padding, padding, img.cols, img.rows)), 255, 0.0, CV_MINMAX, CV_8UC1);

        //        cv::imshow("Deformation", imgShow);
        //        cv::waitKey(2);

        string saveName = saveDir + string("Deformation_") + strDef + "_" + strFlip  +
                          imgExt;

        if(ptScoreMaps != NULL) {
            std::map<int, Mat>::iterator iter = ptScoreMaps->find(nodeIdx);
            if(iter == ptScoreMaps->end()) {
                imgPadding.convertTo(imgShow, CV_8UC1);
                cv::imwrite(saveName, imgShow);
            } else {
                Mat imgShow1;
                imgPadding.convertTo(imgShow1, CV_8UC1);
                cv::cvtColor(imgShow1, imgShow, CV_GRAY2BGR);
                int bigRows = std::max<int>(imgShow.rows, iter->second.rows);
                int bigCols = imgShow.cols + iter->second.cols;
                Mat bigShow(bigRows, bigCols, iter->second.type());
                bigShow = cv::Scalar::all(255);
                imgShow.copyTo(bigShow(cv::Rect(0, 0, imgShow.cols, imgShow.rows)));
                iter->second.copyTo(bigShow(cv::Rect(imgShow.cols, 0, iter->second.cols,
                                                     iter->second.rows)));
                cv::imwrite(saveName, bigShow);
            }
        } else {
            imgPadding.convertTo(imgShow, CV_8UC1);
            cv::imwrite(saveName, imgShow);
        }

    } // for i
}


/// Instantiation
INSTANTIATE_CLASS_(AOGrammar_);
INSTANTIATE_BOOST_SERIALIZATION_(AOGrammar_);


} // namespace RGM
