#include "parse_tree.hpp"
#include "AOGrammar.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilString.hpp"

namespace RGM {

// ------ PtEdge ------

template<class Archive>
void PtEdge::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(idx_);
}

INSTANTIATE_BOOST_SERIALIZATION(PtEdge);



// ------ PtNode ------

PtNode::PtNode(const PtNode & n) {
    idxInEdges_ = n.idxInEdges();
    idxOutEdges_ = n.idxOutEdges();
    idx_ = n.idx();
}

PtNode::PtNode(int gNode) {
    idx_.fill(-1);
    idx_(IDX_VALID) = 1;
    idx_[IDX_G] = gNode;
}

template<class Archive>
void PtNode::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(idxInEdges_);
    ar & BOOST_SERIALIZATION_NVP(idxOutEdges_);
    ar & BOOST_SERIALIZATION_NVP(idx_);
}

INSTANTIATE_BOOST_SERIALIZATION(PtNode);




// ------ PtStates ------

void PtStates::init() {
    isBelief_ = false;
    score_ = 0;
    loss_ = 1;
    margin_ = 0;
    norm_ = 0;
}

template<class Archive>
void PtStates::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(isBelief_);
    ar & BOOST_SERIALIZATION_NVP(score_);
    ar & BOOST_SERIALIZATION_NVP(loss_);
    ar & BOOST_SERIALIZATION_NVP(margin_);
    ar & BOOST_SERIALIZATION_NVP(norm_);
}

INSTANTIATE_BOOST_SERIALIZATION(PtStates);




// ------ ParseTree ------

template<int Dimension>
ParseTree::ParseTree_(const ParseTree & pt)
    : g_(NULL), idxRootNode_(-1), states_(NULL),
      appearanceX_(NULL) {

    nodeSet_.resize(pt.nodeSet().size(), NULL);
    for(int i = 0; i < nodeSet_.size(); ++i) {
        nodeSet_[i] = new PtNode(*pt.nodeSet()[i]);
    }

    edgeSet_.resize(pt.edgeSet().size(), NULL);
    for(int i = 0; i < edgeSet_.size(); ++i) {
        edgeSet_[i] = new PtEdge(*pt.edgeSet()[i]);
    }

    idxRootNode_ = pt.idxRootNode();

    g_ = pt.grammar();

    appearanceSet_.resize(pt.appearanceSet().size(), NULL);
    for(int i = 0; i < appearanceSet_.size(); ++i) {
        appearanceSet_[i] = new AppParam(
                    *pt.appearanceSet()[i]);
    }

    biasSet_ = pt.biasSet();

    deformationSet_.resize(pt.deformationSet().size(), NULL);
    for(int i = 0; i < deformationSet_.size(); ++i) {
        deformationSet_[i] = new Deformation::Param(*pt.deformationSet()[i]);
    }

    scalepriorSet_.resize(pt.scalepriorSet().size(), NULL);
    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        scalepriorSet_[i] = new Scaleprior::Param(*pt.scalepriorSet()[i]);
    }

    parseInfoSet_.resize(pt.parseInfoSet().size(), NULL);
    for(int i = 0; i < parseInfoSet_.size(); ++i) {
        parseInfoSet_[i] = new ParseInfo(*pt.parseInfoSet()[i]);
    }

    if(pt.states() != NULL) {
        states_ = new PtStates(*pt.states());
    }

    appUsage_ = pt.appUsage();

    dataId_ = pt.dataId();

    ptId_ = pt.ptId();

    if(pt.appearaceX() != NULL) {
        appearanceX_ = new AppParam(
                    *(pt.appearaceX()));
    }

    imgWd_ = pt.imgWd();
    imgHt_ = pt.imgHt();

}

template<int Dimension>
ParseTree &
ParseTree::operator=(const ParseTree & pt) {
    if(this == &pt) {
        return *this;
    }

    clear();

    nodeSet_.resize(pt.nodeSet().size(), NULL);
    for(int i = 0; i < nodeSet_.size(); ++i) {
        nodeSet_[i] = new PtNode(*pt.nodeSet()[i]);
    }

    edgeSet_.resize(pt.edgeSet().size(), NULL);
    for(int i = 0; i < edgeSet_.size(); ++i) {
        edgeSet_[i] = new PtEdge(*pt.edgeSet()[i]);
    }

    idxRootNode_ = pt.idxRootNode();

    g_ = pt.grammar();

    appearanceSet_.resize(pt.appearanceSet().size(), NULL);
    for(int i = 0; i < appearanceSet_.size(); ++i) {
        appearanceSet_[i] = new AppParam(
                    *pt.appearanceSet()[i]);
    }

    biasSet_ = pt.biasSet();

    deformationSet_.resize(pt.deformationSet().size(), NULL);
    for(int i = 0; i < deformationSet_.size(); ++i) {
        deformationSet_[i] = new Deformation::Param(*pt.deformationSet()[i]);
    }

    scalepriorSet_.resize(pt.scalepriorSet().size(), NULL);
    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        scalepriorSet_[i] = new Scaleprior::Param(*pt.scalepriorSet()[i]);
    }

    parseInfoSet_.resize(pt.parseInfoSet().size(), NULL);
    for(int i = 0; i < parseInfoSet_.size(); ++i) {
        parseInfoSet_[i] = new ParseInfo(*pt.parseInfoSet()[i]);
    }

    if(pt.states() != NULL) {
        states_ = new PtStates(*pt.states());
    }

    appUsage_ = pt.appUsage();

    dataId_ = pt.dataId();

    ptId_ = pt.ptId();

    if(pt.appearaceX() != NULL) {
        appearanceX_ = new AppParam(
                    *(pt.appearaceX()));
    }

    imgWd_ = pt.imgWd();
    imgHt_ = pt.imgHt();

    return *this;
}

template<int Dimension>
bool ParseTree::operator<(const ParseTree & pt) const {
    return parseInfo(rootNode())->score_ >
            pt.parseInfo(pt.rootNode())->score_; // for decreasing sort
}

template<int Dimension>
void ParseTree::swap(ParseTree & pt) {
    if(this == &pt) {
        return;
    }

    clear();

    //    nodeSet_.swap(pt.getNodeSet());

    //    edgeSet_.swap(pt.getEdgeSet());

    nodeSet_.resize(pt.nodeSet().size(), NULL);
    for(int i = 0; i < nodeSet_.size(); ++i) {
        nodeSet_[i] = new PtNode(*pt.nodeSet()[i]);
    }

    edgeSet_.resize(pt.edgeSet().size(), NULL);
    for(int i = 0; i < edgeSet_.size(); ++i) {
        edgeSet_[i] = new PtEdge(*pt.edgeSet()[i]);
    }

    idxRootNode_ = pt.idxRootNode();

    g_ = pt.grammar();

    //    appearanceSet_.swap(pt.getAppearanceSet());
    //    biasSet_.swap(pt.getBiasSet());
    //    deformationSet_.swap(pt.getDeformationSet());
    //    scalepriorSet_.swap(pt.getScalepriorSet());
    //    parseInfoSet_.swap(pt.getParseInfoSet());
    //    appUsage_.swap(pt.getAppUsage());

    appearanceSet_.resize(pt.appearanceSet().size(), NULL);
    for(int i = 0; i < appearanceSet_.size(); ++i) {
        appearanceSet_[i] = new AppParam(
                    *pt.appearanceSet()[i]);
    }

    biasSet_ = pt.biasSet();

    deformationSet_.resize(pt.deformationSet().size(), NULL);
    for(int i = 0; i < deformationSet_.size(); ++i) {
        deformationSet_[i] = new Deformation::Param(*pt.deformationSet()[i]);
    }

    scalepriorSet_.resize(pt.scalepriorSet().size(), NULL);
    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        scalepriorSet_[i] = new Scaleprior::Param(*pt.scalepriorSet()[i]);
    }

    parseInfoSet_.resize(pt.parseInfoSet().size(), NULL);
    for(int i = 0; i < parseInfoSet_.size(); ++i) {
        parseInfoSet_[i] = new ParseInfo(*pt.parseInfoSet()[i]);
    }

    appUsage_ = pt.appUsage();

    if(pt.states() != NULL) {
        states_ = new PtStates(*pt.states());
        //std::swap(states_, pt.getStates());
    }

    dataId_ = pt.dataId();

    ptId_ = pt.ptId();

    if(pt.appearaceX() != NULL) {
        //std::swap(appearanceX_, pt.getAppearaceX());
        appearanceX_ = new AppParam(*pt.appearaceX());
    }

    imgWd_ = pt.imgWd();
    imgHt_ = pt.imgHt();
}

template<int Dimension>
void ParseTree::clear() {
    for(int i = 0; i < nodeSet_.size(); ++i) {
        delete nodeSet_[i];
    }
    nodeSet_.clear();

    for(int i = 0; i < edgeSet_.size(); ++i) {
        delete edgeSet_[i];
    }
    edgeSet_.clear();

    idxRootNode_ = -1;

    g_ = NULL;

    for(int i = 0; i < appearanceSet_.size(); ++i) {
        delete appearanceSet_[i];
    }
    appearanceSet_.clear();

    biasSet_.clear();

    for(int i = 0; i < deformationSet_.size(); ++i) {
        delete deformationSet_[i];
    }
    deformationSet_.clear();

    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        delete scalepriorSet_[i];
    }
    scalepriorSet_.clear();

    for(int i = 0; i < parseInfoSet_.size(); ++i) {
        delete parseInfoSet_[i];
    }
    parseInfoSet_.clear();

    if(states_ != NULL) {
        delete states_;
        states_ = NULL;
    }

    dataId_ = -1;
    ptId_ = -1;

    if(appearanceX_ != NULL) {
        delete appearanceX_;
        appearanceX_ = NULL;
    }

    imgWd_ = 0;
    imgHt_ = 0;
}

template<int Dimension>
void ParseTree::deleteSubtree(PtNode * node) {
    RGM_CHECK_NOTNULL(node);

    vector<PtNode *> BFS;
    BFS.push_back(node);
    int head = 0;

    while(head < BFS.size()) {
        PtNode * curNode = BFS[head];
        int t = curNode->idx()[PtNode::IDX_TYPE];

        for(int i = 0; i < curNode->idxOutEdges().size(); ++i) {
            PtEdge * edge = edgeSet_[curNode->idxOutEdges()[i]];
            BFS.push_back(nodeSet_[edge->idx()[PtEdge::IDX_TO]]);

            delete edgeSet_[curNode->idxOutEdges()[i]];
            edgeSet_[curNode->idxOutEdges()[i]] = NULL;
        }

        int idx = curNode->idx()[PtNode::IDX_APP];
        if(idx != -1) {
            if(appearanceSet_[idx] != NULL) {
                delete appearanceSet_[idx];
                appearanceSet_[idx] = NULL;
            }
        }

        idx = curNode->idx()[PtNode::IDX_BIAS];
        if(idx != -1) {
            biasSet_[idx] = std::numeric_limits<Scalar>::quiet_NaN();
        }

        idx = curNode->idx()[PtNode::IDX_DEF];
        if(idx != -1) {
            if(deformationSet_[idx] != NULL) {
                delete deformationSet_[idx];
                deformationSet_[idx] = NULL;
            }
        }

        idx = curNode->idx()[PtNode::IDX_SCALEPRIOR];
        if(idx != -1) {
            if(scalepriorSet_[idx] != NULL) {
                delete scalepriorSet_[idx];
                scalepriorSet_[idx] = NULL;
            }
        }

        idx = curNode->idx()[PtNode::IDX_PARSEINFO];
        if(idx != -1) {
            if(parseInfoSet_[idx] != NULL) {
                delete parseInfoSet_[idx];
                parseInfoSet_[idx] = NULL;
            }
        }

        idx = curNode->idx()[PtNode::IDX_MYSELF];
        delete nodeSet_[idx];
        nodeSet_[idx] = NULL;

        ++head;
    }

    // node set
    vector<int> newNodeIdx(nodeSet_.size(), -1);

    vector<int> validIdx;
    for(int i = 0; i < nodeSet_.size(); ++i) {
        if(nodeSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    int j = 0;
    for(; j < validIdx.size(); ++j) {
        nodeSet_[j] = nodeSet_[validIdx[j]];
        newNodeIdx[validIdx[j]] = j;
    }
    nodeSet_.resize(j);

    // edge set
    vector<int> newEdgeIdx(edgeSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < edgeSet_.size(); ++i) {
        if(edgeSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        edgeSet_[j] = edgeSet_[validIdx[j]];
        newEdgeIdx[validIdx[j]] = j;
    }
    edgeSet_.resize(j);

    // app set
    vector<int> newAppIdx(appearanceSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < appearanceSet_.size(); ++i) {
        if(appearanceSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        appearanceSet_[j] = appearanceSet_[validIdx[j]];
        newAppIdx[validIdx[j]] = j;
    }
    appearanceSet_.resize(j);

    // bias set
    vector<int> newBiasIdx(biasSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < biasSet_.size(); ++i) {
        if(biasSet_[i] != numeric_limits<Scalar>::quiet_NaN()) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        biasSet_[j] = biasSet_[validIdx[j]];
        newBiasIdx[validIdx[j]] = j;
    }
    biasSet_.resize(j);

    // def
    vector<int> newDefIdx(deformationSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < deformationSet_.size(); ++i) {
        if(deformationSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        deformationSet_[j] = deformationSet_[validIdx[j]];
        newDefIdx[validIdx[j]] = j;
    }
    deformationSet_.resize(j);

    // scale prior
    vector<int> newScalepriorIdx(scalepriorSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < scalepriorSet_.size(); ++i) {
        if(scalepriorSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        scalepriorSet_[j] = scalepriorSet_[validIdx[j]];
        newScalepriorIdx[validIdx[j]] = j;
    }
    scalepriorSet_.resize(j);

    // parse info
    vector<int> newParseInfoIdx(parseInfoSet_.size(), -1);

    validIdx.clear();
    for(int i = 0; i < parseInfoSet_.size(); ++i) {
        if(parseInfoSet_[i] != NULL) {
            validIdx.push_back(i);
        }
    }
    j = 0;
    for(; j < validIdx.size(); ++j) {
        parseInfoSet_[j] = parseInfoSet_[validIdx[j]];
        newParseInfoIdx[validIdx[j]] = j;
    }
    parseInfoSet_.resize(j);

    idxRootNode_ = newNodeIdx[idxRootNode_];

    // update node and edge set
    for(int i = 0; i < nodeSet_.size(); ++i) {
        nodeSet_[i]->getIdx()[PtNode::IDX_MYSELF] = i;

        int idx = nodeSet_[i]->getIdx()[PtNode::IDX_BIAS];
        if(idx != -1) {
            nodeSet_[i]->getIdx()[PtNode::IDX_BIAS] = newBiasIdx[idx];
        }

        idx = nodeSet_[i]->getIdx()[PtNode::IDX_DEF];
        if(idx != -1) {
            nodeSet_[i]->getIdx()[PtNode::IDX_DEF] = newDefIdx[idx];
        }

        idx = nodeSet_[i]->getIdx()[PtNode::IDX_SCALEPRIOR];
        if(idx != -1) {
            nodeSet_[i]->getIdx()[PtNode::IDX_SCALEPRIOR] = newScalepriorIdx[idx];
        }

        idx = nodeSet_[i]->getIdx()[PtNode::IDX_APP];
        if(idx != -1) {
            nodeSet_[i]->getIdx()[PtNode::IDX_APP] = newAppIdx[idx];
        }

        idx = nodeSet_[i]->getIdx()[PtNode::IDX_PARSEINFO];
        if(idx != -1) {
            nodeSet_[i]->getIdx()[PtNode::IDX_PARSEINFO] = newParseInfoIdx[idx];
        }

        for(int j = 0; j < nodeSet_[i]->getIdxInEdges().size(); ++j) {
            idx = nodeSet_[i]->getIdxInEdges()[j];
            nodeSet_[i]->getIdxInEdges()[j] = newEdgeIdx[idx];
        }

        for(int j = 0; j < nodeSet_[i]->getIdxOutEdges().size(); ++j) {
            idx = nodeSet_[i]->getIdxOutEdges()[j];
            nodeSet_[i]->getIdxOutEdges()[j] = newEdgeIdx[idx];
        }
    }

    for(int i = 0; i < edgeSet_.size(); ++i) {
        edgeSet_[i]->getIdx()[PtEdge::IDX_MYSELF]  = i;

        int idx = edgeSet_[i]->getIdx()[PtEdge::IDX_FROM];
        if(idx != -1) {
            edgeSet_[i]->getIdx()[PtEdge::IDX_FROM] = newNodeIdx[idx];
        }

        idx = edgeSet_[i]->getIdx()[PtEdge::IDX_TO];
        if(idx != -1) {
            edgeSet_[i]->getIdx()[PtEdge::IDX_TO] = newNodeIdx[idx];
        }
    }

}

template<int Dimension>
int ParseTree::idxObjComp() const {
    assert(!empty());

    return rootParseInfo()->c_;
}

template<int Dimension>
void ParseTree::createSample(FeaturePyr &pyr, InferenceParam & param) {
    RGM_CHECK_NOTNULL(g_);

    for ( int i = 0; i < nodeSet_.size(); ++i ) {
        PtNode * ptNode = nodeSet_[i];
        if ( ptNode->idx()[PtNode::IDX_TYPE] != T_NODE ) continue;
        const Node * gNode = g_->findNodeConst(ptNode->idx()[PtNode::IDX_G]);

        ParseInfo * info = getParseInfo(ptNode);

        if(param.createSample_) {
            int ds2 = std::pow<int>(2, info->ds_) - 1;
            int fy = info->y_ - pyr.pady() * ds2;
            int fx = info->x_ - pyr.padx() * ds2;
            int wd = gNode->appearance()->w().cols();
            int ht = gNode->appearance()->w().rows();
            AppParam w = pyr.levels()[info->l_].block(fy, fx, ht, wd);

//            int bs = 20;
//            FeaturePyr::visualize(w, bs);

            ptNode->getIdx()[PtNode::IDX_APP] =
                    addAppearance(w, g_->featType(),
                                     gNode->isLRFlip() && g_->isLRFlip() && g_->sharedLRFlip());

            if(appUsage().size() == 0)
               getAppUsage().assign(g_->appearanceSet().size(), 0);

            getAppUsage()[g_->idxAppearance(gNode->appearance())] += 1;


        }

        if(param.createRootSample2x_) {

            int ads = 1;
            int ax = 0;
            int ay = 0;

            int ads2 = std::pow<int>(2, ads);

            int toX = info->x_ * ads2 + ax;
            int toY = info->y_ * ads2 + ay;
            int toL = info->l_ - g_->interval() * ads;

            int ds2 = ads2 - 1;
            int fy = toY - pyr.pady() * ds2;
            int fx = toX - pyr.padx() * ds2;

            int wd = gNode->appearance()->w().cols() * ads2;
            int ht = gNode->appearance()->w().rows() * ads2;

            assert(appearaceX() == NULL);
            getAppearaceX() = new AppParam();

            if((fy >= 0) && (fy + ht <= pyr.levels()[toL].rows()) &&
                    (fx >= 0) && (fx + wd <= pyr.levels()[toL].cols()))
                *getAppearaceX() = pyr.levels()[toL].block(fy, fx, ht, wd);
            else {
                *getAppearaceX() =
                        Appearance::Param::Constant(
                            ht, wd, pyr.levels()[toL](0, 0));

                        int x1 = std::max<int>(fx, 0);
                int x2 = std::min<int>(fx + wd, pyr.levels()[toL].cols());
                int y1 = std::max<int>(fy, 0);
                int y2 = std::min<int>(fy + ht, pyr.levels()[toL].rows());
                int wd1 = x2 - x1;
                int ht1 = y2 - y1;

                int fx2 = (fx >= 0) ? 0 : -fx;
                int fy2 = (fy >= 0) ? 0 : -fy;

                getAppearaceX()->block(fy2, fx2, ht1, wd1) =
                        pyr.levels()[toL].block(y1, x1, ht1, wd1);
            }
            if(gNode->isLRFlip() && g_->isLRFlip() &&
                    g_->sharedLRFlip())
                *getAppearaceX() = FeaturePyr::Flip(
                        *getAppearaceX(), g_->featType());

            if(appUsage().size() == 0)
               getAppUsage().assign(g_->appearanceSet().size(), 0);

            getAppUsage()[g_->idxAppearance(gNode->appearance())] += 1;
        }

    }

}

template<int Dimension>
int ParseTree::addNode(int gNode, int type) {
    int idx = nodeSet().size();

    //    for ( int i = 0; i < idx; ++i ) {
    //        if ( nodeSet()[i]->idx()[PtNode::IDX_G] == gNode ) {
    //            LOG(FATAL) << "duplicated pt node";
    //            return -1;
    //        }
    //    }

    getNodeSet().push_back(new PtNode(gNode));

    getNodeSet().back()->getIdx()[PtNode::IDX_MYSELF] = idx;
    getNodeSet().back()->getIdx()[PtNode::IDX_TYPE] = type;

    return idx;
}

template<int Dimension>
int ParseTree::addEdge(int fromNode, int toNode, int gEdge, int type) {
    int idx = edgeSet().size();

    for(int i = 0; i < idx; ++i) {
        if(edgeSet()[i]->idx()[PtEdge::IDX_FROM] == fromNode &&
                edgeSet()[i]->idx()[PtEdge::IDX_TO] == toNode) {
            RGM_LOG(error, "duplicated pt edge");
            return -1;
        }
    }

    getEdgeSet().push_back(new PtEdge(fromNode, toNode, gEdge, type));
    getEdgeSet().back()->getIdx()[PtEdge::IDX_MYSELF] = idx;

    getNodeSet()[fromNode]->getIdxOutEdges().push_back(idx);
    getNodeSet()[toNode]->getIdxInEdges().push_back(idx);

    return idx;
}

template<int Dimension>
int ParseTree::AddBias(Scalar w) {
    int idx = biasSet().size();

    getBiasSet().push_back(w);

    return idx;
}

template<int Dimension>
int ParseTree::addScaleprior(Scaleprior::Param & w) {
    int idx = scalepriorSet().size();

    getScalepriorSet().push_back(new Scaleprior::Param(w));

    return idx;
}

template<int Dimension>
int ParseTree::addDeformation(Scalar dx, Scalar dy, bool flip) {
    int idx = deformationSet().size();

    Deformation::Param w;
    w << dx * dx, dx, dy * dy, dy;
    w *= -1.0F;

    if(flip) {
        w(1) *= -1;
    }

    getDeformationSet().push_back(new Deformation::Param(w));

    return idx;
}

template<int Dimension>
int ParseTree::addAppearance(AppParam & w, featureType t, bool flip) {
    int idx = appearanceSet().size();

    getAppearanceSet().push_back(new AppParam());

    if(flip) {
        getAppearanceSet().back()->swap(FeaturePyr::Flip(w, t));
    } else {
        getAppearanceSet().back()->swap(w);
    }

    return idx;
}

template<int Dimension>
int ParseTree::addParseInfo(ParseInfo & info) {
    int idx = parseInfoSet().size();

    getParseInfoSet().push_back(new ParseInfo(info));

    return idx;
}

template<int Dimension>
void ParseTree::showDetection(cv::Mat img, bool display, cv::Scalar color, bool
                              showPart) {
    if(showPart) {
        int tnodeIdx = 0;
        for(int i = 0; i < nodeSet().size(); ++i) {
            if(nodeSet()[i]->idx()(PtNode::IDX_TYPE) !=
                    static_cast<int>(T_NODE)) {
                continue;
            }
            const ParseInfo * info = parseInfo(*nodeSet()[i]);
            cv::rectangle(img, info->cvRect(), cv::Scalar::all(255), 3);
            if(tnodeIdx < rgbTableSz) {
                cv::rectangle(img, info->cvRect(), rgbTable[tnodeIdx], 2);
            } else {
                cv::rectangle(img, info->cvRect(), cv::Scalar(255, 0, 0), 2);
            }
            tnodeIdx++;

            /*if ( display ) {
            cv::String winName("AOGDetection");
            cv::imshow(winName, img);
            cv::waitKey(0);
            }*/
        }

        cv::rectangle(img, rootParseInfo()->cvRect(),
                      cv::Scalar::all(255), 5);
        cv::rectangle(img, rootParseInfo()->cvRect(), color, 3);
    } else {
        cv::rectangle(img, rootParseInfo()->cvRect(), color, 2);
    }

    if(display) {
        cv::String winName("AOGDetection");
        cv::imshow(winName, img);
        cv::waitKey(0);
    }
}

template<int Dimension>
void ParseTree::visualize(string & saveName, std::map<int, Scalar> * label) {
    string strDir = FileUtil::GetParentDir(saveName);

    string dotfile = saveName + ".dot";
    std::ofstream ofs(dotfile.c_str(), std::ios::out);
    if(!ofs.is_open()) {
        RGM_LOG(error, "Can not write file " + dotfile);
        return;
    }

    ofs << "digraph parseTree " << "{\n ";
//        << "rankdir = LR; \n" ;

    ofs << "pack=true;\n overlap=false;\n labelloc=t;\n center=true;\n";

    const string imgExt(".png");
    int showWd = 100;
    int showHt = 100;
    string strFixed = "false";

    for(int i = 0; i < nodeSet().size(); ++i) {
        const PtNode * ptnode = nodeSet()[i];
        int idx = ptnode->idx()[PtNode::IDX_MYSELF];

        nodeType t = static_cast<nodeType>(
                    ptnode->idx()[PtNode::IDX_TYPE]);

        string strApp = NumToString_<int>(idx, 5);
        string strColor;
        string strShape;
        string strStyle;
        string strLabel;
        if(label != NULL) {
            strLabel = NumToString_<Scalar>((*label)[idx]);
        }

        switch(t) {
        case T_NODE:
            strColor = "red";
            strShape = "box";
            strStyle = "bold";
            break;
        case AND_NODE:
            strColor = "blue";
            strShape = "ellipse";
            strStyle = "bold";
            break;
        case OR_NODE:
            strColor = "green";
            strShape = "ellipse";
            strStyle = "bold";
            break;
        }

        ofs << "node" << idx
            << "[shape=" << strShape << ", style=" << strStyle
            << ", color=" << strColor << ", "
            << "label=<<TABLE border=\"0\" cellborder=\"0\">"
            << "<TR><TD width=\"" << showWd << "\" height=\"" << showHt
            << "\" fixedsize=\"" << strFixed << "\">"
            << "<IMG SRC=\""
            << strDir << strApp << imgExt
            << "\"/></TD></TR>"
//            << "<TR><TD><br/><font point-size=\"20\">"
//            << strLabel << "</font></TD></TR>"
            << "</TABLE>>"
            << "];\n";
    }

    for(int i = 0; i < edgeSet().size(); ++i) {
        const PtEdge * ptedge = edgeSet()[i];
        int idxFrom = ptedge->idx()[PtEdge::IDX_FROM];
        int idxTo = ptedge->idx()[PtEdge::IDX_TO];

        edgeType t =
                static_cast<edgeType>(ptedge->idx()[PtEdge::IDX_TYPE]);

        switch(t) {
        case SWITCHING: {
            ofs << "edge [style=bold, color=green];\n";
            break;
        }
        case COMPOSITION: {
            ofs << "edge [style=bold, color=blue];\n";
            break;
        }
        case DEFORMATION: {
            ofs << "edge [style=bold, color=red];\n";
            break;
        }
        case TERMINATION: {
            ofs << "edge [style=bold, color=black];\n";
            break;
        }
        }

        ofs << "node" << idxFrom << " -> node" << idxTo << ";\n";
    }

    ofs << "}\n";
    ofs.close();

    string baseName = FileUtil::GetFileBaseName(saveName);

    string cmd = "dot -Tpdf " + dotfile + " -o " + strDir + baseName + ".pdf";
    std::system(cmd.c_str());

    cmd = "dot -Tpng " + dotfile + " -o " + strDir + baseName +
            ".png"; //-Gsize=15,10\! -Gdpi=100
    std::system(cmd.c_str());
}

template<int Dimension>
int ParseTree::dim() const {
    int d = 0;

    for(int i = 0; i < appearanceSet().size(); ++i) {
        d += appearanceSet()[i]->size() * Dimension;
    }

    d += biasSet().size();

    d += deformationSet().size() * 4;

    d += scalepriorSet().size() * 3;

    return d;
}

template<int Dimension>
int ParseTree::compareFeatures(const ParseTree & pt) const {
    assert(grammar() != NULL && grammar() == pt.grammar());

    for(int i = 0;  i < nodeSet().size(); ++i) {
        int idxG = nodeSet()[i]->idx()(PtNode::IDX_G);

        int ii = 0;
        for(; ii < pt.nodeSet().size(); ++ii) {
            int idxG1 = pt.nodeSet()[ii]->idx()(PtNode::IDX_G);
            if(idxG1 == idxG) {
                break;
            }
        } // for ii

        if(ii == pt.nodeSet().size()) {
            return 1;
        }

        for(int j = PtNode::IDX_BIAS; j < PtNode::IDX_APP + 1; ++j) {
            int fidx  = nodeSet()[i]->idx()(j);
            int fidx1 = pt.nodeSet()[ii]->idx()(j);
            if(fidx != -1 && fidx1 != -1) {
                switch(j) {
                case PtNode::IDX_BIAS: {
                    if(biasSet()[fidx] > pt.biasSet()[fidx1]) {
                        return 1;
                    } else if(biasSet()[fidx] < pt.biasSet()[fidx1]) {
                        return -1;
                    }

                    break;
                }
                case PtNode::IDX_DEF: {
                    const Deformation::Param & p(*deformationSet()[fidx]);
                    const Deformation::Param & p1(*pt.deformationSet()[fidx1]);
                    for(int k = 0; k < 4; ++k) {
                        if(p(k) > p1(k)) {
                            return 1;
                        } else if(p(k) < p1(k)) {
                            return -1;
                        }
                    }
                    break;
                }
                case PtNode::IDX_SCALEPRIOR: {
                    const Scaleprior::Param & p(*scalepriorSet()[fidx]);
                    const Scaleprior::Param & p1(*pt.scalepriorSet()[fidx1]);
                    for(int k = 0; k < 3; ++k) {
                        if(p(k) > p1(k)) {
                            return 1;
                        } else if(p(k) < p1(k)) {
                            return -1;
                        }
                    }
                    break;
                }
                case PtNode::IDX_APP: {
                    const AppParam & p(*appearanceSet()[fidx]);
                    const AppParam & p1(*pt.appearanceSet()[fidx1]);
                    for(int row = 0; row < p.rows(); ++ row)
                        for(int col = 0; col < p.cols(); ++col)
                            for(int k = 0; k < Dimension; ++k) {
                                if(p(row, col)(k) > p1(row, col)(k)) {
                                    return 1;
                                } else if(p(row, col)(k) < p1(row, col)(k)) {
                                    return -1;
                                }
                            }
                    break;
                }
                }
            } else {
                if(fidx != -1 && fidx1 == -1) {
                    return 1;
                }

                if(fidx == -1 && fidx1 != -1) {
                    return -1;
                }
            }
        }

    } // for i

    return 0;
}

template<int Dimension>
Scalar ParseTree::norm() const {
    Scalar n = 0;

    for(int i = 0; i < appearanceSet().size(); ++i) {
        n += FeaturePyr::Map(*(appearanceSet()[i])).squaredNorm();
    }

    n += std::inner_product(biasSet().begin(), biasSet().end(), biasSet().begin(),
                            0);

    for(int i = 0; i < deformationSet().size(); ++i) {
        n += deformationSet()[i]->squaredNorm();
    }

    for(int i = 0; i < scalepriorSet().size(); ++i) {
        n += scalepriorSet()[i]->squaredNorm();
    }

    return std::sqrt(n);
}

template<int Dimension>
Scalar ParseTree::computeOverlapLoss(const Rectangle & ref) const {
    Intersector_<int> inter(ref, 0.5F, true);

    const ParseInfo * p = rootParseInfo();

    Rectangle box(p->x(), p->y(), p->width(), p->height());

    Scalar ov = 0;

    inter(box, &ov);

    return 1 - ov;
}

template<int Dimension>
vector<const PtNode *> ParseTree::findNode(const Node_<Dimension> * n) {
    RGM_CHECK_NOTNULL(grammar());
    RGM_CHECK_NOTNULL(n);

    int gIdx = grammar()->idxNode(n);
    RGM_CHECK_NOTEQ(gIdx, -1);

    return findNode(gIdx);
}

template<int Dimension>
vector<const PtNode *> ParseTree::findNode(const int idxG) {
    vector<const PtNode *>  n;

    for(int i = 0; i < nodeSet().size(); ++i) {
        if(nodeSet()[i]->idx()[PtNode::IDX_G] == idxG) {
            n.push_back(nodeSet()[i]);
        }
    }

    return n;
}

template<int Dimension>
vector<PtNode *> ParseTree::getNode(const int idxG) {
    vector<PtNode *> n;

    for(int i = 0; i < nodeSet().size(); ++i) {
        if(nodeSet()[i]->idx()[PtNode::IDX_G] == idxG) {
            n.push_back(getNodeSet()[i]);
        }
    }

    return n;
}


template<int Dimension>
vector<const PtNode *> ParseTree::findSingleObjAndNodes() const {
    vector<const PtNode *> sobj;

    if(grammar()->isSingleObjModel()) {
        sobj.push_back(toNode(outEdge(0, rootNode())));
    } else {
        // for N-car model, to be rewritten later on
        for(int i = 0; i < nodeSet().size(); ++i) {
            const PtNode * n = nodeSet()[i];
            if(n->idx()[PtNode::IDX_TYPE] != static_cast<int>(T_NODE))
                continue;

            // Bottom-up: tnode -> and-node -> or-node -> single obj and-node
            const PtNode * a1 = fromNode(inEdge(0, n));
            const PtNode * o  = fromNode(inEdge(0, a1));
            const PtNode * a2 = fromNode(inEdge(0, o));

            if(a2->idx()[PtNode::IDX_VALID] <= 0)
                continue;

            bool found = false;
            for(int j = 0; j < sobj.size(); ++j) {
                if(a2 == sobj[j]) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                sobj.push_back(a2);
            }
        }
    }

    return sobj;
}

template<int Dimension>
vector<PtNode *> ParseTree::getSingleObjAndNodes() {
    RGM_CHECK_NOTNULL(grammar());

    vector<PtNode *> sobj;

    if(grammar()->isSingleObjModel()) {
        sobj.push_back(getToNode(getOutEdge(0, rootNode())));
    } else {

        for(int i = 0; i < nodeSet().size(); ++i) {
            PtNode * n = getNodeSet()[i];
            if(n->idx()[PtNode::IDX_TYPE] != static_cast<int>(T_NODE))
                continue;

            // Bottom-up: tnode -> and-node -> or-node -> single obj and-node
            PtNode * a1 = getFromNode(getInEdge(0, n));
            PtNode * o  = getFromNode(getInEdge(0, a1));
            PtNode * a2 = getFromNode(getInEdge(0, o));

            if(a2->idx()[PtNode::IDX_VALID] <= 0)
                continue;

            bool found = false;
            for(int j = 0; j < sobj.size(); ++j) {
                if(a2 == sobj[j]) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                sobj.push_back(a2);
            }
        }
    }

    return sobj;
}

template<int Dimension>
void ParseTree::getSingleObjDet(vector<Detection> & dets, int ptIdx) {
    vector<const PtNode *> sobj = findSingleObjAndNodes();

    for(int i = 0; i < sobj.size(); ++i) {
        const ParseInfo * info = parseInfo(sobj[i]);
        Rectangle_<Scalar> bbox(info->x(), info->y(),
                                info->width(), info->height());
        if(bbox.width() == 0 || bbox.height() == 0) {
            const ParseInfo * info1 = parseInfo(toNode(outEdge(0, sobj[i])));
            bbox.setWidth(info1->width());
            bbox.setHeight(info1->height());
            RGM_CHECK_EQ(info->x(), info1->x());
            RGM_CHECK_EQ(info->y(), info1->y());
        }
        int idxG = sobj[i]->idx()[PtNode::IDX_G];
        Detection det(idxG, info->l_, info->x_, info->y_, info->score_,
                      bbox, ptIdx, sobj[i]->idx()[PtNode::IDX_MYSELF]);
        if(det.clipBbox(imgWd(), imgHt())) {
            dets.push_back(det);
        }
    }
}

template<int Dimension>
void ParseTree::doBboxPred(vector<Detection> & dets, int ptIdx) {
    if(grammar() == NULL) {
        RGM_LOG(error, "No grammar model is specified");
        return;
    }

    if(grammar()->bboxPred().size() == 0)
        return;

    vector<const PtNode *> sobj = findSingleObjAndNodes();

    for(int i = 0; i < sobj.size(); ++i) {
        const PtNode * n = sobj[i];
        // get detection
        const ParseInfo * info = parseInfo(n);
        Rectangle_<Scalar> bbox(info->x(), info->y(),
                                info->width(), info->height());
        if(bbox.width() == 0 || bbox.height() == 0) {
            const ParseInfo * info1 = parseInfo(toNode(outEdge(0, n)));
            bbox.setWidth(info1->width());
            bbox.setHeight(info1->height());
            RGM_CHECK_EQ(info->x(), info1->x());
            RGM_CHECK_EQ(info->y(), info1->y());
        }
        int idxG = n->idx()[PtNode::IDX_G];
        //const ParseInfo *info2 = sobj[i]->inEdge(0, pt)->fromNode(pt)->parseInfo(&pt);
        Detection det(idxG, info->l_, info->x_, info->y_, info->score_,
                      bbox, ptIdx, sobj[i]->idx()[PtNode::IDX_MYSELF]);
        if(!det.clipBbox(imgWd(), imgHt()))
            continue;

        // get prediction model
        std::map<int, Matrix>::const_iterator iter = grammar()->bboxPred().find(idxG);
        //CHECK_NE(iter, grammar()->bboxPred().end());
        const Matrix & pred(iter->second);

        RGM_CHECK_EQ(pred.rows(), n->idxOutEdges().size() * 2 + 1);

        Scalar wd = bbox.width() - 1;
        Scalar ht = bbox.height() -
                1; // bug due to the setting in learning the pred. model, to be fixed
        Scalar rx = bbox.x() + wd / 2.0F;
        Scalar ry = bbox.y() + ht / 2.0F;

        Matrix A(Matrix::Zero(1, pred.rows()));
        int c = 0;

        // get detections of all appearance filters
        for(int j = 0; j < n->idxOutEdges().size(); ++j) {
            const PtNode * o = toNode(outEdge(j, n));
            const PtNode * a = toNode(outEdge(j, o));
            const PtNode * t = toNode(outEdge(j, a));

            const ParseInfo * tinfo = parseInfo(t);

            Scalar tx = tinfo->x() + (tinfo->width() - 1.0F) / 2.0F;
            Scalar ty = tinfo->y() + (tinfo->height() - 1.0F) / 2.0F;

            A(0, c++) = (tx - rx) / wd;
            A(0, c++) = (ty - ry) / ht;
        }

        A(0, c) = 1;

        // compute the predicted bbox
        Matrix dxy = A * pred;

        int x1 = bbox.x() + dxy(0, 0) * wd;
        int y1 = bbox.y() + dxy(0, 1) * ht;
        int x2 = bbox.right() + dxy(0, 2) * wd;
        int y2 = bbox.bottom() + dxy(0, 3) * ht;

        det.setX(x1);
        det.setY(y1);
        det.setWidth(x2 - x1 + 1);
        det.setHeight(y2 - y1 + 1);

        if(det.clipBbox(imgWd(), imgHt())) {
            dets.push_back(det);
        }
    }
}

template<int Dimension>
Scalar ParseTree::score() {
    return parseInfo(*rootNode())->score_;
}

template<int Dimension>
const ParseInfo * ParseTree::rootParseInfo() const {
    return parseInfo(*rootNode());
}

template<int Dimension>
Rectangle ParseTree::bbox() {
    const ParseInfo  * info = rootParseInfo();
    return Rectangle(info->x(), info->y(), info->width(), info->height());
}

template<int Dimension>
const PtNode * ParseTree::fromNode(const PtEdge & e) const {
    int i = e.idx()[PtEdge::IDX_FROM];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return nodeSet()[i];
}

template<int Dimension>
PtNode * ParseTree::getFromNode(const PtEdge & e) {
    int i = e.idx()[PtEdge::IDX_FROM];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return getNodeSet()[i];
}

template<int Dimension>
const PtNode * ParseTree::fromNode(const PtEdge * e) const {
    assert(e != NULL);
    int i = e->idx()[PtEdge::IDX_FROM];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return nodeSet()[i];
}

template<int Dimension>
PtNode * ParseTree::getFromNode(const PtEdge * e) {
    RGM_CHECK_NOTNULL(e);
    int i = e->idx()[PtEdge::IDX_FROM];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return getNodeSet()[i];
}

template<int Dimension>
const PtNode * ParseTree::toNode(const PtEdge & e) const {
    int i = e.idx()[PtEdge::IDX_TO];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return nodeSet()[i];
}

template<int Dimension>
PtNode * ParseTree::getToNode(const PtEdge & e) {
    int i = e.idx()[PtEdge::IDX_TO];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return getNodeSet()[i];
}

template<int Dimension>
const PtNode * ParseTree::toNode(const PtEdge * e) const {
    assert(e != NULL);
    int i = e->idx()[PtEdge::IDX_TO];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return nodeSet()[i];
}

template<int Dimension>
PtNode * ParseTree::getToNode(const PtEdge * e) {
    RGM_CHECK_NOTNULL(e);
    int i = e->idx()[PtEdge::IDX_TO];
    if(i < 0 || i >= nodeSet().size())
        return NULL;

    return getNodeSet()[i];
}

template<int Dimension>
const PtEdge * ParseTree::inEdge(int i, const PtNode & n) const {
    if(i < 0  || i >= n.idxInEdges().size())
        return NULL;

    return edgeSet()[n.idxInEdges()[i]];
}

template<int Dimension>
PtEdge * ParseTree::getInEdge(int i, const PtNode & n) {
    if(i < 0  || i >= n.idxInEdges().size())
        return NULL;

    return getEdgeSet()[n.idxInEdges()[i]];
}


template<int Dimension>
const PtEdge * ParseTree::inEdge(int i, const PtNode * n) const {
    assert(n != NULL);
    if(i < 0  || i >= n->idxInEdges().size())
        return NULL;

    return edgeSet()[n->idxInEdges()[i]];
}

template<int Dimension>
PtEdge * ParseTree::getInEdge(int i, const PtNode * n) {
    RGM_CHECK_NOTNULL(n);
    if(i < 0  || i >= n->idxInEdges().size())
        return NULL;

    return getEdgeSet()[n->idxInEdges()[i]];
}

template<int Dimension>
const PtEdge * ParseTree::outEdge(int i, const PtNode & n) const {
    if(i < 0  || i >= n.idxOutEdges().size())
        return NULL;

    return edgeSet()[n.idxOutEdges()[i]];
}

template<int Dimension>
PtEdge * ParseTree::getOutEdge(int i, const PtNode & n) {
    if(i < 0  || i >= n.idxOutEdges().size())
        return NULL;

    return getEdgeSet()[n.idxOutEdges()[i]];
}

template<int Dimension>
const PtEdge * ParseTree::outEdge(int i, const PtNode * n) const {
    assert(n != NULL);
    if(i < 0  || i >= n->idxOutEdges().size())
        return NULL;

    return edgeSet()[n->idxOutEdges()[i]];
}

template<int Dimension>
PtEdge * ParseTree::getOutEdge(int i, const PtNode * n) {
    RGM_CHECK_NOTNULL(n);
    if(i < 0  || i >= n->idxOutEdges().size())
        return NULL;

    return getEdgeSet()[n->idxOutEdges()[i]];
}

template<int Dimension>
const ParseInfo  * ParseTree::parseInfo(const PtNode & n) const {
    return parseInfoSet()[n.idx()[PtNode::IDX_PARSEINFO]];
}

template<int Dimension>
ParseInfo *& ParseTree::getParseInfo(const PtNode & n) {
    return getParseInfoSet()[n.idx()[PtNode::IDX_PARSEINFO]];
}

template<int Dimension>
const ParseInfo  * ParseTree::parseInfo(const PtNode * n) const {
    assert(n != NULL);
    return parseInfoSet()[n->idx()[PtNode::IDX_PARSEINFO]];
}

template<int Dimension>
ParseInfo *& ParseTree::getParseInfo(const PtNode * n) {
    RGM_CHECK_NOTNULL(n);
    return getParseInfoSet()[n->idx()[PtNode::IDX_PARSEINFO]];
}

template<int Dimension>
template<class Archive>
void ParseTree::serialize(Archive & ar, const unsigned int version) {
    ar.register_type(static_cast<PtNode *>(NULL));
    ar.register_type(static_cast<PtEdge *>(NULL));
    ar.register_type(static_cast<AppParam *>(NULL));
    ar.register_type(static_cast<Deformation::Param *>(NULL));
    ar.register_type(static_cast<Scaleprior::Param *>(NULL));
    ar.register_type(static_cast<ParseInfo *>(NULL));
    ar.register_type(static_cast<PtStates *>(NULL));

    ar.template register_type<PtNode>();
    ar.template register_type<PtEdge>();
    ar.template register_type<AppParam>();
    ar.template register_type<Deformation::Param>();
    ar.template register_type<Scaleprior::Param>();
    ar.template register_type<ParseInfo>();
    ar.template register_type<PtStates>();

    ar & BOOST_SERIALIZATION_NVP(nodeSet_);
    ar & BOOST_SERIALIZATION_NVP(edgeSet_);
    ar & BOOST_SERIALIZATION_NVP(idxRootNode_);
    ar & BOOST_SERIALIZATION_NVP(appearanceSet_);
    ar & BOOST_SERIALIZATION_NVP(biasSet_);
    ar & BOOST_SERIALIZATION_NVP(deformationSet_);
    ar & BOOST_SERIALIZATION_NVP(scalepriorSet_);
    ar & BOOST_SERIALIZATION_NVP(parseInfoSet_);
    ar & BOOST_SERIALIZATION_NVP(dataId_);
    ar & BOOST_SERIALIZATION_NVP(ptId_);
    ar & BOOST_SERIALIZATION_NVP(states_);
    ar & BOOST_SERIALIZATION_NVP(appearanceX_);
    ar & BOOST_SERIALIZATION_NVP(imgWd_);
    ar & BOOST_SERIALIZATION_NVP(imgHt_);
}

/// Instantiation
INSTANTIATE_CLASS_(ParseTree_);
INSTANTIATE_BOOST_SERIALIZATION_(ParseTree_);



// ------ TrainSample ------

template<int Dimension>
TrainSample::TrainSample_(const TrainSample & ex) :
    marginBound_(ex.marginBound()), beliefNorm_(ex.beliefNorm()),
    maxNonbeliefNorm_(ex.maxNonbeliefNorm()),  nbHist_(ex.nbHist()) {
    pts_.resize(ex.pts().size());
    for(int i = 0; i < pts_.size(); ++i) {
        pts_[i] = ex.pts()[i];
    }
}

template<int Dimension>
TrainSample & TrainSample::operator=(const TrainSample & ex) {
    if(this == &ex) {
        return *this;
    }

    pts_.clear();
    pts_.resize(ex.pts().size());
    for(int i = 0; i < pts_.size(); ++i) {
        pts_[i] = ex.pts()[i];
    }

    marginBound_ = ex.marginBound();
    beliefNorm_ = ex.beliefNorm();
    maxNonbeliefNorm_ = ex.maxNonbeliefNorm();
    nbHist_ = ex.nbHist();

    return *this;
}

template<int Dimension>
void TrainSample::swap(TrainSample &ex) {
    if(this == &ex) return;

    pts_.swap(ex.getPts());
    std::swap(marginBound_, ex.getMarginBound());
    std::swap(beliefNorm_, ex.getBeliefNorm());
    std::swap(maxNonbeliefNorm_, ex.getMaxNonbeliefNorm());
    std::swap(nbHist_, ex.getNbHist());
}

template<int Dimension>
bool TrainSample::isEqual(const ParseTree & pt) const {
    assert(pts().size() >= 2);

    const ParseInfo * p = pts()[0]->rootParseInfo();
    const ParseInfo * p1 = pt.rootParseInfo();

    assert(p != NULL);
    assert(p1 != NULL);

    return (pts()[0]->dataId() == pt.dataId()) &&
            (p->c_ == p1->c_) && (p->l_ == p1->l_) &&
            (p->x_ == p1->x_) && (p->y_ == p1->y_);
}

template<int Dimension>
template<class Archive>
void TrainSample::serialize(Archive & ar,
                            const unsigned int version) {
    ar.register_type(static_cast<ParseTree *>(NULL));
    ar.template register_type<ParseTree >();

    ar & BOOST_SERIALIZATION_NVP(pts_);
    ar & BOOST_SERIALIZATION_NVP(marginBound_);
    ar & BOOST_SERIALIZATION_NVP(beliefNorm_);
    ar & BOOST_SERIALIZATION_NVP(maxNonbeliefNorm_);
    ar & BOOST_SERIALIZATION_NVP(nbHist_);
}


/// Instantiation
INSTANTIATE_CLASS_(TrainSample_);
INSTANTIATE_BOOST_SERIALIZATION_(TrainSample_);


// ------- TrainSampleSet_ ------

template<int Dimension>
Scalar TrainSampleSet::computeLoss(bool isPos, Scalar C, int start) {
    Scalar loss  = 0;

    typename TrainSampleSet::iterator iter;
    for(iter = this->begin() + start; iter != this->end();  ++iter) {
        Scalar s = iter->getPts()[0]->states()->score_;
        loss += max<Scalar>(0, isPos ? (1 - s) : (1 + s));
    }

    return loss * C;
}

template<int Dimension>
vector<ParseTree *> TrainSampleSet::getAllNonZeroPts() {
    vector<ParseTree *> pts;
    typename TrainSampleSet::iterator iter;
    for(iter = this->begin(); iter != this->end();  ++iter) {
        pts.push_back(iter->getPts()[0]);
    }

    return pts;
}

template<int Dimension>
template<class Archive>
void TrainSampleSet::serialize(Archive & ar,
                               const unsigned int version) {
    ar & boost::serialization::base_object<vector<TrainSample > >(*this);
}

/// Instantiation
INSTANTIATE_CLASS_(TrainSampleSet_);
INSTANTIATE_BOOST_SERIALIZATION_(TrainSampleSet_);

// ------- PtIntersector ------

template<int Dimension>
PtIntersector::PtIntersector_(const ParseTree  & reference,
                              Scalar threshold, bool dividedByUnion) :
    reference_(&reference), threshold_(threshold), dividedByUnion_(dividedByUnion) {
    RGM_CHECK_NOTNULL(reference_);
}

template<int Dimension>
bool PtIntersector::operator()(const ParseTree & pt, Scalar * score) const {
    if(score) {
        *score = 0.0;
    }

    const ParseInfo * ref = reference_->rootParseInfo();
    const ParseInfo * cur = pt.rootParseInfo();

    const int left = max<int>(ref->left(), cur->left());
    const int right = min<int>(ref->right(), cur->right());

    if(right < left) {
        return false;
    }

    const int top = max<int>(ref->top(), cur->top());
    const int bottom = min<int>(ref->bottom(), cur->bottom());

    if(bottom < top) {
        return false;
    }

    const int intersectionArea = (right - left + 1) * (bottom - top + 1);
    const int rectArea = cur->area();

    if(dividedByUnion_) {
        const int referenceArea = ref->area();
        const int unionArea = referenceArea + rectArea - intersectionArea;

        if(score) {
            *score = static_cast<Scalar>(intersectionArea) / unionArea;
        }

        if(intersectionArea >= unionArea * threshold_) {
            return true;
        }
    } else {
        if(score) {
            *score = static_cast<Scalar>(intersectionArea) / rectArea;
        }

        if(intersectionArea >= rectArea * threshold_) {
            return true;
        }
    }

    return false;
}

/// Instantiation
INSTANTIATE_CLASS_(PtIntersector_);


} //namespace RGM

