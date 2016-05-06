#include "parameter_learner.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilString.hpp"

namespace RGM {

//#define RGM_LEARNING_CHECK
#define RGM_USE_AOG_TRACKING

template <int Dimension>
ParameterLearner::ParameterLearner_(AOGrammar & g, int maxNumSamples, bool useZeroPt,
                                    Scalar maxMemoryMB) :
    grammar_(&g), wHist_(NULL), normDowHist_(NULL), useZeroPt_(useZeroPt),
    maxMemoryInUse_(maxMemoryMB), maxNumExInUse_(maxNumSamples) {

    Scalar m = g.computeMaxPtMemory();
    int num = maxMemoryMB / m;
    num = min(maxNumSamples, num);

    int num1 = num * (useZeroPt ? 2 : 1);
    ptPool_.resize(num1);
    ptPoolState_.assign(num1, false);

    RGM_LOG(normal, boost::format("[ParameterLearner]: ptPool size %d w.r.t."
                                  " (maxNumSamples=%d, maxMemoryMB=%f)")
            % num % maxNumSamples % maxMemoryMB);

}

template <int Dimension>
ParameterLearner::ParameterLearner_(AOGrammar & g, const string & featCacheDir,
                                    int maxNumSamples, bool useZeroPt,
                                    Scalar maxMemoryMB) : grammar_(&g),
    featCacheDir_(featCacheDir), wHist_(NULL), normDowHist_(NULL),
    useZeroPt_(useZeroPt),
    maxMemoryInUse_(maxMemoryMB), maxNumExInUse_(maxNumSamples) {

    RGM_CHECK(FileUtil::VerifyDirectoryExists(featCacheDir_, false), error);

    Scalar m = g.computeMaxPtMemory();
    int num = maxMemoryMB / m;
    num = min(maxNumSamples, num);

    int num1 = num * (useZeroPt ? 2 : 1);
    ptPool_.resize(num1);
    ptPoolState_.assign(num1, false);

    RGM_LOG(normal, boost::format("[ParameterLearner]: ptPool size %d w.r.t."
                                  " (maxNumSamples=%d, maxMemoryMB=%f)")
            % num % maxNumSamples % maxMemoryMB);
}

template <int Dimension>
ParameterLearner::~ParameterLearner_() {
    if(wHist_ != NULL) {
        delete wHist_;
        wHist_ = NULL;
    }

    if(normDowHist_ != NULL) {
        delete normDowHist_;
        normDowHist_ = NULL;
    }

    vector<ParseTree>().swap(ptPool_);
    ptPool_.clear();
    ptPoolState_.clear();

    TrainSampleSet().swap(posSet_);
    posSet_.clear();
    posFilterStat_.clear();
    posPtStat_.clear();

    TrainSampleSet().swap(negSet_);
    negSet_.clear();
    negFilterStat_.clear();
    negPtStat_.clear();

    vector<ParseTree>().swap(ptPoolCpy_);
    ptPoolCpy_.clear();
    ptPoolStateCpy_.clear();
    TrainSampleSet().swap(posSetCpy_);
    posSetCpy_.clear();
    TrainSampleSet().swap(negSetCpy_);
    negSetCpy_.clear();
}


template < int Dimension >
Scalar ParameterLearner::train(TrackerResult &input, int count,
                               int numFrameUsedToTrain, int numRelabel,
                               float fgOverlap, float maxBgOverlap,
                               bool nms, bool dividedByUnion, bool useOverlapLoss,
                               int maxNumEx, float C,
                               int partScale, bool restart,  bool useRootOnly,
                               vector<bool> &nodeOnOffHistory,
                               vector<bool> &edgeOnOffHistory,
                               string & saveDir) {

    if(grammar_->empty() || (grammar_->padx() < 1) || (grammar_->pady() < 1) ||
            (grammar_->interval() < 1) ||
            (count < 0) || (numRelabel < 1) || (fgOverlap <= 0.1) ||
            (C <= 0.0)) {
        RGM_LOG(error, "Invalid training parameters");
        return std::numeric_limits<double>::quiet_NaN();
    }

    RGM_CHECK((partScale == 0 || partScale == 1), error);

    // record the training loss
    double trainLoss = 0;

    // turn on createSample flag
    inferenceParam_.createSample_ = false;
    inferenceParam_.createRootSample2x_ = false;
    inferenceParam_.computeTNodeScores_ = false;
    inferenceParam_.useNMS_ = nms;
    inferenceParam_.nmsOverlap_ = fgOverlap;
    inferenceParam_.nmsDividedByUnion_ = dividedByUnion;
    inferenceParam_.useOverlapLoss_ = useOverlapLoss;
    inferenceParam_.clipPt_ = false;

    // get frame idx for training
    vector<int> frameIdx(input.getFrameIdxForTrain(count, numFrameUsedToTrain));

    // prepare data
    for(int j = 0; j < frameIdx.size(); ++j) {
        OneFrameResult & res(input.getOutput()[frameIdx[j]]);
        res.computeROIFeatPyr(grammar_->featParam());
    }

    // train root if needed
    if(restart) {
        trainRoot(input, count, numFrameUsedToTrain, maxBgOverlap,
                  dividedByUnion, maxNumEx, C, partScale, useRootOnly, true);
    }

    // init margin bound pruning
    initMarginBoundPruning();

    // record the hinge loss on positives before and after relabeling
    vector<std::pair<double, double> > posLoss(numRelabel,
                                               std::make_pair(0, 0));

    // record the cache information
    // per row: objVal on pos, objVal on neg, reg., total objVal
    Eigen::MatrixXd cache = Eigen::MatrixXd::Zero(numRelabel, 4);

    // record the hinge loss on neg
    vector<double> negLoss(numRelabel, 0);

    // State of using data mining
    bool keepDataMine = true;

    std::string strPosLoss = std::string("[%s] posLoss @ relabel: %d (%d),") +
            std::string("before: %f, after: %f, ratio: %f");
    boost::format frmtPosLoss(strPosLoss);
    boost::format frmtPosLossUp("[%s] Warning: posLoss went up");

    boost::format frmtPosPtStat("[%s] Parse tree usage stats (Pos relabel %d of %d):");
    boost::format frmtNegPtStat("[%s] Parse tree usage stats (Neg datamine %d of %d):");
    boost::format frmtPtStat("  parse tree %d got %d/%d (%.2f%%)");

    boost::format frmtStopRelabel("[%s] Stop relabeling since it does not reduce the posLoss by much");
    boost::format frmtNoChange("[%s] The model hasn't changed since the last data mining iteration.");

    boost::format frmtNegLoss("[%s] The hinge loss of negatives of old model is %f");
    boost::format frmtObjInfo("[%s] [datamine %d of %d] obj on full: %f, on cache: %f, ratio %f");
    boost::format frmtCacheInfo("[%s] Cache info: fg %f, bg %f, reg %f, total %f");
    boost::format frmtConverg("[%s] Hard negative mining convergence condition met.");

    boost::format frmtPosSV("[%s] Relable: %d(%d), #pos: %d, #posSV: %d");
    boost::format frmtNegSV("[%s] Datamine: %d(%d), #neg: %d, #negSV: %d");

    bool alldone = false;

    // vis
    if ( FileUtil::exists(saveDir) ) {
        string saveName = "AOG_" + NumToString_<int>(count+1) + "_Init";
        grammar_->visualize(saveDir, saveName);
    }

    // init pool
    ptPoolState_.assign(ptPoolState_.size(), false);
    clearPosSet();
    clearNegSet();
    vector<int> idxNegSV;

    for(int relabel = 0; relabel < numRelabel; ++ relabel) {
        alldone = false;
        // compute hinge loss on positives BEFORE relabeling
        posLoss[relabel].first = posSet_.computeLoss(true, C, 0);

        bool stopRelable = false;

        // get all latent pos and hard negs
        bool gotData = getTrackerTrainEx_ver2(input, frameIdx, maxNumEx, fgOverlap,
                                              maxBgOverlap, dividedByUnion);
        if(!gotData) {
            RGM_LOG(error, "*************not found data in getTrackerTrainEx");
            if ( relabel == 0 ) {
                return -std::numeric_limits<Scalar>::infinity();
            } else {
                break;
            }
        }

        // compute hinge loss on positives AFTER relabeling
        posLoss[relabel].second = posSet_.computeLoss(true, C, 0);

        // print the pos loss
        for(int i = 1, ii = 2; i < relabel; ++i, ++ii) {
            double r = posLoss[i].second / posLoss[i].first;
            RGM_LOG(normal, (frmtPosLoss % grammar_->name() % ii % numRelabel %
                             posLoss[i].first % posLoss[i].second % r));
        }

        // check the relabeling status
        if(relabel > 0 &&
                posLoss[relabel].second * 0.99999F > posLoss[relabel].first) {
            RGM_LOG(normal, (frmtPosLossUp % grammar_->name()));
            //            sleep(60); // wait 60 seconds
        }

        // stop if the relabeling doesn't help much
        if(relabel > 0 &&
                posLoss[relabel].second / posLoss[relabel].first > 0.999F) {
            stopRelable = true;
        }

        // print ptStat
        int relabel1 = relabel + 1;
        RGM_LOG(normal,
                (frmtPosPtStat % grammar_->name() % relabel1 % numRelabel));

        for(int i = 0; i < posPtStat_.size(); ++i) {
            int numPosExAdded = posSet_.size();
            float r = 100 * posPtStat_[i] / (float)numPosExAdded;
            RGM_LOG(normal,
                    (frmtPtStat % i % posPtStat_[i] % numPosExAdded % r));
        }

        //        if(stopRelable) {
        //            RGM_LOG(normal, (frmtStopRelabel % grammar_->name()));
        //            break;
        //        }

        if(!keepDataMine) {
            RGM_LOG(normal, (frmtNoChange % grammar_->name()));
            keepDataMine = true;
        } else {
            bool complete = ((negSet_.size() + posSet_.size()) < maxNumEx);

            // compute the hinge loss on negatives
            for(int i = 0; i < negSet_.size(); ++i) {
                negLoss[relabel] +=
                        std::max<Scalar>(0, 1 +
                                         negSet_[i].getPts()[0]->states()->score_);
            }
            negLoss[relabel] *= C;

            RGM_LOG(normal, (frmtNegLoss % grammar_->name() % negLoss[relabel]));

            // print the objective function information
            for(int i = 1, ii = 2; i <= relabel; ++i, ++ii) {
                double objVal  = cache(i - 1, 3);
                double fullVal = cache(i - 1, 3) - cache(i - 1, 1) + negLoss[i];
                double r = fullVal / objVal;
                RGM_LOG(normal,
                        (frmtObjInfo % grammar_->name() % ii % numRelabel % objVal % fullVal % r));
            }

            // check the status of keeping data mining
            if(relabel > 0 && complete) {
                double objVal  = cache(relabel - 1, 3);
                double fullVal = cache(relabel - 1, 3) - cache(relabel - 1, 1) + negLoss[relabel];
                if(fullVal / objVal < 1.05F) {
                    keepDataMine = false;
                }
            }

            // print ptStat
            RGM_LOG(normal,
                    (frmtNegPtStat % grammar_->name() % relabel1 % numRelabel));
            for(int i = 0; i < negPtStat_.size(); ++i) {
                int numAddedNeg = negSet_.size();
                float r = 100 * negPtStat_[i] / (float)numAddedNeg;
                RGM_LOG(normal,
                        (frmtPtStat % i % negPtStat_[i] % numAddedNeg % r));
            }

            //            if(!keepDataMine) {
            //                RGM_LOG(normal, (frmtConverg % grammar_->name()));
            //                break;
            //            }
        }

        // optimize the objective function using L-BFGS
        const int maxIterations = 1000;//std::min(std::max(10.0 * sqrt(static_cast<double>(posSet_.size())), 100.0), 1000.0);

        cache.row(relabel) = train(posSet_, negSet_, C, maxIterations);

        RGM_LOG(normal,
                (frmtCacheInfo % grammar_->name() % cache(relabel, 0) % cache(relabel, 1) % cache(relabel, 2) % cache(relabel, 3)));
        trainLoss = cache(relabel, 3);

        // grammar has changed
        grammar_->getCachedFFTStatus() = false;

        //grammar_->initAppFromRoot(partScale, -1);

        // count the number of support vectors
        int nbPosSV = 0;
        for(int i = 0; i < posSet_.size(); ++i) {
            for(int j = 0; j < posSet_[i].getPts().size(); ++j) {
                if(!posSet_[i].getPts()[j]->states()->isBelief_ &&
                        posSet_[i].getPts()[j]->states()->margin_ < 0.000001F) {
                    ++nbPosSV;
                }
            }
        }

        int nbNegSV = 0;
        idxNegSV.clear();
        idxNegSV.reserve(negSet_.size());
        for(int i = 0; i < negSet_.size(); ++i) {
            assert(negSet_[i].getPts().size() == 2);
            if(negSet_[i].getPts()[0]->states()->margin_ < 0.0001F) {
                idxNegSV.push_back(i);
                if(negSet_[i].getPts()[0]->states()->margin_ < 0.000001F) {
                    ++nbNegSV;
                }
            } /*else {
                for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                    for(int k = 0; k < ptPool_.size(); ++k) {
                        if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                            ptPoolState_[k] = false;
                            break;
                        }
                    }
                }
            }*/
        }

        // print the status of the training process
        RGM_LOG(normal,
                (frmtPosSV % grammar_->name() % relabel1 % numRelabel % posSet_.size() % nbPosSV));
        RGM_LOG(normal,
                (frmtNegSV % grammar_->name() % relabel1 % numRelabel % negSet_.size() % nbNegSV));

        // remove easy neg
        if(relabel == numRelabel - 1) {
            int maxNeg = (maxNumEx - posSet_.size()) / 2;

            if(negSet_.size() > maxNeg) {
                int i = 0;
                for(; i < idxNegSV.size(); ++i) {
                    negSet_[i].swap(negSet_[idxNegSV[i]]);
                }

                int numKept = std::max<int>(i, maxNeg);
                for(i = numKept; i < negSet_.size(); ++i) {
                    for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                        for(int k = 0; k < ptPool_.size(); ++k) {
                            if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                                ptPoolState_[k] = false;
                                break;
                            }
                        }
                    }
                }

                if(numKept < negSet_.size())
                    negSet_.resize(numKept);
            }
        }

        alldone = true;

    } // for relabel

    if(!alldone) {
        int maxNeg = (maxNumEx - posSet_.size()) / 2;

        if(negSet_.size() > maxNeg) {
            int i = 0;
            for(; i < idxNegSV.size(); ++i) {
                negSet_[i].swap(negSet_[idxNegSV[i]]);
            }

            int numKept = std::max<int>(i, maxNeg);
            for(i = numKept; i < negSet_.size(); ++i) {
                for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                    for(int k = 0; k < ptPool_.size(); ++k) {
                        if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                            ptPoolState_[k] = false;
                            break;
                        }
                    }
                }
            }

            if(numKept < negSet_.size())
                negSet_.resize(numKept);
        }
        //        int numKept = negSet_.size() * 0.7;
        //        for(int i = numKept; i < negSet_.size(); ++i) {
        //            for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
        //                for(int k = 0; k < ptPool_.size(); ++k) {
        //                    if(negSet_[i].getPts()[j] == &ptPool_[k]) {
        //                        ptPoolState_[k] = false;
        //                        break;
        //                    }
        //                }
        //            }
        //        }

        //        negSet_.resize(numKept);
    }


    grammar_->getOnOff(nodeOnOffHistory, edgeOnOffHistory);

    // vis
    if ( FileUtil::exists(saveDir) ) {
        string saveName = "AOG_" + NumToString_<int>(count+1) + "_Trained";
        grammar_->visualize(saveDir, saveName);
    }

#ifdef RGM_USE_AOG_TRACKING
    if ( posPtStat_.size() > 1 )
    {
        // refine the AOG
        //        int numTotal = posPtStat_[0];
        //        for(int i = 1; i < posPtStat_.size(); ++i)
        //            numTotal += posPtStat_[i];

        //        vector<int> prunePtIds;
        //        for(int i = 0; i < posPtStat_.size(); ++i) {
        //            float r = float(posPtStat_[i]) / numTotal;
        //            if(r < 0.1F) {
        //                prunePtIds.push_back(i);
        //            }
        //        }

        grammar_->setOnOff(false);
        for ( int i = 0; i < posSet_.size(); ++i ) {
            ParseTree * pt(posSet_[i].getPts()[0]);
            //            bool pruned = false;
            //            for(int j = 0; j < prunePtIds.size(); ++j) {
            //                if(prunePtIds[j] == pt->ptId()) {
            //                    pruned = true;
            //                    break;
            //                }
            //            }
            //            if(pruned) continue;

            for ( int j = 0; j < pt->nodeSet().size(); ++j ) {
                const PtNode * node(pt->nodeSet()[j]);
                int idx = node->idx()[PtNode::IDX_G];
                Node * gNode = grammar_->findNode(idx);
                gNode->getOnOff() = true;
            }
            for ( int j = 0; j < pt->edgeSet().size(); ++j ) {
                const PtEdge * edge(pt->edgeSet()[j]);
                int idx = edge->idx()[PtEdge::IDX_G];
                Edge * gEdge = grammar_->findEdge(idx);
                gEdge->getOnOff() = true;
            }
        }

        grammar_->traceDFSandBFS();
        grammar_->getCachedFFTStatus() = false;

        //    // change model type and reg. method: L2 should be used since the root is shared
        //    if(grammar_->isDAG()) {
        //        grammar_->getType() = GRAMMAR;
        //        grammar_->getRegMethod() = REG_L2;
        //    } else {
        //        grammar_->getType() = STARMIXTURE;
        //        grammar_->getRegMethod() = REG_MAX;
        //    }

        // get all latent pos and hard negs
        bool gotData = getTrackerTrainEx_ver2(input, frameIdx, maxNumEx, fgOverlap,
                                              maxBgOverlap, dividedByUnion);
        if(!gotData) {
            RGM_LOG(error, "*************not found data in getTrackerTrainEx");
            return -std::numeric_limits<Scalar>::infinity();
        }

        // optimize the objective function using L-BFGS
        const int maxIterations = 1000;//std::min(std::max(10.0 * sqrt(static_cast<double>(posSet_.size())), 100.0), 1000.0);

        train(posSet_, negSet_, C, maxIterations);

        // grammar has changed
        grammar_->getCachedFFTStatus() = false;

        //grammar_->initAppFromRoot(partScale, -1);

        // count the number of support vectors
        int nbPosSV = 0;
        for(int i = 0; i < posSet_.size(); ++i) {
            for(int j = 0; j < posSet_[i].getPts().size(); ++j) {
                if(!posSet_[i].getPts()[j]->states()->isBelief_ &&
                        posSet_[i].getPts()[j]->states()->margin_ < 0.000001F) {
                    ++nbPosSV;
                }
            }
        }

        int nbNegSV = 0;
        idxNegSV.clear();
        idxNegSV.reserve(negSet_.size());
        for(int i = 0; i < negSet_.size(); ++i) {
            assert(negSet_[i].getPts().size() == 2);
            if(negSet_[i].getPts()[0]->states()->margin_ < 0.0001F) {
                idxNegSV.push_back(i);
                if(negSet_[i].getPts()[0]->states()->margin_ < 0.000001F) {
                    ++nbNegSV;
                }
            }
        }

        // print the status of the training process
        int numRelabel1 = numRelabel + 1;
        RGM_LOG(normal,
                (frmtPosSV % grammar_->name() % numRelabel1 % numRelabel1 % posSet_.size() % nbPosSV));
        RGM_LOG(normal,
                (frmtNegSV % grammar_->name() % numRelabel1 % numRelabel1 % negSet_.size() % nbNegSV));

        // remove easy neg
        int maxNeg = (maxNumEx - posSet_.size()) / 2;

        if(negSet_.size() > maxNeg) {
            int i = 0;
            for(; i < idxNegSV.size(); ++i) {
                negSet_[i].swap(negSet_[idxNegSV[i]]);
            }

            int numKept = std::max<int>(i, maxNeg);
            for(i = numKept; i < negSet_.size(); ++i) {
                for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                    for(int k = 0; k < ptPool_.size(); ++k) {
                        if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                            ptPoolState_[k] = false;
                            break;
                        }
                    }
                }
            }

            if(numKept < negSet_.size())
                negSet_.resize(numKept);
        }


        //    int negIdx = 0;
        //    for ( int i = 0; i < negSet_.size(); ++i ) {
        //        ParseTree * pt(negSet_[i].getPts()[0]);
        //        bool done = false;
        //        for ( int j = 0; j < pt->nodeSet().size(); ++j ) {
        //            const PtNode * node(pt->nodeSet()[j]);
        //            int idx = node->idx()[PtNode::IDX_G];
        //            Node * gNode = grammar_->findNode(idx);
        //            if ( !gNode->onOff() )  {
        //                for(int p = 0; p < negSet_[i].getPts().size(); ++p) {
        //                    for(int k = 0; k < ptPool_.size(); ++k) {
        //                        if(negSet_[i].getPts()[p] == &ptPool_[k]) {
        //                            ptPoolState_[k] = false;
        //                            break;
        //                        }
        //                    }
        //                }
        //                done = true;
        //                break;
        //            }
        //        }

        //        if ( !done ) {
        //            for ( int j = 0; j < pt->edgeSet().size(); ++j ) {
        //                const PtEdge * edge(pt->edgeSet()[j]);
        //                int idx = edge->idx()[PtEdge::IDX_G];
        //                Edge * gEdge = grammar_->findEdge(idx);
        //                if ( !gEdge->onOff() )  {
        //                    for(int p = 0; p < negSet_[i].getPts().size(); ++p) {
        //                        for(int k = 0; k < ptPool_.size(); ++k) {
        //                            if(negSet_[i].getPts()[p] == &ptPool_[k]) {
        //                                ptPoolState_[k] = false;
        //                                break;
        //                            }
        //                        }
        //                    }
        //                    done = true;
        //                    break;
        //                }
        //            }
        //        }

        //        if ( !done ) {
        //            negSet_[negIdx++].swap(negSet_[i]);
        //        }
        //    }

        //    negSet_.resize(negIdx);


        // vis
        if ( FileUtil::exists(saveDir) ) {
            string saveName = "AOG_" + NumToString_<int>(count+1) + "_Tuned";
            grammar_->visualize(saveDir, saveName);
        }
    }

#endif

    // update app for T-nodes which are NOT upated in learning
    vector<int> idxUpdatedTNodes;
    idxUpdatedTNodes.reserve(grammar_->nodeSet().size());

    for ( int i = 0; i < posSet_.size(); ++i ) {
        ParseTree & pt(*posSet_[i].getPts()[0]);
        for ( int j = 0; j < pt.nodeSet().size(); ++j ) {
            const PtNode * node(pt.nodeSet()[j]);
            int idx = node->idx()[PtNode::IDX_G];
            Node * gNode = grammar_->findNode(idx);
            if ( gNode->type() != T_NODE )
                continue;

            idxUpdatedTNodes.push_back(idx);
        }
    }

    uniqueVector_<int>(idxUpdatedTNodes);

    grammar_->initAppFromRoot(partScale, idxUpdatedTNodes, true);

    return trainLoss;
}

template < int Dimension >
Scalar ParameterLearner::trainRoot(TrackerResult &input, int count,
                                   int numFrameUsedToTrain, float maxBgOverlap,
                                   bool dividedByUnion, int maxNumEx, float C,
                                   int partScale,  bool useRootOnly,
                                   bool useZeroPt) {
    // keep root template on only and reset parameters
    vector<int> ridx(1, 0);
    grammar_->turnOnRootOnly(ridx);

    // get frame idx for training
    vector<int> frameIdx(input.getFrameIdxForTrain(count, numFrameUsedToTrain));

    int modelWd = grammar_->maxDetectWindow().width();
    int modelHt = grammar_->maxDetectWindow().height();

    // prepare warped pos and roi feat pyr (for randneg)
    int numPos = 0;
    for(int i = 0; i < frameIdx.size(); ++i) {
        OneFrameResult &res(input.getOutput()[frameIdx[i]]);
        res.computeROIFeatPyr(grammar_->featParam());

        numPos += res.prepareShiftPos(modelWd, modelHt,
                                      grammar_->featParam());

        //        numPos += res.prepareWarpPos(modelWd, modelHt, (frameIdx[i] == 0),
        //                                     grammar_->featParam());


        if(frameIdx[i] == 0) {
            res.computeWarpedFeatPyr(grammar_->featParam());
        }
    }

    // init pool
    ptPoolState_.assign(ptPoolState_.size(), false);
    clearPosSet();
    clearNegSet();

    // init margin bound pruning
    initMarginBoundPruning();

    // compute pos training example
    numPos = std::min<int>(numPos, maxNumEx / 2);
    posSet_.resize(numPos);
    int numUsed = std::max<int>(1, numPos / frameIdx.size());
    int countPos = 0;

    for(int i = 0; i < frameIdx.size(); ++i) {
        OneFrameResult &res(input.getOutput()[frameIdx[i]]);
        int num = (frameIdx[i] == 0 ? res.warpedPosFeat_.size() :
                                      std::min<int>(numUsed, res.warpedPosFeat_.size()));
        for(int j = 0; j < num; ++j) {
            Level feat = res.warpedPosFeat_[j];
            TrainSample & example(posSet_[countPos++]);
            if(partScale == 0) {
                createOneRootSample(example, true, useZeroPt, feat, NULL);
            } else {
                createOneRootSample(example, true, useZeroPt, feat,
                                    &res.warpedPosFeatX_[j]);
            }
        }
    }
    posSet_.resize(countPos);

    // rand neg
    boost::random::mt19937 rng(3);

    int numNegToGet = maxNumEx - countPos;
    RGM_CHECK_GT(numNegToGet, 0);
    negSet_.resize(numNegToGet);
    int countNeg = 0;

    // get more neg from the recent frames
    vector<Scalar> areas;
    Scalar totalArea = 0;
    int startLevel = (partScale == 0 ? 0 : grammar_->interval());

    for(int i = frameIdx.size() - 1; i >= 0 ; i--) {
        OneFrameResult &res(input.getOutput()[frameIdx[i]]);
        FeaturePyr &pyr(*res.pyr_);
        totalArea = 0;
        areas.resize(pyr.nbLevels());
        for(int l = 0; l < pyr.nbLevels(); ++l) {
            areas[l] = pyr.levels()[l].rows() * pyr.levels()[l].cols();
            totalArea += areas[l];
        }

        Scalar factor = ((i == 0 || frameIdx[i] == 0) ? 1.0 : 0.5F);
        int curNumToGet = ROUND((numNegToGet - countNeg) * factor);
        int curGot = 0;

        Rectangle_<Scalar> gt(res.bbox_.x() - res.roi_.x(),
                              res.bbox_.y() - res.roi_.y(),
                              res.bbox_.width(), res.bbox_.height());

        float maxBgOverlapUsed = (frameIdx[i] == 0 ? 0.3F : maxBgOverlap);
        Intersector_<Scalar> inter(gt, maxBgOverlapUsed, dividedByUnion);
        Intersector_<Scalar> inter1(gt, 0.5F, false, true);

        for(int l = startLevel; l < pyr.nbLevels(); ++l) {
            if(countNeg >= numNegToGet || curGot >= curNumToGet)
                break;

            const Level & level(pyr.levels()[l]);
            Scalar scale = pyr.cellSize() / pyr.scales()[l];
            Scalar wd = modelWd * scale;
            Scalar ht = modelHt * scale;
            Scalar x1, y1, x2, y2;

            if(pyr.padx() + modelWd > level.cols() - pyr.padx() ||
                    pyr.pady() + modelHt > level.rows() - pyr.pady())
                continue;

            int xstart = pyr.padx();
            int ystart = pyr.pady();
            int xend = level.cols() - modelWd - pyr.padx();
            int yend = level.rows() - modelHt - pyr.pady();

            boost::random::uniform_int_distribution<> xrange(xstart, xend);
            boost::random::uniform_int_distribution<> yrange(ystart, yend);

            vector<int> addXs, addYs;

            int numToGet = floor(areas[l] / totalArea * curNumToGet);
            for(int n = 0; n < numToGet; ++n) {
                if(countNeg >= numNegToGet || curGot >= curNumToGet)
                    break;

                int x = xrange(rng);
                int y = yrange(rng);

                x1 = (x - pyr.padx()) * scale;
                x2 = std::min<Scalar>(x1 + wd, pyr.imgWd() - 1);
                y1 = (y - pyr.pady()) * scale;
                y2 = std::min<Scalar>(y1 + ht, pyr.imgHt() - 1);
                Rectangle_<Scalar> box(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

                if(inter(box) || inter1(box))
                    continue;

                bool visited = false;
                for(int k = 0; k < addXs.size(); ++k) {
                    if(std::abs(x - addXs[k]) < 3 &&
                            std::abs(y - addYs[k]) < 3) {
                        visited = true;
                        break;
                    }
                }
                if(visited) continue;

                addXs.push_back(x);
                addYs.push_back(y);

                Level feat = level.block(y, x, modelHt, modelWd);
                Level featx;
                if(partScale == 1) {
                    float scaleBase = grammar().featParam().scaleBase_;
                    x1 = ROUND((x - pyr.padx()) * scaleBase);
                    y1 = ROUND((y - pyr.pady()) * scaleBase);
                    const Level &levelx(pyr.levels()[l - pyr.interval()]);
                    featx = levelx.block(y1, x1, modelHt * 2, modelWd * 2);
                }
                TrainSample & example(negSet_[countNeg++]);
                curGot++;

                if(partScale == 0) {
                    createOneRootSample(example, false, useZeroPt, feat, NULL);
                } else {
                    createOneRootSample(example, false, useZeroPt, feat, &featx);
                }
            }
        }
    }
    negSet_.resize(countNeg);

    // Train the model
    const int maxIter = 1000;
    Eigen::VectorXd cache = train(posSet_, negSet_, C, maxIter);
    grammar_->getCachedFFTStatus() = false;

    // Retrain the model
    float fgOverlap = 0.7F;
    if ( getTrackerTrainEx_ver2(input, frameIdx, maxNumEx,
                                fgOverlap, maxBgOverlap, dividedByUnion) ) {
        cache = train(posSet_, negSet_, C, maxIter);
    }

    // count the number of support vectors
    int nbPosSV = 0;
    for(int i = 0; i < posSet_.size(); ++i) {
        for(int j = 0; j < posSet_[i].getPts().size(); ++j) {
            if(!posSet_[i].getPts()[j]->states()->isBelief_ &&
                    posSet_[i].getPts()[j]->states()->margin_ < 0.000001F) {
                ++nbPosSV;
            }
        }
    }

    int nbNegSV = 0;
    vector<int> idxNegSV;
    idxNegSV.reserve(negSet_.size());
    for(int i = 0; i < negSet_.size(); ++i) {
        assert(negSet_[i].getPts().size() == 2);
        if(negSet_[i].getPts()[0]->states()->margin_ < 0.0001F) {
            idxNegSV.push_back(i);
            if(negSet_[i].getPts()[0]->states()->margin_ < 0.000001F) {
                ++nbNegSV;
            }
        } else {
            for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                for(int k = 0; k < ptPool_.size(); ++k) {
                    if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                        ptPoolState_[k] = false;
                        break;
                    }
                }
            }
        }
    }

    // remove easy neg
    int i = 0;
    for(; i < idxNegSV.size(); ++i) {
        negSet_[i].swap(negSet_[idxNegSV[i]]);
    }
    negSet_.resize(i);

    //    int maxNeg = (maxNumEx - posSet_.size()) / 2 ;
    //    if ( negSet_.size() > maxNeg ) {
    //        int i = 0;
    //        for(; i < idxNegSV.size(); ++i) {
    //            negSet_[i].swap(negSet_[idxNegSV[i]]);
    //        }
    //        int numKept = std::max<int>(i, maxNeg);
    //        for(i = numKept; i < idxNegSV.size(); ++i) {
    //            for(int j = 0; j < negSet_[idxNegSV[i]].getPts().size(); ++j) {
    //                for(int k = 0; k < ptPool_.size(); ++k) {
    //                    if(negSet_[idxNegSV[i]].getPts()[j] == &ptPool_[k]) {
    //                        ptPoolState_[k] = false;
    //                        break;
    //                    }
    //                }
    //            }
    //        }
    //        if ( numKept < negSet_.size() )
    //            negSet_.resize(numKept);
    //    }

    RGM_LOG(normal, boost::format("[%s] trainRoot, nPosSV %d nNegSv %d")
            % grammar_->name() % nbPosSV % nbNegSV);

    if ( !useRootOnly ) {
        // init the structure of the AOG
        AOGrid & gridAOG(*grammar_->getGridAOG()[0]);

        // Get root appearance parameters
        Node * t = grammar_->getRootNode();
        while(t->type() != T_NODE) {
            t = t->getOutEdges()[0]->getToNode();
        }
        AppParam root = t->appearanceParam();

        AppParam rootx;
        if(partScale == 0) {
            rootx.swap(root);
        } else {
            FeaturePyr::resize(root, 2, rootx);
        }

        vector<ParseTree *> posPts(posSet_.getAllNonZeroPts());
        computeAOGridTermNodeScores(rootx, gridAOG, posPts, true);

        int betaRule = gridAOG.param().betaRule_;

        if(negSet_.size() == 0) {
            gridAOG.getParam().betaRule_ = 1;
        }

        if(gridAOG.param().betaRule_ == 0) {
            vector<ParseTree *> negPts(negSet_.getAllNonZeroPts());
            computeAOGridTermNodeScores(rootx, gridAOG, negPts, false);
        }

        gridAOG.parse();

        //    std::string strTmp("/home/tfwu/Tmp/");
        //    sAOG.visualize(strTmp, cv::Size(1000, 1000), 0, 4);

        // retrieve the part configuration using BFS
        vector<int> BFS(gridAOG.parseTreeBFS());

        gridAOG.getParam().betaRule_ = betaRule;

        // turn on scale prior if had
        vector<Node *> sub = grammar_->getSubcategoryRootAndNodes(false, false);
        for(int i = 0; i < sub.size(); ++i) {
            Node * n = sub[i];
            if(n->scaleprior() != NULL) {
                n->getScaleprior()->reset(true);
            }
        }


        // turn on nodes in grammar_ based on the computed BFS
        grammar_->setOnOff(0, BFS);

        //    bool isTree = true;

#ifdef RGM_USE_AOG_TRACKING
        // find equivalently good ones
        Scalar goodness = gridAOG.nodeSet()[gridAOG.BFSqueue()[0]].goodness_;
        while(gridAOG.findEqualBestChildForRootOr(goodness) ) {
            BFS = gridAOG.parseTreeBFS();
            grammar_->setOnOff(0, BFS);
            //        isTree = false;
        }
#endif

        // turn off the root template (not work yet)
        //    if ( partScale == 0 ) {
        //        for ( int i = 0; i < sub.size(); ++i ) {
        //            Edge * e = sub[i]->getOutEdges()[0];
        //            Node * n = e->getToNode();
        //            while ( n != NULL ) {
        //                e->getOnOff() = false;
        //                n->getOnOff() = false;
        //                if ( n->outEdges().size() == 0 ) break;
        //                e = n->getOutEdges()[0];
        //                n = e->getToNode();
        //            }
        //        }
        //    }

        grammar_->traceDFSandBFS();
        grammar_->initAppFromRoot(partScale, 0, true);
        grammar_->getCachedFFTStatus() = false;

#ifdef RGM_USE_AOG_TRACKING
        Scalar m = grammar_->computeMaxPtMemory();
        int num = maxMemoryInUse_ / m;
        num = std::min<int>(maxNumExInUse_, num);
        int num1 = num * (useZeroPt_ ? 2 : 1);

        clearPosSet();
        clearNegSet();

        ptPool_.resize(num1);
        ptPoolState_.assign(num1, false);
#endif

        // change model type and reg. method
        //    if(!isTree) {
        //        grammar_->getType() = GRAMMAR;
        //        grammar_->getRegMethod() = REG_L2;
        //    } else {
        //        grammar_->getType() = STARMIXTURE;
        //        grammar_->getRegMethod() = REG_MAX;
        //    }

        //        // change model type and reg. method: L2 should be used since the root is shared
        //        if(grammar_->isDAG()) {
        //            grammar_->getType() = GRAMMAR;
        //            grammar_->getRegMethod() = REG_L2;
        //        } else {
        //            grammar_->getType() = STARMIXTURE;
        //            grammar_->getRegMethod() = REG_MAX;
        //        }

        //        grammar_->visualize("/home/matt/Tmp/", "tmp");
    }

    return cache(3);
}

template <int Dimension>
Scalar ParameterLearner::trainOnline(OneFrameResult &res,
                                     std::vector<ParseTree> &pts, int maxNumEx,
                                     Scalar fgOverlap, Scalar maxBgOverlap,
                                     bool dividedByUnion,
                                     bool useZeroPt, int partScale, float C) {

    if(pts.size() == 0) return 0.0F;

    // assume pts[0] is postivie

    RGM_CHECK(partScale == 0 || partScale == 1, error);

    // init margin bound pruning
    initMarginBoundPruning();

    const ParseInfo *info = pts[0].rootParseInfo();
    Rectangle_<Scalar> ref(info->x(), info->y(), info->width(), info->height());
    Rectangle refi(ROUND(info->x()), ROUND(info->y()),
                   ROUND(info->width()), ROUND(info->height()));

    Intersector_<Scalar> inter(ref, fgOverlap, dividedByUnion);
    //    Intersector_<Scalar> inter(ref, 0.9, true);

    std::vector<int> isPos(pts.size(), 0);
    isPos[0] = 1;

    int numPosToAdd = 1;
    int numNegToAdd = 0;

#ifdef RGM_LEARNING_CHECK
    boost::format frmtNegNotMatch("********** On Negatives: Scores do not match: %f **********");
    boost::format frmtPosNotMatch("********** On Positives: Scores do not match: %f **********");
#endif

    inferenceParam_.createSample_ = true;

    Scalar ov;
    for(int i = 1; i < pts.size(); ++i) {
        const ParseInfo * info1 = pts[i].rootParseInfo();
        Rectangle_<Scalar> box(info1->x(), info1->y(),
                               info1->width(), info1->height());

        //        if ( inter(box, &ov) ) {
        //            isPos[i] = 1;
        //        } else if ( ov < maxBgOverlap ) {
        //            isPos[i] = -1;
        //        }
        bool isOv = inter(box, &ov);
        if(inferenceParam_.useOverlapLoss_ && isOv) {
            isPos[i] = 1;
            numPosToAdd++;
        }

        if(ov < maxBgOverlap) {
            isPos[i] = -1;
            numNegToAdd++;
        }
    }

    int numPts = numPtsToFill();

    if(numPts <= numPosToAdd /*|| numNegToAdd == 0*/) {
        return 0.0F;
    }

    // add pos
    for(int i = 0; i < pts.size(); ++i) {
        if(isPos[i] != 1)
            continue;
        int ptIdx;
        ParseTree *ptInPool = getPtInPool(&ptIdx);
        if(ptInPool == NULL) break;
        ParseTree *pt0InPool = NULL;
        if(useZeroPt) {
            pt0InPool = getPtInPool();
            if(pt0InPool == NULL) {
                ptPoolState_[ptIdx] = false;
                break;
            }
        }


        ptInPool->swap(pts[i]);

        posSet_.resize(posSet_.size() + 1);

        TrainSample & ex(posSet_.back());

        ex.getPts().resize((useZeroPt ? 2 : 1));

        // State information for margin-bound pruning
        ex.getNbHist() = 0;
        ex.getMarginBound() = -1.0F;

        ex.getPts()[0] = ptInPool;

        ex.getPts()[0]->createSample(*res.pyr_, inferenceParam_);

        // states
        bool isBelief = true;
        Scalar score = ptInPool->rootParseInfo()->score_;
        Scalar loss = inferenceParam_.useOverlapLoss_ ?
                    ptInPool->computeOverlapLoss(refi) :
                    ptInPool->rootParseInfo()->loss_;
        Scalar norm = ptInPool->norm();
        ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

#ifdef RGM_LEARNING_CHECK
        Scalar s = grammar_->dot(*ptInPool);
        Scalar diff = std::abs(s - score);
        if(diff > 1e-5) {
            RGM_LOG(error, frmtPosNotMatch % diff);
        }
#endif
        ex.getBeliefNorm() = norm;

        if(useZeroPt) {
            // associated with the background output (zero pt)
            // for a foreground example

            // states
            isBelief = false;
            score = 0;
            loss = 1;
            norm = 0;
            ex.getPts()[1] = pt0InPool;
            pt0InPool->getStates() = new PtStates(isBelief, score, loss, norm);

            ex.getMaxNonbeliefNorm() = norm;
        }
    }

    int numGot = posSet_.size() + negSet_.size();

    for(int i = 0; i < pts.size(); ++i) {
        if(isPos[i] != -1)
            continue;

        if(numGot++ >= maxNumEx)
            break;
        int ptIdx;
        ParseTree *ptInPool = getPtInPool(&ptIdx);
        if(ptInPool == NULL) break;
        ParseTree *pt0InPool = NULL;
        if(useZeroPt) {
            pt0InPool = getPtInPool();
            if(pt0InPool == NULL) {
                ptPoolState_[ptIdx] = false;
                break;
            }
        }

        ptInPool->swap(pts[i]);

        negSet_.resize(negSet_.size() + 1);
        TrainSample & ex(negSet_.back());

        ex.getPts().resize((useZeroPt ? 2 : 1));

        // State information for margin-bound pruning
        ex.getNbHist() = 0;
        ex.getMarginBound() = -1.0F;

        ex.getPts()[0] = ptInPool;

        ex.getPts()[0]->createSample(*res.pyr_, inferenceParam_);

        // states
        bool isBelief = false;
        Scalar score = ptInPool->rootParseInfo()->score_;
        Scalar loss = 1;
        Scalar norm = ptInPool->norm();
        ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

        ex.getMaxNonbeliefNorm() = norm;

#ifdef RGM_LEARNING_CHECK
        Scalar s = grammar_->dot(*ptInPool);
        Scalar diff = std::abs(s - score);
        if(diff  > 1e-5) {
            RGM_LOG(error, frmtNegNotMatch % diff);
            return -1;
        }
#endif

        if(useZeroPt) {
            // associated with the belief (zero pt) for a background example

            // states
            isBelief = true;
            score = 0;
            loss = 0;
            norm = 0;
            ex.getPts()[1] = pt0InPool;
            pt0InPool->getStates() = new PtStates(isBelief, score, loss, norm);

            ex.getBeliefNorm() = norm;
        }
    }

    inferenceParam_.createSample_ = false;

    const int maxIter = 1000;
    Eigen::VectorXd cache = train(posSet_, negSet_, C, maxIter);

    // count the number of support vectors
    int nbPosSV = 0;
    std::vector<int> idxPosSV;
    for(int i = 0; i < posSet_.size(); ++i) {
        for(int j = 0; j < posSet_[i].getPts().size(); ++j) {
            if(!posSet_[i].getPts()[j]->states()->isBelief_) {
                if(posSet_[i].getPts()[j]->states()->margin_ < 0.0001F) {
                    idxPosSV.push_back(i);
                    if(posSet_[i].getPts()[j]->states()->margin_ < 0.000001F)
                        ++nbPosSV;
                }
            }
        }
    }


    int nbNegSV = 0;
    std::vector<int> idxNegSV;
    idxNegSV.reserve(negSet_.size());
    for(int i = 0; i < negSet_.size(); ++i) {
        assert(negSet_[i].getPts().size() == 2);
        if(negSet_[i].getPts()[0]->states()->margin_ < 0.0001F) {
            idxNegSV.push_back(i);
            if(negSet_[i].getPts()[0]->states()->margin_ < 0.000001F) {
                ++nbNegSV;
            }
        } /*else {
            for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                for(int k = 0; k < ptPool_.size(); ++k) {
                    if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                        ptPoolState_[k] = false;
                        break;
                    }
                }
            }
        }*/
    }

    // remove easy neg
    int maxNeg = (maxNumEx - posSet_.size()) / 2;

    if(negSet_.size() > maxNeg) {
        int i = 0;
        for(; i < idxNegSV.size(); ++i) {
            negSet_[i].swap(negSet_[idxNegSV[i]]);
        }

        int numKept = std::max<int>(i, maxNeg);
        for(i = numKept; i < negSet_.size(); ++i) {
            for(int j = 0; j < negSet_[i].getPts().size(); ++j) {
                for(int k = 0; k < ptPool_.size(); ++k) {
                    if(negSet_[i].getPts()[j] == &ptPool_[k]) {
                        ptPoolState_[k] = false;
                        break;
                    }
                }
            }
        }

        if(numKept < negSet_.size())
            negSet_.resize(numKept);
    }

    RGM_LOG(normal, boost::format("[%s] online training, nPosSV %d nNegSv %d")
            % grammar_->name() % nbPosSV % nbNegSV);

    //    std::vector<int> idxUpdatedTNodes;
    //    idxUpdatedTNodes.reserve(grammar_->nodeSet().size());
    //    for ( int i = 0; i < positives.size(); ++i ) {
    //        ParseTree & pt(positives[i].getPts()[0]);
    //        for ( int j = 0; j < pt.nodeSet().size(); ++j ) {
    //            const PtNode * node(pt.nodeSet()[j]);
    //            int idx = node->idx()[PtNode::IDX_G];
    //            Node * gNode = grammar_->findNode(idx);
    //            if ( gNode->type() != Node::T_NODE )
    //                continue;

    //            idxUpdatedTNodes.push_back(idx);
    //        }
    //    }
    //    uniqueVector_<int>(idxUpdatedTNodes);

    //    grammar_->initAppFromRoot(partScale, idxUpdatedTNodes, false);

    grammar_->getCachedFFTStatus() = false;

    return cache(3);

    //    return 1;
}

template <int Dimension>
Eigen::VectorXd ParameterLearner::train(TrainSampleSet & positives,
                                        TrainSampleSet & negatives,
                                        Scalar C, int maxIter) {
    Loss_<Dimension> loss(*this, positives, negatives, C, maxIter);

    LBFGS lbfgs(&loss);

    // Start from the current models
    Eigen::VectorXd x(loss.dim());
    Eigen::VectorXd lb(x.size());

    grammar_->getParameters(x.data(), 0);
    grammar_->getParameters(lb.data(), 1);

    // 0 - charles dubout's LBFGS implementation, minConf, otherwise
    int method = 1;
    const double l = lbfgs(x.data(), lb.data(), method);

    return loss.computeCacheInfo();
}

template <int Dimension>
ParseTree * ParameterLearner::getPtInPool(int * idx) {
    int i = 0;
    for(; i < ptPoolState_.size(); ++i) {
        if(!ptPoolState_[i])
            break;
    }

    if(i >= ptPoolState_.size())
        return NULL;

    ptPoolState_[i] = true;

    ptPool_[i].clear();
    ptPool_[i].setGrammar(*grammar_);

    if(idx != NULL) {
        *idx = i;
    }

    return &ptPool_[i];
}

template <int Dimension>
void ParameterLearner::clearPosSet() {
    for(int i = 0; i < posSet_.size(); ++i) {
        TrainSample & pos(posSet_[i]);
        for(int j = 0; j < pos.getPts().size(); ++j) {
            bool found = false;
            for(int k = 0; k < ptPool_.size(); ++k) {
                if(pos.getPts()[j] == &ptPool_[k]) {
                    found = true;
                    ptPoolState_[k] = false;
                    ptPool_[k].clear();
                    ptPool_[k].setGrammar(*grammar_);
                    break;
                }
            }
            RGM_CHECK(found, error);
        }
    }

    posSet_.clear();
}

template <int Dimension>
void ParameterLearner::clearNegSet() {
    for(int i = 0; i < negSet_.size(); ++i) {
        TrainSample & neg(negSet_[i]);
        for(int j = 0; j < neg.getPts().size(); ++j) {
            bool found = false;
            for(int k = 0; k < ptPool_.size(); ++k) {
                if(neg.getPts()[j] == &ptPool_[k]) {
                    found = true;
                    ptPoolState_[k] = false;
                    ptPool_[k].clear();
                    ptPool_[k].setGrammar(*grammar_);
                    break;
                }
            }
            RGM_CHECK(found, error);
        }
    }

    negSet_.clear();
}

template <int Dimension>
int ParameterLearner::numPtsToFill() {
    int num = 0;

    for(int i = 0; i < ptPoolState_.size(); ++i) {
        num += (ptPoolState_[i] ? 0 : 1);
    }

    if(useZeroPt_)
        num *= 0.5F;

    return num;
}

template <int Dimension>
int ParameterLearner::szPtPool() {
    int num = ptPool_.size();
    if(useZeroPt_)
        num *= 0.5F;

    return num;
}

template <int Dimension>
bool ParameterLearner::checkPtPool() {
    int numEx1 = posSet_.size() + negSet_.size();
    int numEx2 = ptPoolState_.size() * (useZeroPt_ ? 0.5F : 1) - numPtsToFill();
    return (numEx1 == numEx2);
}


template <int Dimension>
ParseTree * ParameterLearner::createOneRootPt(Level & feat) {
    ParseTree * pt = getPtInPool();
    if(pt == NULL) {
        return NULL;
    }

    const vector<Node *> & DFS(grammar_->nodeDFS());

    int wd = DFS[0]->detectWindow().width();
    RGM_CHECK_EQ(wd, feat.cols());
    int ht = DFS[0]->detectWindow().height();
    RGM_CHECK_EQ(ht, feat.rows());

    for(int i = 0; i < DFS.size(); ++i) {
        const Node * gn = DFS[i];
        int idxGn = grammar_->idxNode(gn);
        int t = static_cast<int>(gn->type());

        int idxPt = pt->addNode(idxGn, t);
        PtNode * ptn = pt->getNodeSet()[idxPt];

        // set bias and scale prior for object AND-node
        if(gn->bias() != NULL) {
            ptn->getIdx()[PtNode::IDX_BIAS] =
                    pt->AddBias(grammar_->featureBias());
        }

        // set scale prior
        if(gn->scaleprior() != NULL) {
            Scaleprior::Param prior;
            prior.setZero();
            ptn->getIdx()[PtNode::IDX_SCALEPRIOR] = pt->addScaleprior(prior);
        }

        switch(gn->type()) {
        case T_NODE: {
            ptn->getIdx()[PtNode::IDX_APP] =
                    pt->addAppearance(feat, grammar_->featType(), false);
            break;
        }
        case AND_NODE: {
            for(int j = 0; j < gn->outEdges().size(); ++j) {
                const Edge * ge = gn->outEdges()[j];
                int idxGe = grammar_->idxEdge(ge);
                vector<const PtNode *> ptnch =
                        pt->findNode(ge->toNode());
                for(int k = 0; k < ptnch.size(); ++k) {
                    pt->addEdge(idxPt, ptnch[k]->idx()[PtNode::IDX_MYSELF],
                            idxGe, ge->type());
                }
            }
            break;
        }
        case OR_NODE: {
            bool found = false;
            for(int j = 0; j < gn->outEdges().size(); ++j) {
                const Edge * ge = gn->outEdges()[j];
                int idxGe = grammar_->idxEdge(ge);
                vector<const PtNode *> ptnch =
                        pt->findNode(ge->toNode());
                if(ptnch.size() == 1) {
                    pt->addEdge(idxPt, ptnch[0]->idx()[PtNode::IDX_MYSELF],
                            idxGe, ge->type());
                    found = true;
                    break;
                }
            }
            RGM_CHECK(found, error);

            break;
        }
        }
    }

    return pt;
}

template <int Dimension>
bool ParameterLearner::createOneRootSample(TrainSample & sample,
                                           bool isPos, bool useZeroPt,
                                           Level & feat,
                                           Level * featx) {
    ParseTree * pt0 = createOneRootPt(feat);
    ParseTree * pt1 = useZeroPt ? getPtInPool() : NULL;
    if(pt0 == NULL || (useZeroPt && pt1 == NULL))
        return false;

    if(featx != NULL) {
        if(pt0->getAppearaceX() != NULL) {
            delete pt0->getAppearaceX();
            pt0->getAppearaceX() = NULL;
        }
        pt0->getAppearaceX() = new AppParam(*featx);
    }

    sample.getPts().resize(useZeroPt ? 2 : 1, NULL);

    // State information for margin-bound pruning
    sample.getNbHist() = 0;
    sample.getMarginBound() = -1.0F;

    // states
    bool isBelief = isPos;
    Scalar loss = isPos ? 0 : 1;
    Scalar norm = pt0->norm();
    pt0->getStates() = new PtStates(isBelief, loss, norm);

    sample.getPts()[0] = pt0;

    sample.getBeliefNorm() = norm;

    if(useZeroPt) {
        // zero pt
        // associated with the background output for
        // a foreground example

        // states
        isBelief = !isPos;
        loss = isPos ? 1 : 0;
        norm = 0;
        pt1->getStates() =
                new PtStates(isBelief, loss, norm);

        sample.getPts()[1] = pt1;

        sample.getMaxNonbeliefNorm() = norm;
    }

    return true;
}


template <int Dimension>
bool ParameterLearner::getTrackerTrainEx(TrackerResult & input,
                                         const vector<int> &frameIdx,
                                         int maxNumEx,
                                         Scalar fgOverlap, Scalar maxBgOverlap,
                                         bool dividedByUnion, bool useZeroPt) {
    // caching fft
    grammar_->cachingFilters();

    maxNumEx = std::min<int>(maxNumEx, ptPool_.size());

    // ptStat
    int numPt = grammar_->rootNode()->outEdges().size() - 1;
    Node * objAnd = grammar_->getRootNode()->getOutEdges()[0]->getToNode();
    int idxEdge = (objAnd->idxOutEdge().size() > 1 ? 1 : 0);
    Node * objConfigOr = objAnd->getOutEdges()[idxEdge]->getToNode();
    for(int i = 0; i < objConfigOr->outEdges().size(); ++i) {
        if(objConfigOr->outEdges()[i]->onOff()) {
            ++numPt;
        }
    }
    if(numPt == 0) {
        posPtStat_.assign(1, 0);
        negPtStat_.assign(1, 0);
    } else {
        posPtStat_.assign(numPt, 0);
        negPtStat_.assign(numPt, 0);
    }

    const Scalar thr = std::min<Scalar>(grammar_->thresh() - 0.5F, -1.002F); //-1.002F; //

    vector<vector<ParseTree> > allPts(frameIdx.size() + 1);
    vector<vector<int> > isPos(frameIdx.size() + 1);
    vector<vector<int> > allPtId(frameIdx.size() + 1);

    Scalar detLimit = maxNumEx / std::max<int>(4, frameIdx.size());

#ifdef RGM_LEARNING_CHECK
    boost::format strPosScoreNotMatch(" *********** On Positives: Scores do not match: %f ***********");
    boost::format strNegScoreNotMatch(" *********** On Negatives: Scores do not match: %f ***********");
#endif

    Scalar initFgOverlap = 0.65F;

    //    #pragma omp parallel for num_threads(4)
    for(int i = 0; i < frameIdx.size(); ++i) {
        OneFrameResult & res(input.getOutput()[frameIdx[i]]);
        FeaturePyr &pyr(*(res.pyr_));

        DPInference inference(*grammar_, inferenceParam_);

        inference.runDetection(thr, pyr, detLimit, allPts[i]);

        //        for ( int j = 0; j < allPts[i].size(); ++j )
        //            allPts[i][j].showDetection(res.img_, true);

        Rectangle_<Scalar> gt(res.bbox_.x() - res.roi_.x(),
                              res.bbox_.y() - res.roi_.y(),
                              res.bbox_.width(),
                              res.bbox_.height());

        Intersector_<Scalar> inter(gt, (frameIdx[i] == 0 ? initFgOverlap : fgOverlap), dividedByUnion);
        Intersector_<Scalar> inter1(gt, 0.5F, false, true);

        // divide into pos. vs. neg
        int num = allPts[i].size();

        isPos[i].assign(num, 0);
        allPtId[i].assign(num, -1);

        Rectangle_<Scalar> box;
        Scalar ov;
        bool isOv;
        Scalar maxov = 0;
        int maxIdx = -1;

        vector<int> posPtIdx;
        bool foundPos = false;
        for(int j = 0; j < num; ++j) {
            ParseTree &pt(allPts[i][j]);
            box = static_cast<Rectangle_<Scalar> >(*pt.rootParseInfo());

            if(numPt > 0) {
                PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
                int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
                PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(idxEdge, objPtAnd));
                PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
                PtNode * objConfigPtAnd = pt.getToNode(edge);
                int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
                const Node * objConfigAnd = grammar_->findNode(idxG);
                RGM_CHECK_NOTNULL(objConfigAnd);

                for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                    if(!objConfigOr->outEdges()[k]->onOff())
                        continue;
                    if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                        allPtId[i][j] = kk;
                        break;
                    }
                    ++kk;
                }
                RGM_CHECK_NOTEQ(allPtId[i][j], -1);
            } else {
                allPtId[i][j] = 0;
            }

            pt.getPtId() = allPtId[i][j];

            isOv = inter(box, &ov);

            if(ov > maxov) {
                maxov = ov;
                maxIdx = j;
            }

            //            isPos[i][maxIdx] = 1;
            //            posPtIdx.push_back(maxIdx);

            if(isOv) {
                foundPos = true;
                //                if(inferenceParam_.useOverlapLoss_) {
                //                    isPos[i][j] = 1;
                //                    posPtIdx.push_back(j);
                //                }
            } else if(ov < maxBgOverlap && (frameIdx[i] == 0 || !inter1(box))) {
                isPos[i][j] = -1;
            }
        }

        if(foundPos) {
            int k = 0;
            for(; k < posPtIdx.size(); ++k) {
                if(posPtIdx[k] == maxIdx) break;
            }
            if(posPtIdx.size() == 0 || k < posPtIdx.size()) {
                isPos[i][maxIdx] = 1;
                posPtIdx.push_back(maxIdx);
            }
            //allPts[i][maxIdx].showDetection(res.img_(res.roi_.cvRect()), true);
        } else {
            RGM_LOG(warning, "*************** not found pos in poslatent");
            if(frameIdx[i] > 0) {
                isPos[i].clear();
                allPts[i].clear();
            }
        }

        inference.release();

        for (int j = 0; j < isPos[i].size(); ++j ) {
            if ( isPos[i][j] == 0 ) {
                allPts[i][j].clear();
            }
        }
    }

#if 1
    // warp pos
    OneFrameResult & res(input.getOutput()[0]);
    for(int j = 0; j < res.warpedPyrs_.size(); ++j) {
        FeaturePyr &pyramid(*res.warpedPyrs_[j]);
        pyramid.getValidLevels().assign(pyramid.validLevels().size(), true);

        DPInference inference(*grammar_, inferenceParam_);

        if(!inference.runDP(pyramid)) continue;

        float factor = 1; //res.workingScale_ *  grammar().featParam().scaleBase_;
        vector<Rectangle> gts(1);
        gts[0] = Rectangle(ROUND((res.warpedBbox_.x()) * factor),
                           ROUND((res.warpedBbox_.y()) * factor),
                           ROUND(res.warpedBbox_.width() * factor),
                           ROUND(res.warpedBbox_.height() * factor));

        // compute overlap maps and index of valid levels
        // each level, each box, and each obj comp.
        vector<vector<vector<Matrix> > > overlapMaps;
        pyramid.getValidLevels() =
                inference.computeOverlapMaps(gts, pyramid, overlapMaps, fgOverlap);

        // get pos
        // apply output inhibition
        inference.inhibitOutput(0, overlapMaps, fgOverlap, false);
        ParseTree posPt;
        if(!inference.runParsing(-1000.0F, pyramid, posPt)) {
            RGM_LOG(warning, "*************** not found pos in poslatent (warp)");
            continue;
        }

        // debug
        //                posPt.showDetection(res.warpedImgs_[j], true);

        if(numPt > 0) {
            int ptId = -1;
            PtNode * objPtAnd = posPt.getToNode(posPt.getOutEdge(0, posPt.rootNode()));
            int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
            PtNode * objConfigPtOr = posPt.getToNode(posPt.getOutEdge(idxEdge, objPtAnd));
            PtEdge * edge = posPt.getOutEdge(0, objConfigPtOr);
            PtNode * objConfigPtAnd = posPt.getToNode(edge);
            int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
            const Node * objConfigAnd = grammar_->findNode(idxG);
            RGM_CHECK_NOTNULL(objConfigAnd);

            for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                if(!objConfigOr->outEdges()[k]->onOff())
                    continue;
                if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                    ptId = kk;
                    break;
                }
                ++kk;
            }
            RGM_CHECK_NOTEQ(ptId, -1);
            allPtId[frameIdx.size()].push_back(ptId);
        } else {
            allPtId[frameIdx.size()].push_back(0);
        }

        posPt.getPtId() = allPtId[frameIdx.size()].back();

        isPos[frameIdx.size()].push_back(1);
        allPts[frameIdx.size()].push_back(posPt);

        inference.release();
    }
#endif

    int numPos = 0;
    int numNeg = 0;
    vector<int> posPtId;
    for(int i = 0; i < isPos.size(); ++i) {
        for(int j = 0; j < isPos[i].size(); ++j) {
            if(isPos[i][j] == 1) {
                numPos++;
                posPtId.push_back(allPtId[i][j]);
            } else if(isPos[i][j] == -1) {
                numNeg++;
            }
        }
    }

    if(numPos == 0)
        return false;

    if(numNeg == 0) {
        OneFrameResult & res(input.getOutput()[0]);
        FeaturePyr &pyr(*(res.pyr_));
        pyr.getValidLevels().assign(pyr.validLevels().size(), true);

        DPInference inference(*grammar_, inferenceParam_);

        vector<ParseTree> pts;
        inference.runDetection(-100, pyr, detLimit, pts);

        Rectangle_<Scalar> gt(res.bbox_.x() - res.roi_.x(),
                              res.bbox_.y() - res.roi_.y(),
                              res.bbox_.width(),
                              res.bbox_.height());

        Intersector_<Scalar> inter(gt, initFgOverlap, dividedByUnion);

        // divide into pos. vs. neg
        int num = pts.size();

        Rectangle_<Scalar> box;
        Scalar ov;

        for(int j = 0; j < num; ++j) {
            if(numNeg >= maxNumEx - numPos) break;
            ParseTree &pt(pts[j]);
            box = static_cast<Rectangle_<Scalar> >(*pt.rootParseInfo());

            if(numPt > 0) {
                PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
                int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
                PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(idxEdge, objPtAnd));
                PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
                PtNode * objConfigPtAnd = pt.getToNode(edge);
                int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
                const Node * objConfigAnd = grammar_->findNode(idxG);
                RGM_CHECK_NOTNULL(objConfigAnd);

                for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                    if(!objConfigOr->outEdges()[k]->onOff())
                        continue;
                    if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                        allPtId[0].push_back(kk);
                        break;
                    }
                    ++kk;
                }
            } else {
                allPtId[0].push_back(0);
            }

            pt.getPtId() = allPtId[0].back();

            inter(box, &ov);
            if(ov < maxBgOverlap) {
                isPos[0].push_back(-1);
                allPts[0].push_back(pt);
                ++numNeg;
            }
        }
    }

    if(numNeg == 0)
        return false;

    uniqueVector_<int>(posPtId);

    if(numPos + numNeg > maxNumEx) {
        vector<std::pair<Scalar, int> > neg(numNeg);
        vector<std::pair<int, int> > idx(numNeg);
        for(int i = 0, k = 0; i < isPos.size(); ++i) {
            for(int j = 0; j < isPos[i].size(); ++j) {
                if(isPos[i][j] == -1) {
                    ParseTree &pt(allPts[i][j]);
                    Scalar score = pt.score();
                    neg[k] = std::make_pair(-score, k);
                    idx[k] = std::make_pair(i, j);
                    ++k;
                }
            }
        }
        std::sort(neg.begin(), neg.end()); // ascending

        int numToGet = maxNumEx - numPos;
        numNeg = 0;
        bool found;
        int n = 0;
        for(; n < neg.size() && numNeg < numToGet; ++n) {
            int k = neg[n].second;
            found = false;
            for(int i = 0; i < posPtId.size(); ++i) {
                if(allPtId[idx[k].first][idx[k].second] == posPtId[i]) {
                    found = true;
                    numNeg++;
                    break;
                }
            }
            if(!found) {
                isPos[idx[k].first][idx[k].second] = 0;
            }
        }

        for(; n < neg.size(); ++n) {
            int k = neg[n].second;
            isPos[idx[k].first][idx[k].second] = 0;
        }

        if(numNeg == 0)
            return false;
    }

    // create training examples

    // init pool
    ptPoolState_.assign(ptPoolState_.size(), false);
    clearPosSet();
    clearNegSet();

    posSet_.resize(numPos);
    negSet_.resize(numNeg);
    int idxPos = 0;
    int idxNeg = 0;
    int idxPtPool = 0;

    bool isPoolFull = false;
    for(int i = 0; i < allPts.size(); ++i) {
        if(isPoolFull) break;

        vector<ParseTree> & pts(allPts[i]);
        int dataId = (i < frameIdx.size() ? frameIdx[i] : 0);

        OneFrameResult & res(input.getOutput()[dataId]);

        Rectangle gt;
        if(i == frameIdx.size()) {
            gt = res.warpedBbox_;
        } else {
            gt = Rectangle(res.bbox_.x() - res.roi_.x(),
                           res.bbox_.y() - res.roi_.y(),
                           res.bbox_.width(), res.bbox_.height());
        }

        for(int j = 0; j < pts.size(); ++j) {
            ParseTree &curPt(pts[j]);
            curPt.getDataId() = dataId;

            int curPtId = allPtId[i][j];

            if(isPos[i][j] == 1) {
                ParseTree * ptInPool = getPtInPool(&idxPtPool);
                if(ptInPool == NULL) {
                    isPoolFull = true;
                    break;
                }
                ParseTree * pt0InPool = NULL;
                if ( useZeroPt ) {
                    pt0InPool = getPtInPool();
                    if( pt0InPool == NULL) {
                        ptPoolState_[idxPtPool] = false;
                        isPoolFull = true;
                        break;
                    }
                }
                posPtStat_[curPtId] += 1;
                ptInPool->swap(curPt);

                TrainSample & ex(posSet_[idxPos++]);

                ex.getPts().resize((useZeroPt ? 2 : 1));

                // State information for margin-bound pruning
                ex.getNbHist() = 0;
                ex.getMarginBound() = -1.0F;

                ex.getPts()[0] = ptInPool;

                // states
                bool isBelief = true;
                Scalar score = ptInPool->rootParseInfo()->score_;
                Scalar loss = inferenceParam_.useOverlapLoss_ ?
                            ptInPool->computeOverlapLoss(gt) :
                            ptInPool->rootParseInfo()->loss_;
                Scalar norm = ptInPool->norm();
                ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

#ifdef RGM_LEARNING_CHECK
                if(!inferenceParam_.createRootSample2x_) {
                    Scalar s = grammar_->dot(*ptInPool);
                    Scalar diff = std::abs(s - score);
                    if(diff > 1e-5) {
                        RGM_LOG(error, strPosScoreNotMatch % diff);
                    }
                }
#endif
                ex.getBeliefNorm() = norm;

                if(useZeroPt) {
                    ex.getPts()[1] = pt0InPool;

                    // associated with the background output (zero pt)
                    // for a foreground example

                    // states
                    isBelief = false;
                    score = 0;
                    loss = 1;
                    norm = 0;
                    pt0InPool->getStates() = new PtStates(isBelief, score,
                                                          loss, norm);

                    ex.getMaxNonbeliefNorm() = norm;
                }

            } else if(isPos[i][j] == -1 && idxNeg < numNeg) {
                ParseTree * ptInPool = getPtInPool(&idxPtPool);
                if(ptInPool == NULL) {
                    isPoolFull = true;
                    break;
                }
                ParseTree * pt0InPool = NULL;
                if ( useZeroPt ) {
                    pt0InPool = getPtInPool();
                    if( pt0InPool == NULL) {
                        ptPoolState_[idxPtPool] = false;
                        isPoolFull = true;
                        break;
                    }
                }

                negPtStat_[curPtId] += 1;

                ptInPool->swap(curPt);

                TrainSample & ex(negSet_[idxNeg++]);

                ex.getPts().resize((useZeroPt ? 2 : 1));

                // State information for margin-bound pruning
                ex.getNbHist() = 0;
                ex.getMarginBound() = -1.0F;

                ex.getPts()[0] = ptInPool;

                // states
                bool isBelief = false;
                Scalar score = ptInPool->rootParseInfo()->score_;
                Scalar loss = 1;
                Scalar norm = ptInPool->norm();
                ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

                ex.getMaxNonbeliefNorm() = norm;

#ifdef RGM_LEARNING_CHECK
                if(!inferenceParam_.createRootSample2x_) {
                    Scalar s = grammar_->dot(*ptInPool);
                    Scalar diff = std::abs(s - score);
                    if(diff  > 1e-5) {
                        RGM_LOG(error, strNegScoreNotMatch % diff);
                    }
                }
#endif

                if(useZeroPt) {

                    ex.getPts()[1] = pt0InPool;

                    // associated with the belief (zero pt) for a background example

                    // states
                    isBelief = true;
                    score = 0;
                    loss = 0;
                    norm = 0;
                    pt0InPool->getStates() = new PtStates(isBelief, score, loss, norm);

                    ex.getBeliefNorm() = norm;
                }
            }
        }
    }

    posSet_.resize(idxPos);
    negSet_.resize(idxNeg);

    return true;
}


template <int Dimension>
bool ParameterLearner::getTrackerTrainEx_ver2(TrackerResult & input,
                                              const vector<int> &frameIdx,
                                              int maxNumEx,
                                              Scalar fgOverlap, Scalar maxBgOverlap,
                                              bool dividedByUnion, bool useZeroPt) {
    // caching fft
    grammar_->cachingFilters();

    maxNumEx = std::min<int>(maxNumEx, ptPool_.size());

    // ptStat
    int numPt = grammar_->rootNode()->outEdges().size() - 1;
    Node * objAnd = grammar_->getRootNode()->getOutEdges()[0]->getToNode();
    int idxEdge = (objAnd->idxOutEdge().size() > 1 ? 1 : 0);
    Node * objConfigOr = objAnd->getOutEdges()[idxEdge]->getToNode();
    for(int i = 0; i < objConfigOr->outEdges().size(); ++i) {
        if(objConfigOr->outEdges()[i]->onOff()) {
            ++numPt;
        }
    }
    if(numPt == 0) {
        posPtStat_.assign(1, 0);
        negPtStat_.assign(1, 0);
    } else {
        posPtStat_.assign(numPt, 0);
        negPtStat_.assign(numPt, 0);
    }

    const Scalar thr = std::min<Scalar>(grammar_->thresh() - 0.5F, -1.002F); //-1.002F; //

    vector<vector<ParseTree> > allPts(frameIdx.size() + 1);
    vector<vector<int> > isPos(frameIdx.size() + 1);
    vector<vector<int> > allPtId(frameIdx.size() + 1);

    Scalar detLimit = maxNumEx / std::min<int>(4, frameIdx.size());

#ifdef RGM_LEARNING_CHECK
    boost::format strPosScoreNotMatch(" *********** On Positives: Scores do not match: %f ***********");
    boost::format strNegScoreNotMatch(" *********** On Negatives: Scores do not match: %f ***********");
#endif

    Scalar initFgOverlap = 0.65F;

#pragma omp parallel for //num_threads(4)
    for(int i = 0; i < frameIdx.size(); ++i) {
        OneFrameResult & res(input.getOutput()[frameIdx[i]]);
        FeaturePyr &pyr(*(res.pyr_));

        DPInference inference(*grammar_, inferenceParam_);

        inference.runDetection(thr, pyr, detLimit, allPts[i]);

        //                for ( int j = 0; j < allPts[i].size() && j < 10; ++j ) {
        //                    cv::Mat img = res.img_.clone();
        //                    allPts[i][j].showDetection(img, true);
        //                }

        Rectangle_<Scalar> gt(res.bbox_.x() - res.roi_.x(),
                              res.bbox_.y() - res.roi_.y(),
                              res.bbox_.width(),
                              res.bbox_.height());

        Intersector_<Scalar> inter(gt, (frameIdx[i] == 0 ? initFgOverlap : fgOverlap), dividedByUnion);
        Intersector_<Scalar> inter1(gt, 0.5F, false, true);

        // divide into pos. vs. neg
        int num = allPts[i].size();

        isPos[i].assign(num, 0);
        allPtId[i].assign(num, -1);

        Rectangle_<Scalar> box;
        Scalar ov;
        bool isOv;
        Scalar maxov = 0;
        int maxIdx = -1;

        vector<int> posPtIdx;
        bool foundPos = false;
        for(int j = 0; j < num; ++j) {
            ParseTree &pt(allPts[i][j]);
            box = static_cast<Rectangle_<Scalar> >(*pt.rootParseInfo());

            if(numPt > 0) {
                PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
                int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
                PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(idxEdge, objPtAnd));
                PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
                PtNode * objConfigPtAnd = pt.getToNode(edge);
                int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
                const Node * objConfigAnd = grammar_->findNode(idxG);
                RGM_CHECK_NOTNULL(objConfigAnd);

                for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                    if(!objConfigOr->outEdges()[k]->onOff())
                        continue;
                    if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                        allPtId[i][j] = kk;
                        break;
                    }
                    ++kk;
                }
                RGM_CHECK_NOTEQ(allPtId[i][j], -1);
            } else {
                allPtId[i][j] = 0;
            }

            pt.getPtId() = allPtId[i][j];

            isOv = inter(box, &ov);

            if(ov > maxov) {
                maxov = ov;
                maxIdx = j;
            }

            //            isPos[i][maxIdx] = 1;
            //            posPtIdx.push_back(maxIdx);

            if(isOv) {
                foundPos = true;
                //                if(inferenceParam_.useOverlapLoss_) {
                //                    isPos[i][j] = 1;
                //                    posPtIdx.push_back(j);
                //                }
            } else if(ov < maxBgOverlap && (frameIdx[i] == 0 || !inter1(box))) {
                isPos[i][j] = -1;
            }
        }

        if(foundPos) {
            int k = 0;
            for(; k < posPtIdx.size(); ++k) {
                if(posPtIdx[k] == maxIdx) break;
            }
            if(posPtIdx.size() == 0 || k < posPtIdx.size()) {
                isPos[i][maxIdx] = 1;
                posPtIdx.push_back(maxIdx);
            }
            //allPts[i][maxIdx].showDetection(res.img_(res.roi_.cvRect()), true);
        } else {
            RGM_LOG(warning, "*************** not found pos in poslatent");
            if(frameIdx[i] > 0) {
                isPos[i].clear();
                allPts[i].clear();
            }
        }

        inference.release();
    }

    inferenceParam_.createSample_ = true;

#if 1
    // warp pos
    OneFrameResult & res(input.getOutput()[0]);
    for(int j = 0; j < res.warpedPyrs_.size(); ++j) {
        FeaturePyr &pyramid(*res.warpedPyrs_[j]);
        pyramid.getValidLevels().assign(pyramid.validLevels().size(), true);

        DPInference inference(*grammar_, inferenceParam_);

        if(!inference.runDP(pyramid)) continue;

        vector<Rectangle> gts;
        gts.push_back(res.warpedBbox_);

        // compute overlap maps and index of valid levels
        // each level, each box, and each obj comp.
        vector<vector<vector<Matrix> > > overlapMaps;
        pyramid.getValidLevels() =
                inference.computeOverlapMaps(gts, pyramid, overlapMaps, fgOverlap);

        // get pos
        // apply output inhibition
        inference.inhibitOutput(0, overlapMaps, fgOverlap, false);
        ParseTree posPt;
        if(!inference.runParsing(-1000.0F, pyramid, posPt)) {
            RGM_LOG(warning, "*************** not found pos in poslatent (warp)");
            continue;
        }

        // debug
        //                        posPt.showDetection(res.warpedImgs_[j], true);

        if(numPt > 0) {
            int ptId = -1;
            PtNode * objPtAnd = posPt.getToNode(posPt.getOutEdge(0, posPt.rootNode()));
            int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
            PtNode * objConfigPtOr = posPt.getToNode(posPt.getOutEdge(idxEdge, objPtAnd));
            PtEdge * edge = posPt.getOutEdge(0, objConfigPtOr);
            PtNode * objConfigPtAnd = posPt.getToNode(edge);
            int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
            const Node * objConfigAnd = grammar_->findNode(idxG);
            RGM_CHECK_NOTNULL(objConfigAnd);

            for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                if(!objConfigOr->outEdges()[k]->onOff())
                    continue;
                if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                    ptId = kk;
                    break;
                }
                ++kk;
            }
            RGM_CHECK_NOTEQ(ptId, -1);
            allPtId[frameIdx.size()].push_back(ptId);
        } else {
            allPtId[frameIdx.size()].push_back(0);
        }

        posPt.getPtId() = allPtId[frameIdx.size()].back();

        isPos[frameIdx.size()].push_back(1);
        allPts[frameIdx.size()].push_back(posPt);

        inference.release();
    }
#endif

    int numPos = 0;
    int numNeg = 0;
    vector<int> posPtId;
    for(int i = 0; i < isPos.size(); ++i) {
        for(int j = 0; j < isPos[i].size(); ++j) {
            if(isPos[i][j] == 1) {
                numPos++;
                posPtId.push_back(allPtId[i][j]);
            } else if(isPos[i][j] == -1) {
                numNeg++;
            }
        }
    }

    if(numPos == 0)
        return false;

    if(numNeg == 0) {
        OneFrameResult & res(input.getOutput()[0]);
        FeaturePyr &pyr(*(res.pyr_));
        pyr.getValidLevels().assign(pyr.validLevels().size(), true);

        DPInference inference(*grammar_, inferenceParam_);

        vector<ParseTree> pts;
        inference.runDetection(-100, pyr, detLimit, pts);

        Rectangle_<Scalar> gt(res.bbox_.x() - res.roi_.x(),
                              res.bbox_.y() - res.roi_.y(),
                              res.bbox_.width(),
                              res.bbox_.height());

        Intersector_<Scalar> inter(gt, initFgOverlap, dividedByUnion);

        // divide into pos. vs. neg
        int num = pts.size();

        Rectangle_<Scalar> box;
        Scalar ov;

        for(int j = 0; j < num; ++j) {
            if(numNeg >= maxNumEx - numPos) break;
            ParseTree &pt(pts[j]);
            box = static_cast<Rectangle_<Scalar> >(*pt.rootParseInfo());

            if(numPt > 0) {
                PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
                int idxEdge = (objPtAnd->idxOutEdges().size() > 1 ? 1 : 0);
                PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(idxEdge, objPtAnd));
                PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
                PtNode * objConfigPtAnd = pt.getToNode(edge);
                int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
                const Node * objConfigAnd = grammar_->findNode(idxG);
                RGM_CHECK_NOTNULL(objConfigAnd);

                for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                    if(!objConfigOr->outEdges()[k]->onOff())
                        continue;
                    if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                        allPtId[0].push_back(kk);
                        break;
                    }
                    ++kk;
                }
            } else {
                allPtId[0].push_back(0);
            }

            pt.getPtId() = allPtId[0].back();

            inter(box, &ov);
            if(ov < maxBgOverlap) {
                isPos[0].push_back(-1);
                allPts[0].push_back(pt);
                ++numNeg;
            }
        }
    }

    if(numNeg == 0)
        return false;

    uniqueVector_<int>(posPtId);

    if(numPos + numNeg > maxNumEx) {
        vector<std::pair<Scalar, int> > neg(numNeg);
        vector<std::pair<int, int> > idx(numNeg);
        for(int i = 0, k = 0; i < isPos.size(); ++i) {
            for(int j = 0; j < isPos[i].size(); ++j) {
                if(isPos[i][j] == -1) {
                    ParseTree &pt(allPts[i][j]);
                    Scalar score = pt.score();
                    neg[k] = std::make_pair(-score, k);
                    idx[k] = std::make_pair(i, j);
                    ++k;
                }
            }
        }
        std::sort(neg.begin(), neg.end()); // ascending

        int numToGet = maxNumEx - numPos;
        numNeg = 0;
        bool found;
        int n = 0;
        for(; n < neg.size() && numNeg < numToGet; ++n) {
            int k = neg[n].second;
            found = false;
            for(int i = 0; i < posPtId.size(); ++i) {
                if(allPtId[idx[k].first][idx[k].second] == posPtId[i]) {
                    found = true;
                    numNeg++;
                    break;
                }
            }
            if(!found) {
                isPos[idx[k].first][idx[k].second] = 0;
            }
        }

        for(; n < neg.size(); ++n) {
            int k = neg[n].second;
            isPos[idx[k].first][idx[k].second] = 0;
        }

        if(numNeg == 0)
            return false;
    }

    // create training examples

    // init pool
    ptPoolState_.assign(ptPoolState_.size(), false);
    clearPosSet();
    clearNegSet();

    posSet_.resize(numPos);
    negSet_.resize(numNeg);
    int idxPos = 0;
    int idxNeg = 0;
    int idxPtPool = 0;

    bool isPoolFull = false;
    for(int i = 0; i < allPts.size(); ++i) {
        if(isPoolFull) break;

        vector<ParseTree> & pts(allPts[i]);
        int dataId = (i < frameIdx.size() ? frameIdx[i] : 0);

        OneFrameResult & res(input.getOutput()[dataId]);

        Rectangle gt;
        if(i == frameIdx.size()) {
            gt = res.warpedBbox_;
        } else {
            gt = Rectangle(res.bbox_.x() - res.roi_.x(),
                           res.bbox_.y() - res.roi_.y(),
                           res.bbox_.width(), res.bbox_.height());
        }

        for(int j = 0; j < pts.size(); ++j) {
            ParseTree &curPt(pts[j]);
            curPt.getDataId() = dataId;

            int curPtId = allPtId[i][j];

            if(isPos[i][j] == 1) {
                ParseTree * ptInPool = getPtInPool(&idxPtPool);
                if(ptInPool == NULL) {
                    isPoolFull = true;
                    break;
                }
                ParseTree * pt0InPool = NULL;
                if ( useZeroPt ) {
                    pt0InPool = getPtInPool();
                    if( pt0InPool == NULL) {
                        ptPoolState_[idxPtPool] = false;
                        isPoolFull = true;
                        break;
                    }
                }
                posPtStat_[curPtId] += 1;
                ptInPool->swap(curPt);

                TrainSample & ex(posSet_[idxPos++]);

                ex.getPts().resize((useZeroPt ? 2 : 1));

                // State information for margin-bound pruning
                ex.getNbHist() = 0;
                ex.getMarginBound() = -1.0F;

                ex.getPts()[0] = ptInPool;

                if (i < frameIdx.size())
                    ex.getPts()[0]->createSample(*res.pyr_, inferenceParam_);

                // states
                bool isBelief = true;
                Scalar score = ptInPool->rootParseInfo()->score_;
                Scalar loss = inferenceParam_.useOverlapLoss_ ?
                            ptInPool->computeOverlapLoss(gt) :
                            ptInPool->rootParseInfo()->loss_;
                Scalar norm = ptInPool->norm();
                ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

#ifdef RGM_LEARNING_CHECK
                if(!inferenceParam_.createRootSample2x_) {
                    Scalar s = grammar_->dot(*ptInPool);
                    Scalar diff = std::abs(s - score);
                    if(diff > 1e-5) {
                        RGM_LOG(error, strPosScoreNotMatch % diff);
                    }
                }
#endif
                ex.getBeliefNorm() = norm;

                if(useZeroPt) {
                    ex.getPts()[1] = pt0InPool;

                    // associated with the background output (zero pt)
                    // for a foreground example

                    // states
                    isBelief = false;
                    score = 0;
                    loss = 1;
                    norm = 0;
                    pt0InPool->getStates() = new PtStates(isBelief, score,
                                                          loss, norm);

                    ex.getMaxNonbeliefNorm() = norm;
                }

            } else if(isPos[i][j] == -1 && idxNeg < numNeg) {
                ParseTree * ptInPool = getPtInPool(&idxPtPool);
                if(ptInPool == NULL) {
                    isPoolFull = true;
                    break;
                }
                ParseTree * pt0InPool = NULL;
                if ( useZeroPt ) {
                    pt0InPool = getPtInPool();
                    if( pt0InPool == NULL) {
                        ptPoolState_[idxPtPool] = false;
                        isPoolFull = true;
                        break;
                    }
                }

                negPtStat_[curPtId] += 1;

                ptInPool->swap(curPt);

                TrainSample & ex(negSet_[idxNeg++]);

                ex.getPts().resize((useZeroPt ? 2 : 1));

                // State information for margin-bound pruning
                ex.getNbHist() = 0;
                ex.getMarginBound() = -1.0F;

                ex.getPts()[0] = ptInPool;

                if (i < frameIdx.size())
                    ex.getPts()[0]->createSample(*res.pyr_, inferenceParam_);

                // states
                bool isBelief = false;
                Scalar score = ptInPool->rootParseInfo()->score_;
                Scalar loss = 1;
                Scalar norm = ptInPool->norm();
                ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

                ex.getMaxNonbeliefNorm() = norm;

#ifdef RGM_LEARNING_CHECK
                if(!inferenceParam_.createRootSample2x_) {
                    Scalar s = grammar_->dot(*ptInPool);
                    Scalar diff = std::abs(s - score);
                    if(diff  > 1e-5) {
                        RGM_LOG(error, strNegScoreNotMatch % diff);
                    }
                }
#endif

                if(useZeroPt) {

                    ex.getPts()[1] = pt0InPool;

                    // associated with the belief (zero pt) for a background example

                    // states
                    isBelief = true;
                    score = 0;
                    loss = 0;
                    norm = 0;
                    pt0InPool->getStates() = new PtStates(isBelief, score, loss, norm);

                    ex.getBeliefNorm() = norm;
                }
            }
        }
    }

    posSet_.resize(idxPos);
    negSet_.resize(idxNeg);

    inferenceParam_.createSample_ = false;

    return true;
}

template <int Dimension>
bool ParameterLearner::getTrackerTrainEx1(TrackerResult & input,
                                          const vector<int> &frameIdx,
                                          int maxNumEx,
                                          Scalar fgOverlap, Scalar maxBgOverlap,
                                          bool dividedByUnion, bool useZeroPt) {
    // caching fft
    grammar_->cachingFilters();

    // init pool
    ptPoolState_.assign(ptPoolState_.size(), false);
    clearPosSet();
    clearNegSet();

    maxNumEx = std::min<int>(maxNumEx, ptPool_.size());

    // ptStat
    int numPt = grammar_->rootNode()->outEdges().size() - 1;
    Node * objAnd = grammar_->getRootNode()->getOutEdges()[0]->getToNode();
    Node * objConfigOr = objAnd->getOutEdges()[1]->getToNode();
    for(int i = 0; i < objConfigOr->outEdges().size(); ++i) {
        if(objConfigOr->outEdges()[i]->onOff()) {
            ++numPt;
        }
    }
    if(numPt == 0) {
        posPtStat_.assign(1, 0);
        negPtStat_.assign(1, 0);
    } else {
        posPtStat_.assign(numPt, 0);
        negPtStat_.assign(numPt, 0);
    }

    const Scalar thr = std::min<Scalar>(grammar_->thresh() - 0.5F, -1.002F);

    vector<ParseTree> allPosPts(frameIdx.size());
    vector<int> allPosPtId(frameIdx.size());

    vector<vector<ParseTree> > allNegPts(frameIdx.size());
    vector<vector<int> > allNegPtId(frameIdx.size());

    Scalar detLimit = maxNumEx / frameIdx.size();

    boost::format strPosScoreNotMatch(" *********** On Positives: Scores do not match: %f ***********");
    boost::format strNegScoreNotMatch(" *********** On Negatives: Scores do not match: %f ***********");

    //#pragma omp parallel for
    for(int i = 0; i < frameIdx.size(); ++i) {
        OneFrameResult & res(input.getOutput()[frameIdx[i]]);
        FeaturePyr &pyr(*(res.pyr_));
        pyr.getValidLevels().assign(pyr.getValidLevels().size(), true);

        DPInference inference(*grammar_, inferenceParam_);

        if(!inference.runDP(pyr)) continue;

        float factor = 1.0F; //res.workingScale_ *  grammar().featParam().scaleBase_;
        vector<Rectangle> gt(1);
        gt[0] = Rectangle(ROUND((res.bbox_.x() - res.roi_.x()) * factor),
                          ROUND((res.bbox_.y() - res.roi_.y()) * factor),
                          ROUND(res.bbox_.width() * factor),
                          ROUND(res.bbox_.height() * factor));

        // compute overlap maps and index of valid levels
        // each level, each box, and each obj comp.
        vector<vector<vector<Matrix> > > overlapMaps;
        vector<bool> validLevels =
                inference.computeOverlapMaps(gt, pyr, overlapMaps, fgOverlap);

        bool needCpy = true;
        vector<Node *> sub = grammar_->getSubcategoryRootAndNodes(true, true);
        for(int c = 0; c < sub.size(); ++c) {
            inference.copyScoreMaps(sub[c]);
        }

        // get pos
        allPosPtId[i] = -1;
        pyr.getValidLevels() = validLevels;
        // apply output inhibition
        inference.inhibitOutput(0, overlapMaps, fgOverlap, needCpy);
        if(!inference.runParsing((frameIdx[i] == 0 ? -1000 : thr), pyr, allPosPts[i])) {
            RGM_LOG(warning, "*************** not found pos in poslatent");
            continue;
        }

        // get pos pt id
        if(numPt > 0) {
            ParseTree &pt(allPosPts[i]);
            PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
            PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(1, objPtAnd));
            PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
            PtNode * objConfigPtAnd = pt.getToNode(edge);
            int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
            const Node * objConfigAnd = grammar_->findNode(idxG);
            RGM_CHECK_NOTNULL(objConfigAnd);

            for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                if(!objConfigOr->outEdges()[k]->onOff())
                    continue;
                if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                    allPosPtId[i] = kk;
                    break;
                }
                ++kk;
            }
            RGM_CHECK_NOTEQ(allPosPtId[i], -1);
        } else {
            allPosPtId[i] = 0;
        }

        // restore the score maps
        for(int c = 0; c < sub.size(); ++c) {
            inference.recoverScoreMaps(sub[c]);
        }

        // get neg
        pyr.getValidLevels().assign(pyr.getValidLevels().size(), true);
        inference.inhibitAllFg(overlapMaps, -1, maxBgOverlap, false);
        inference.runParsing(thr, pyr, detLimit, allNegPts[i]);

        int num = allNegPts[i].size();
        allNegPtId[i].assign(num, -1);
        for(int j = 0; j < num; ++j) {
            ParseTree &pt(allNegPts[i][j]);
            if(numPt > 0) {
                PtNode * objPtAnd = pt.getToNode(pt.getOutEdge(0, pt.rootNode()));
                PtNode * objConfigPtOr = pt.getToNode(pt.getOutEdge(1, objPtAnd));
                PtEdge * edge = pt.getOutEdge(0, objConfigPtOr);
                PtNode * objConfigPtAnd = pt.getToNode(edge);
                int idxG = objConfigPtAnd->idx()[PtNode::IDX_G];
                const Node * objConfigAnd = grammar_->findNode(idxG);
                RGM_CHECK_NOTNULL(objConfigAnd);

                for(int k = 0, kk = 0; k < objConfigOr->outEdges().size(); ++k) {
                    if(!objConfigOr->outEdges()[k]->onOff())
                        continue;
                    if(objConfigAnd == objConfigOr->outEdges()[k]->toNode()) {
                        allNegPtId[i][j] = kk;
                        break;
                    }
                    ++kk;
                }
                RGM_CHECK_NOTEQ(allNegPtId[i][j], -1);
            } else {
                allNegPtId[i][j] = 0;
            }
        }

    }

    int numPos = 0;
    vector<int> posPtId;
    for(int i = 0; i < allPosPtId.size(); ++i) {
        if(allPosPtId[i] != -1) {
            ++numPos;
            posPtId.push_back(allPosPtId[i]);
        }
    }
    if(numPos == 0)
        return false;

    uniqueVector_<int>(posPtId);

    int numNeg = 0;
    for(int i = 0; i < allNegPtId.size(); ++i) {
        numNeg += allNegPtId[i].size();
    }
    numNeg = std::min<int>(numNeg, maxNumEx - numPos);

    // create training examples
    posSet_.resize(numPos);
    negSet_.resize(numNeg);

    int idxPos = 0;
    int idxNeg = 0;
    int idxPtPool = 0;

    bool isPoolFull = false;
    for(int i = 0; i < allPosPts.size(); ++i) {
        if(isPoolFull) break;

        int curPtId = allPosPtId[i];
        if(curPtId == -1) continue;

        int dataId = frameIdx[i];
        OneFrameResult & res(input.getOutput()[dataId]);
        Rectangle gt(res.bbox_.x() - res.roi_.x(),
                     res.bbox_.y() - res.roi_.y(),
                     res.bbox_.width(), res.bbox_.height());

        ParseTree & pt(allPosPts[i]);
        pt.getDataId() = dataId;

        posPtStat_[curPtId] += 1;

        ParseTree * ptInPool = getPtInPool(&idxPtPool);
        ParseTree * pt0InPool = (useZeroPt ? getPtInPool(&idxPtPool) : NULL);
        if(ptInPool == NULL || (useZeroPt && pt0InPool == NULL)) {
            isPoolFull = true;
            break;
        }
        ptInPool->swap(pt);

        TrainSample & ex(posSet_[idxPos++]);

        ex.getPts().resize((useZeroPt ? 2 : 1));

        // State information for margin-bound pruning
        ex.getNbHist() = 0;
        ex.getMarginBound() = -1.0F;

        ex.getPts()[0] = ptInPool;

        // states
        bool isBelief = true;
        Scalar score = ptInPool->rootParseInfo()->score_;
        Scalar loss = inferenceParam_.useOverlapLoss_ ?
                    ptInPool->computeOverlapLoss(gt) :
                    ptInPool->rootParseInfo()->loss_;
        Scalar norm = ptInPool->norm();
        ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

#ifdef RGM_LEARNING_CHECK
        if(!inferenceParam_.createRootSample2x_) {
            Scalar s = grammar_->dot(*ptInPool);
            Scalar diff = std::abs(s - score);
            if(diff > 1e-5) {
                RGM_LOG(error, strPosScoreNotMatch % diff);
            }
        }
#endif
        ex.getBeliefNorm() = norm;

        if(useZeroPt) {
            ex.getPts()[1] = pt0InPool;

            // associated with the background output (zero pt)
            // for a foreground example

            // states
            isBelief = false;
            score = 0;
            loss = 1;
            norm = 0;
            pt0InPool->getStates() = new PtStates(isBelief, score,
                                                  loss, norm);

            ex.getMaxNonbeliefNorm() = norm;
        }
    }

    for(int i = 0; i < allNegPts.size(); ++i) {
        if(isPoolFull) break;
        vector<ParseTree> &pts(allNegPts[i]);

        int dataId = frameIdx[i];

        for(int j = 0; j < pts.size(); ++j) {
            ParseTree &curPt(pts[j]);
            curPt.getDataId() = dataId;
            int curPtId = allNegPtId[i][j];

            if(idxNeg < numNeg) {
                negPtStat_[curPtId] += 1;

                ParseTree * ptInPool = getPtInPool(&idxPtPool);
                ParseTree * pt0InPool = (useZeroPt ? getPtInPool(&idxPtPool) : NULL);
                if(ptInPool == NULL || (useZeroPt && pt0InPool == NULL)) {
                    isPoolFull = true;
                    break;
                }

                ptInPool->swap(curPt);

                TrainSample & ex(negSet_[idxNeg++]);

                ex.getPts().resize((useZeroPt ? 2 : 1));

                // State information for margin-bound pruning
                ex.getNbHist() = 0;
                ex.getMarginBound() = -1.0F;

                ex.getPts()[0] = ptInPool;

                // states
                bool isBelief = false;
                Scalar score = ptInPool->rootParseInfo()->score_;
                Scalar loss = 1;
                Scalar norm = ptInPool->norm();
                ptInPool->getStates() = new PtStates(isBelief, score, loss, norm);

                ex.getMaxNonbeliefNorm() = norm;

#ifdef RGM_LEARNING_CHECK
                if(!inferenceParam_.createRootSample2x_) {
                    Scalar s = grammar_->dot(*ptInPool);
                    Scalar diff = std::abs(s - score);
                    if(diff  > 1e-5) {
                        RGM_LOG(error, strNegScoreNotMatch % diff);
                    }
                }
#endif

                if(useZeroPt) {

                    ex.getPts()[1] = pt0InPool;

                    // associated with the belief (zero pt) for a background example

                    // states
                    isBelief = true;
                    score = 0;
                    loss = 0;
                    norm = 0;
                    pt0InPool->getStates() = new PtStates(isBelief, score, loss, norm);

                    ex.getBeliefNorm() = norm;
                }
            }
        }
    }

    posSet_.resize(idxPos);
    negSet_.resize(idxNeg);

    return true;
}

template <int Dimension>
void ParameterLearner::computeAOGridTermNodeScores(AppParam & w,
                                                   AOGrid & grid,
                                                   vector<ParseTree *> & pts,
                                                   bool isPos) {
    int numPt = pts.size();
    if(numPt == 0)
        return;

    const int shiftDT = Deformation::BoundedShiftInDT;
    int shift = shiftDT * 2 + 1;
    int shift2 = std::pow(shift, 2);

    // compute score maps for each cell
    vector<vector<Matrix> > cellScoreMaps(numPt);

    if(isPos) {
        grid.getPscores().resize(numPt);
        grid.getPdxdy().resize(numPt);
    }

#pragma omp parallel for
    for(int i = 0; i < numPt; ++i) {
        ParseTree & pt(*pts[i]);

        const AppParam * app =
                (pt.appearaceX() == NULL ? pt.appearanceSet()[0] : pt.appearaceX());
        const AppParam & F(*app);

        vector<Matrix> & scoreMaps(cellScoreMaps[i]);
        scoreMaps.resize(shift2, Matrix::Zero(w.rows(), w.cols()));

        for(int y = 0; y < w.rows(); ++y) {
            for(int x = 0; x < w.cols(); ++x) {
                Cell wc = w(y, x);

                for(int sy = -shiftDT, s = 0; sy <= shiftDT; ++sy) {
                    int yy = sy + y;
                    for(int sx = -shiftDT; sx <= shiftDT; ++sx, ++s) {
                        int xx = sx + x;
                        if(yy < 0 || yy >= w.rows() ||
                                xx < 0 || xx >= w.cols())  {
                            scoreMaps[s](y, x)  = 0;
                            //wc(FeaturePyr::NbFeatures-1); // truncation
                        } else {
                            scoreMaps[s](y, x) =
                                    wc.cwiseProduct(F(yy, xx)).sum();
                        }
                    } // for sx
                } // for sy
            } // for x
        } // for y

        if(isPos) {
            grid.getPscores()[i] = -std::numeric_limits<Scalar>::infinity();

            for(int sy = -shiftDT, s = 0; sy <= shiftDT; ++sy) {
                for(int sx = -shiftDT; sx <= shiftDT; ++sx, ++s) {
                    Scalar tmp = scoreMaps[s].sum();
                    if(tmp > grid.getPscores()[i]) {
                        grid.getPscores()[i] = tmp;
                        grid.getPdxdy()[i] = std::make_pair(sx, sy);
                    }
                }
            }
            //            grid.getPscores()[i] = FeaturePyr::Map(F).cwiseProduct(
            //                        FeaturePyr::Map(w)).sum();
        }
    } // for i

    // compute local max score of each term. node
    Deformation::Param defw;
    defw << 0.01F, 0.0F, 0.01F, 0.0F; // default
    //defw.setZero();

    Deformation::Param defv;
    Matrix defCost(1, shift2);
    for(int sy = -shiftDT, s = 0; sy <= shiftDT; ++sy) {
        for(int sx = -shiftDT; sx <= shiftDT; ++sx, ++s) {
            defv << sx * sx, sx, sy * sy, sy;
            defCost(0, s) = defw.cwiseProduct(defv).sum();
        }
    }

    vector<AOGrid::Vertex> & gridNodeSet(grid.getNodeSet());
    for(int i = 0; i < gridNodeSet.size(); i++) {
        AOGrid::Vertex & node(gridNodeSet[i]);
        if(node.type_ != T_NODE) {
            continue;
        }

        Rectangle instance(grid.instanceBbox(
                               node.idx_[AOGrid::Vertex::ID_IN_INSTANCE_SET]));

        if(isPos) {
            node.pscores_.resize(numPt);
            node.pdxdy_.resize(numPt);
        } else {
            node.nscores_.resize(numPt);
            node.ndxdy_.resize(numPt);
        }

        //#pragma omp parallel for
        for(int j = 0; j < numPt; ++j) {
            int jj = j;

            const vector<Matrix> & scoreMaps(cellScoreMaps[j]);
            Scalar maxScore = -std::numeric_limits<Scalar>::infinity();
            std::pair<int, int> dxdy;
            for(int sy = -shiftDT, s = 0; sy <= shiftDT; ++sy) {
                for(int sx = -shiftDT; sx <= shiftDT; ++sx, ++s) {
                    Scalar score = scoreMaps[s].block(instance.y(),
                                                      instance.x(),
                                                      instance.height(),
                                                      instance.width()).sum() -
                            defCost(0, s);
                    if(score > maxScore) {
                        maxScore = score;
                        dxdy = std::make_pair(sx, sy);
                    }
                }
            }

            if(isPos) {
                node.pscores_[jj] = maxScore;
                node.pdxdy_[jj] = dxdy;
            } else {
                node.nscores_[jj] = maxScore;
                node.ndxdy_[jj] = dxdy;
            }
        } // for j
    } // for i
}

template <int Dimension>
void ParameterLearner::initMarginBoundPruning() {
    if(wHist_ == NULL) {
        wHist_ = new std::deque<Eigen::VectorXd>();
    }

    wHist_->clear();
    wHist_->resize(TrainSample::NbHist);

    if(normDowHist_ == NULL) {
        normDowHist_ = new Eigen::VectorXd();
    }

    normDowHist_->resize(TrainSample::NbHist);
    normDowHist_->fill(std::numeric_limits<double>::infinity());
}

template <int Dimension>
void ParameterLearner::copyPool() {
    ptPoolStateCpy_ = ptPoolState_;
    vector<ParseTree >().swap(ptPoolCpy_);
    ptPoolCpy_.clear();
    for(int i = 0; i < ptPoolState_.size(); ++i) {
        if(!ptPoolState_[i]) continue;
        ptPoolCpy_.push_back(ptPool_[i]);
    }

    TrainSampleSet().swap(posSetCpy_);
    posSetCpy_.clear();
    TrainSampleSet().swap(negSetCpy_);
    negSetCpy_.clear();;
    posSetCpy_ = posSet_;
    negSetCpy_ = negSet_;
}

template <int Dimension>
void ParameterLearner::restorePool() {

    ptPoolState_ = ptPoolStateCpy_;
    int j = 0;
    for(int i = 0; i < ptPoolState_.size(); ++i) {
        if(!ptPoolState_[i]) {
            ptPool_[i].clear();
            ptPool_[i].setGrammar(*grammar_);
            continue;
        }
        ptPool_[i] = ptPoolCpy_[j++];
    }

    TrainSampleSet().swap(posSet_);
    posSet_.clear();
    TrainSampleSet().swap(negSet_);
    negSet_.clear();

    posSet_  = posSetCpy_;
    negSet_  = negSetCpy_;
}

/// Instantiation
INSTANTIATE_CLASS_(ParameterLearner_);


// ------ Loss ------


template <int Dimension>
Loss_<Dimension>::Loss_(ParameterLearner & learner,
                        TrainSampleSet & positives,
                        TrainSampleSet & negatives,
                        double C, int maxIterations) :
    learner_(learner), positives_(positives), negatives_(negatives), C_(C),
    maxIterations_(maxIterations) {
    samples_.resize(positives_.size() + negatives_.size(), NULL);
    int i = 0;
    for(; i < positives_.size(); ++i) {
        samples_[i] = &(positives_[i]);
    }

    for(int j = 0; j < negatives_.size(); ++j, ++i) {
        samples_[i] = &(negatives_[j]);
    }
}

template <int Dimension>
double Loss_<Dimension>::operator()(const double * x, double * grad) {
    RGM_CHECK_NOTNULL(x);

    int len = dim();
    RGM_CHECK_GT(len, 0);

    // Recopy the parameters into the grammar
    learner_.getGrammar().assignParameters(x);

    // Remove oldest historical parameter vector
    std::deque<Eigen::VectorXd> & wHist(*(learner_.getWHist()));

    Eigen::Map<const Eigen::VectorXd> w(x, len);
    wHist.pop_back();
    wHist.push_front(w);

    // Compute ||dw|| between cur_w and all historical w's
    Eigen::VectorXd & dw(*(learner_.getNormDowHist()));
    dw(0) = 0;
    for(int i = 1; i < TrainSample::NbHist; ++i) {
        if(wHist[i].size() == w.size()) {
            dw(i) = (wHist[i] - w).norm();
        }
    }

    const double Inf = std::numeric_limits<double>::infinity();

    // Compute the loss and gradient over the samples
    if(grad) {
        learner_.getGrammar().initGradient();
    }

    // objective value
    const int nbEx = samples_.size();
    Eigen::VectorXd objVals(Eigen::VectorXd::Zero(nbEx));

    // record the parse trees used to update gradient
    vector<ParseTree *> updatingPt(nbEx, NULL);
    vector<ParseTree *> updatingBeliefPt(nbEx, NULL);

#pragma omp parallel for
    for(int i = 0; i < nbEx; ++i) {
        // Check margin-bound pruning condition
        samples_[i]->getNbHist()++;
        int hist = samples_[i]->nbHist();
        if(hist < TrainSample::NbHist) {
            double skip = samples_[i]->marginBound()
                    - dw(hist) * (samples_[i]->beliefNorm() +
                                  samples_[i]->maxNonbeliefNorm());
            if(skip > 0) {
                continue;
            }
        }

        TrainSample & ex(*(samples_[i]));

        int I  = 0;
        int beliefI = 0;

        double V                 = -Inf;
        double beliefScore       = 0;
        double maxNonbeliefScore = -Inf;

        for(int j = 0;  j < ex.getPts().size(); ++j) {
            ParseTree & curPt(*ex.getPts()[j]);
            PtStates & state(*(curPt.getStates()));

            state.score_        = learner_.getGrammar().dot(curPt);
            double lossAdjScore = state.score_ + state.loss_;

            // record score of belief
            if(state.isBelief_) {
                beliefScore = state.score_;
                beliefI = j;
            } else if(lossAdjScore > maxNonbeliefScore) {
                maxNonbeliefScore = lossAdjScore;
            }

            if(lossAdjScore > V) {
                I = j;
                V = lossAdjScore;
            }
        } // for j

        objVals(i) = C_ * (V - beliefScore);

        samples_[i]->getMarginBound()  = beliefScore - maxNonbeliefScore;
        samples_[i]->getNbHist()       = 0;

        if(I != beliefI) {
            updatingPt[i]       = ex.getPts()[I];
            updatingBeliefPt[i] = ex.getPts()[beliefI];
        }
    } // for i

    // update the gradient
    for(int i = 0; i < updatingPt.size(); ++i) {
        if(updatingPt[i] != NULL) {
            learner_.getGrammar().updateGradient(*updatingPt[i], C_);
            learner_.getGrammar().updateGradient(*updatingBeliefPt[i],
                                                 -1.0F * C_);
        }
    }

    cacheInfo_(OBJ_REG) = learner_.getGrammar().computeNorm(grad);

    cacheInfo_(OBJ_FG) = 0;
    cacheInfo_(OBJ_FG) = std::accumulate(objVals.data(),
                                         objVals.data() + positives_.size(),
                                         cacheInfo_(OBJ_FG));

    cacheInfo_(OBJ_BG) = 0;
    cacheInfo_(OBJ_BG) = std::accumulate(objVals.data() + positives_.size(),
                                         objVals.data() + objVals.size(),
                                         cacheInfo_(OBJ_BG));

    cacheInfo_(OBJ_TOTAL) = cacheInfo_(OBJ_REG) + cacheInfo_(OBJ_FG) +
            cacheInfo_(OBJ_BG);

    // get the gradient
    if(grad) {
        learner_.getGrammar().getParameters(grad, 2);
    }

    return cacheInfo_(OBJ_TOTAL);
}

template<int Dimension>
CacheInfo Loss_<Dimension>::computeCacheInfo() {
    const double Inf = std::numeric_limits<double>::infinity();

    const int nbEx = samples_.size();

    cacheInfo_(OBJ_REG) = learner_.getGrammar().computeNorm(false);

    // objective value
    Eigen::VectorXd objVals(Eigen::VectorXd::Zero(nbEx));

#pragma omp parallel for
    for(int i = 0; i < nbEx; ++i) {
        TrainSample & ex(*(samples_[i]));

        double V            = -Inf;
        double beliefScore  = 0;

        for(int j = 0;  j < ex.getPts().size(); ++j) {
            ParseTree & curPt(*ex.getPts()[j]);
            PtStates & state(*(curPt.getStates()));

            state.score_        = learner_.getGrammar().dot(curPt);
            double lossAdjScore = state.score_ + state.loss_;

            // record score of belief
            if(state.isBelief_) {
                beliefScore = state.score_;
            }

            if(lossAdjScore > V) {
                V = lossAdjScore;
            }
        } // for j

        objVals(i) = C_ * (V - beliefScore);

        // compute margin
        for(int j = 0;  j < ex.getPts().size(); ++j) {
            ParseTree & curPt(*ex.getPts()[j]);
            PtStates & state(*(curPt.getStates()));

            state.margin_ = beliefScore - (state.score_ + state.loss_);
        } // for j

        // reset margin bound pruning
        ex.getNbHist() = 0;
        ex.getMarginBound() = -1.0F;

    } // for i

    // compute threshold
    vector<Scalar> posScore(positives_.size(), 0);
    for(int i = 0; i < positives_.size(); ++i) {
        posScore[i] = positives_[i].pts()[0]->states()->score_;
    } // for i

    std::sort(posScore.begin(), posScore.end());

    learner_.getGrammar().getThresh() = posScore[floor(posScore.size() * 0.05F)];

    // return cache info
    cacheInfo_(OBJ_FG) = 0;
    cacheInfo_(OBJ_FG) = std::accumulate(objVals.data(),
                                         objVals.data() + positives_.size(),
                                         cacheInfo_(OBJ_FG));

    cacheInfo_(OBJ_BG) = 0;
    cacheInfo_(OBJ_BG) = std::accumulate(objVals.data() + positives_.size(),
                                         objVals.data() + objVals.size(),
                                         cacheInfo_(OBJ_BG));

    cacheInfo_(OBJ_TOTAL) = cacheInfo_(OBJ_REG) + cacheInfo_(OBJ_FG) +
            cacheInfo_(OBJ_BG);

    return cacheInfo_;
}

/// Instantiation
INSTANTIATE_CLASS_(Loss_);

} // namespace RGM
