#ifndef RGM_TRACKER_PARAM_HPP_
#define RGM_TRACKER_PARAM_HPP_

#include "common.hpp"

namespace RGM {

/// Tracking parameters
struct TrackerParam {
    TrackerParam();

    void readConfig(const string & configFilename);

    bool useRootOnly_;
    bool notUpdateAOGStruct_;
    bool regularCheckAOGStruct_;

    int partScale_;
    Scalar fgOverlap_;
    Scalar maxBgOverlap_;
    int maxNumEx_ ;
    int maxNumExInput_;
    float maxMemoryMB_;
    Scalar C_;
    bool allowOverlapInAOGrid_;
    int betaType_;
    Scalar betaImprovement_;
    int numFramesUsedInTrain_;

    Scalar searchROI_;
    int numCand_; // kept in tracking-by-detection

    bool nms_;
    bool nmsInput_;
    bool nmsDividedByUnion_;
    bool useOverlapLoss_;

    string genericAOGFile_;

    bool showResult_;
    bool showDemo_;    
    bool showClearTmp_;
};

} // namespace RGM

#endif // RGM_TRACKER_PARAM_HPP_
