#ifndef RGM_TRACKER_HPP_
#define RGM_TRACKER_HPP_

#include "tracker_param.hpp"
#include "tracker_data.hpp"
#include "tracker_result.hpp"
#include "parameter_learner.hpp"
#include "AOGrid.hpp"
#include "tracker_lk.hpp"
#include "inference_dp.hpp"

namespace RGM {

/// AOGTracker
template<int Dimension, int DimensionG>
class AOGTracker_ {
  public:
    explicit AOGTracker_(const string & configFilename);

    virtual ~AOGTracker_();

  private:
    /// Reads configurations from xml file
    bool readConfig();
    /// Runs the tracker
    void run();
#ifdef RGM_USE_MPI
    void runMPI();
#endif
    /// Initialize data and output demo
    void init();
    /// Initialize the AOG model
    bool initAOG();    
    /// Runs the AOGTracker
    void runAOGTracker();
    /// Tracking-by-Parsing
    void TLP(int count, Scalar maxNum, vector<ParseTree> &pts);
    /// Runs the generic AOG
    void runGenericAOG(int count, vector<ParseTree_<DimensionG> > &pts);

    /// Finalizes predicted bounding boxes
    void finalizeResult();
    /// Shows the results
    void showResult(const std::string & cacheDir, int frameIdx);

    /// Release memory
    void release();

    /// Collects results
    bool collectResults(bool firstTime);

    /// Gets hard negs
    void getHardNeg(OneFrameResult &curRes, vector<ParseTree> &pts, Scalar candThr);

  public:
    /// VOT
    void init(const string &cacheDir, const string &seqName,
              cv::Mat img, const Rectangle & inputBox, int numFrames);
    bool initAOG(const string & cacheDir, int curFrameIdx, int numFrames,
                 const string &seqName);
    Rectangle runAOGTracker(cv::Mat img, const string & cacheDir, int curFrameIdx,
                            int &numValid, int &numIntrackable, bool &showScoreMaps);

  private:
    string configFilename_;

    // data
    TrackerData data_;
    Rectangle inputBbox_; // at original resolution in the current frame
    Mat curFrameImg_;
    int count_;

    // online specific AOG
    FeatureParam featParam_;
    int modelArea_;
    AOGrammar  sAOG_; // specific AOG learned online
    Scalar sTau_;
    Rectangle displacement_; // added to input bounding box
    Scalar trainLoss_;
    vector<bool> nodeOnOffHistory_;
    vector<bool> edgeOnOffHistory_;

    int maxWd_; // for fft
    int maxHt_;

    bool useFixedRandSeed_;
    bool runBaseLine_;
    int runMode_;

    TrackerParam param_;

    ParameterLearner *learner_;

    DPInference sDet_;
    InferenceParam sDetParam_;
    Scalar thrTolerance_;
    bool useFixedDetThr_;

    // off line trained generic AOG model
    AOGrammar_<DimensionG> gAOG_;
    int countExecution_;

    DPInference_<DimensionG> gDet_;
    InferenceParam gDetParam_;

    // short-term tracker
    LKTracker lk_;

    // color features
    Matrix w2crs_;

    // results
    TrackerResult results_;
    int preValidIdx_;

    vector<Scalar> trackabilities_;

    std::map<int, Scalar> ptQuality_;
    vector<int> turnedOffTnodes_;
    Scalar turnedOffThr_;


    Mat showResult_;
    boost::format ptNameFrmt_;
    cv::VideoWriter resultVideo_;
    Mat resultFrame_;

    cv::Mat  frameShow_;
    cv::Rect frameROI_;

    cv::Mat objShow_;
    cv::Rect objROI_;

    cv::Rect modelROI_;

    string modelName_;
    string showWindName_;

    vector<int> allFrameIdx_;

    DEFINE_RGM_LOGGER;
};

#define AOGTracker AOGTracker_<Dimension, DimensionG>

/// Run AOGTracker
void RunAOGTracker(string & configFilename);

}

#endif
