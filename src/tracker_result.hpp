#ifndef RGM_TRACKER_RESULT_HPP_
#define RGM_TRACKER_RESULT_HPP_

#include "parse_tree.hpp"

namespace RGM {

#ifndef OneFrameResult
#define OneFrameResult OneFrameResult_<Dimension>
#endif

#ifndef TrackerResult
#define TrackerResult TrackerResult_<Dimension>
#endif

struct TrackerConst {
    static const int maxROISz = 1000;
    static const int NbOctave = 2;
    static const int NbWarpedPosInit = 20;
    static const int NbWarpedPosUpdate = 5;
    static const int NbFirstFramesKept = 15;
    static const int FPS = 15;
    static const int NbRelabel = 2;
    static const int TimeCheckAOGStructure = 10; // in frames
};

template<int Dimension>
class OneFrameResult_ {
public:
    typedef typename FeaturePyr::Cell  Cell;
    typedef typename FeaturePyr::Level  Level;

    OneFrameResult_();
    ~OneFrameResult_();

    /// Assign operator
    OneFrameResult & operator=(const OneFrameResult & res);

    void shiftPts();

    void computeROIFeatPyr(const FeatureParam &featParam);
    void computeWarpedFeatPyr(const FeatureParam &featParam);

    int prepareWarpPos(int modelWd, int modelHt, bool isInit,
                       const FeatureParam & featParam);

    int prepareShiftPos(int modelWd, int modelHt, const FeatureParam & featParam);

    void computeROI(int modelSz, Scalar searchROI);
    void computeWorkingScale(int modelSz, int maxWd, int maxHt);

    Scalar computeCandConsistency(const Rectangle *predictBox);

    void blurGrayImg();

    void release();

    vector<ParseTree>  pts_; // in the original frame coordinate
    int best_;
    bool isValid_;
    Scalar consistency_; // among pts_

    Mat img_;
    Rectangle roi_;
    Rectangle bbox_;
    vector<Rectangle> hardNegs_;

    Mat grayImg_;

    Rectangle lkbox_;

    // at which the sz of objBbox and the sz of model are roughly equal
    Scalar workingScale_;

    Scalar predictScale_;

    // 2 octaves: 2x - 1x - 0.5x workingScale, for roi
    FeaturePyr * pyr_;    

    vector<Mat> warpedPos_;
    vector<Level> warpedPosFeat_; // model size
    vector<Level> warpedPosFeatX_; // x model size

    // pyr for warped pos ( only for the 1st frame )
    Rectangle warpedBbox_;
    vector<Mat> warpedImgs_;
    vector<FeaturePyr *> warpedPyrs_;

    DEFINE_RGM_LOGGER;
};




/// Tracking results
template<int Dimension>
class TrackerResult_ {
  public:

    /// Default constructor
    TrackerResult_() : start_(0) {}

    ~TrackerResult_() { clear(); }

    /// Clear
    void clear();

    // Returns training frame idx
    std::vector<int> getFrameIdxForTrain(int count, int numFrameUsedToTrain);

    /// Gets output
    vector<OneFrameResult> & getOutput() { return output_; }

  private:
    vector<OneFrameResult>  output_;
    int start_;

    DEFINE_RGM_LOGGER;
};

}


#endif
