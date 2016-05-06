#ifndef RGM_PARAMETER_LEARNER_HPP_
#define RGM_PARAMETER_LEARNER_HPP_

#include "inference_dp.hpp"
#include "lbfgs.hpp"
#include "AOGrid.hpp"
#include "tracker_result.hpp"

namespace RGM {

// Predeclaration
template <int Dimension>
class ParameterLearner_;

template <int Dimension>
class Loss_;

enum objFuncValType {
    OBJ_FG = 0, OBJ_BG, OBJ_REG, OBJ_TOTAL
};

// cache information
typedef Eigen::Matrix < double, 1, OBJ_TOTAL + 1 > CacheInfo;

// Loss provides objective function and gradient evaluations.
template <int Dimension>
class Loss_ : public LBFGS::IFunction {

  public:
    /// Constructor
    Loss_(ParameterLearner & leaner, TrainSampleSet & positives,
          TrainSampleSet & negatives, double C, int maxIterations);

    /// Returns the length of the concatenated parameter vector
    virtual int dim() const { return learner_.getGrammar().dim();}

    virtual double operator()(const double * x, double * grad = 0);

    /// Computes the cache info
    CacheInfo computeCacheInfo();

  private:
    ParameterLearner & learner_;
    TrainSampleSet & positives_;
    TrainSampleSet & negatives_;
    vector<TrainSample * > samples_;
    double C_;
    int maxIterations_;
    CacheInfo cacheInfo_;

    DEFINE_RGM_LOGGER;
}; // class Loss



/// Estimating the parameters of AOGrammar Model using Latent SSVM
template <int Dimension>
class ParameterLearner_ {
  public:
    typedef typename FeaturePyr::Cell  Cell;
    typedef typename FeaturePyr::Level Level;
    typedef typename Appearance::Param AppParam;

    /// Constructor
    explicit ParameterLearner_(AOGrammar & g, int maxNumSamples, bool useZeroPt,
                               Scalar maxMemoryMB);

    explicit ParameterLearner_(AOGrammar & g, const string & featCacheDir,
                               int maxNumSamples, bool useZeroPt,
                               Scalar maxMemoryMB);
    /// Destructor
    virtual ~ParameterLearner_();

    /// Trains a grammar
//    Scalar train(const vector<PosData> & pos, const vector<DataIdx> & posIdx,
//                 int numFP, const vector<NegData> & neg, int numNegToUse,
//                 TrainParam trainParam, int numRelabel, int numNegMining);

    /// Estimates parameters of a grammar
    Eigen::VectorXd train(TrainSampleSet & positives, TrainSampleSet & negatives,
                          Scalar C, int maxIter);

    /// Training tracker
    Scalar train(TrackerResult &input, int count,
                 int numFrameUsedToTrain, int numRelabel,
                 float fgOverlap, float maxBgOverlap,
                 bool nms, bool dividedByUnion, bool useOverlapLoss,
                 int maxNumEx, float C,
                 int partScale, bool restart,  bool useRootOnly,
                 vector<bool> &nodeOnOffHistory, vector<bool> &edgeOnOffHistory,
                 string & saveDir);

    Scalar trainRoot(TrackerResult &input, int count,
                     int numFrameUsedToTrain, float maxBgOverlap,
                     bool dividedByUnion,
                     int maxNumEx, float C, int partScale, bool useRootOnly,
                     bool useZeroP = true);

    Scalar trainOnline(OneFrameResult &res, std::vector<ParseTree> &pts,
                       int maxNumEx, Scalar fgOverlap, Scalar maxBgOverlap,
                       bool dividedByUnion, bool useZeroPt, int partScale,
                       float C);

    /// Returns the queue of the previous parameter vector
    const std::deque<Eigen::VectorXd>  * wHist() const { return wHist_; }
    std::deque<Eigen::VectorXd> *& getWHist() { return wHist_; }

    const Eigen::VectorXd  * normDowHist() const { return normDowHist_; }
    Eigen::VectorXd *& getNormDowHist() { return normDowHist_; }

    /// Returns grammar
    const AOGrammar & grammar() const { return *grammar_; }
    AOGrammar & getGrammar() { return *grammar_; }

    void setGrammar(AOGrammar & g) {
        RGM_CHECK(!g.empty(), error);
        grammar_ = &g;
    }

    /// Returns feature cache directory
    const string & featCacheDir() const { return featCacheDir_; }
    string & getFeatCacheDir() { return featCacheDir_; }

    /// Returns bg mu
    const Cell & bgmu() const { return bgmu_; }
    Cell & getBgmu() { return bgmu_; }

    /// Returns a pt in pool
    ParseTree * getPtInPool(int * idx = NULL);

    /// Clears pos set and set the pt pool states
    void clearPosSet();

    /// Clears neg set and set the pt pool states
    void clearNegSet();

    /// Returns the number of available slots of pts in the pool
    int numPtsToFill();

    /// Returns the size of pt pool
    int szPtPool();

    /// Checks the pool
    bool checkPtPool();


    /// Creates a pt for the AOGrammar with object root only
    ParseTree * createOneRootPt(Level & feat);

    /// Creates a sample for the AOGrammar with object root only
    bool createOneRootSample(TrainSample & sample, bool isPos, bool useZeroPt,
                             Level & feat, Level * featx);

    /// Computes T-Node scores in a AOGrid
    void computeAOGridTermNodeScores(AppParam & wx, AOGrid & grid,
                                     vector<ParseTree *> & pts, bool isPos);

    /// Computes both latent pos and hard neg
    bool getTrackerTrainEx(TrackerResult & input,
                           const vector<int> &frameIdx,
                           int maxNumEx,
                           Scalar fgOverlap, Scalar maxBgOverlap,
                           bool dividedByUnion, bool useZeroPt = true);

    bool getTrackerTrainEx_ver2(TrackerResult & input,
                           const vector<int> &frameIdx,
                           int maxNumEx,
                           Scalar fgOverlap, Scalar maxBgOverlap,
                           bool dividedByUnion, bool useZeroPt = true);

    /// Computes both latent pos and hard neg
    bool getTrackerTrainEx1(TrackerResult & input,
                           const vector<int> &frameIdx,
                           int maxNumEx,
                           Scalar fgOverlap, Scalar maxBgOverlap,
                           bool dividedByUnion, bool useZeroPt = true);

    /// Init margin-bound pruning
    void initMarginBoundPruning();

    /// Copy pt pool
    void copyPool();
    void restorePool();

  private:
    AOGrammar * grammar_;
    InferenceParam inferenceParam_;

    string featCacheDir_; // precomputed feature pyr.
    Cell bgmu_;

    vector<ParseTree > ptPool_;
    vector<bool> ptPoolState_;
    bool useZeroPt_;

    TrainSampleSet   posSet_;
    vector<int> posFilterStat_;
    vector<int> posPtStat_;

    TrainSampleSet   negSet_;
    vector<int> negFilterStat_;
    vector<int> negPtStat_;

    // copy ptPool
    vector<ParseTree > ptPoolCpy_;
    vector<bool> ptPoolStateCpy_;

    TrainSampleSet   posSetCpy_;
    TrainSampleSet   negSetCpy_;

    Scalar maxMemoryInUse_;
    int maxNumExInUse_;

    // Margin-bound pruning state Historical weight vectors
    std::deque<Eigen::VectorXd> * wHist_;
    // Norms of the difference between the current weight vector and
    // each historical weight vector
    Eigen::VectorXd * normDowHist_;

    DEFINE_RGM_LOGGER;

};


}


#endif
