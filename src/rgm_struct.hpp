#ifndef RGM_STRUCT_HPP_
#define RGM_STRUCT_HPP_

#include "rectangle.hpp"

namespace RGM {

/// Types of node and edge
/// Three types of nodes:
/// AND-node (structural decomposition),
/// OR-node (alternative decompositions),
/// TERMINAL-node (link to data appearance).
enum nodeType {
    AND_NODE = 0, OR_NODE, T_NODE, UNKNOWN_NODE
};

/// Four types of edges:
/// Switching   (OR-node to AND-node),
/// Composition (AND-node to OR-node),
/// Deformation (AND-node to TERMINAL-node / OR-node)
/// Terminate   (AND-node to TERMINAL-node)
enum edgeType {
    SWITCHING = 0, COMPOSITION, DEFORMATION, TERMINATION, UNKNOWN_EDGE
};

/// Types of Grammar
enum grammarType {
    STARMIXTURE = 0, GRAMMAR, UNKNOWN_GRAMMAR
};

/// Type of regularization
enum regType {
    REG_L2 = 0, REG_MAX
};

/// Methods used to assign latent structures
enum partConfigType {
    GREEDY_PURSUIT = 0, AOG_SEARCH
};

/// Methods for splitting training data
enum dataSplitType {
    SPLIT_ASPECT_RATIO = 0, SPLIT_AOG_APCLUSTER
};

/// Cluster metric for left-right data split
enum dataClusterMetric {
    EUCLIDEAN = 0, CHISQ, HIK
};

/// The rule of splitting a primtive type into two child primtive types
/// allowing overlaps between the two children
enum tangramSplitRule {
    UNKNOWN_SPLIT = 0, HOR_SPLIT, VER_SPLIT
};

/// Feature types
enum featureType {
    HOG_DPM = 0, HOG_FFLD, HOG_LBP, HOG_COLOR, HOG_LBP_COLOR,
    HOG_SIMPLE, HOG_SIMPLE_LBP, HOG_SIMPLE_COLOR, HOG_SIMPLE_LBP_COLOR, // for tracking, no truncation feature
    UNSPECIFIED_FEATURE
};
/// Dim of each feature type
static const int FeatureDim[] = { 32, 32, 42, 38, 48,
                                  22, 32, 28, 38, // for tracking, no truncation feature
                                  -1  };
/// Names of feature types
static const string FeatureName[] = { "HOG_DPM", "HOG_FFLD", "HOG_LBP",
                                      "HOG_COLOR", "HOG_LBP_COLOR",
                                      "HOG_SIMPLE", "HOG_SIMPLE_LBP",
                                      "HOG_SIMPLE_COLOR", "HOG_SIMPLE_LBP_COLOR",
                                      "UNSPECIFIED_FEAT" };
// Wheter a feature is left-right flippable
static const bool FeatureFlippable[] = { true, true, true, true, true,
                                         true, true, true, true,
                                         false };

/// Grammar model parameters
struct GrammarParam {
    grammarType gType_;
    string name_;
    string note_;
    string year_;

    bool isZero_; // not trained yet
    bool isSingleObjModel_;
    bool isSpecific_;

    bool isLRFlip_;
    bool sharedLRFlip_; // if parameters are shared by L, R

    regType regMethod_;

    GrammarParam() { init(); }
    void init();

    DEFINE_SERIALIZATION;
};


/// Parameters for creating the AOGrid
struct AOGridParam {
    // in the coordinate of feature map (e.g., HOG) or image lattice
    Rectangle inputWindow_;

    int    gridWidth_; // the size of the AOGrid
    int    gridHeight_;

    // for controlling the side length or area of a tiling primitive
    int    minSize_;
    bool   controlSideLength_;

    // allow the whole grid to terminate directly
    bool   allowGridTermNode_;

    /// Allow overlap or gap between the two child sub-grids
    /// when cutting a parent sub-grid
    bool   allowOverlap_;
    bool   allowGap_;
    float  ratio_;

    bool   countConfig_;

    // for the (gridHeigth_-1)*(gridWidth_-1) cells in the grid
    int    cellWidth_;
    int    cellHeight_;
    // for the last row and col
    int    cellWidthLast_;
    int    cellHeightLast_;

    int betaRule_; // 0: using classification error, 1: using variance
    Scalar betaImprovement_; // factor

    AOGridParam();
    string name() const;

    DEFINE_SERIALIZATION;
};

/// Feature extraction parameter
struct FeatureParam {
    featureType type_;
    int cellSize_;
    int padx_;
    int pady_;
    int octave_;
    int interval_;
    bool extraOctave_;
    bool partOctave_;
    Scalar scaleBase_; // standard 2.0F
    Scalar featureBias_;
    int minLevelSz_;
    bool useTrunc_;

    FeatureParam();
    bool isValid() const;
    DEFINE_SERIALIZATION;
};


/// Training setting
struct TrainParam {
    bool flipPositives_;
    bool useDifficultPos_;
    bool flipModel_; // use LR flip in model
    bool flipSharingParam_; // L R component share appearance paramters
    Scalar C_; // regularization parameter, default 0.001
    int numFP_;

    Scalar cacheMemory_; // in MB
    int cacheExampleLimit_; // 24000
    int numNegUsedSmall_;
    int numNegUsedLarge_;
    float fgOverlap_;
    float maxBgOverlap_;

    dataSplitType dataSplit_;
    int minClusters_;
    int maxClusters_;
    dataClusterMetric dataMetric_;

    partConfigType partConfig_;
    int partCount_;
    int partWidth_;
    int partHeigth_;

    TrainParam() { init(); }
    void init();
};


struct InferenceParam {
    InferenceParam() : thresh_(0.0F), useNMS_(false), useExtNMS_(false),
        nmsOverlap_(0.5F), nmsDividedByUnion_(false), createSample_(false),
        useOverlapLoss_(false), createRootSample2x_(false), pad_(0),
        computeTNodeScores_(false), clipPt_(true) {
    }

    Scalar thresh_;
    bool useNMS_; // standard NMS
    bool useExtNMS_; // extended NMS for our N-car AOG
    Scalar nmsOverlap_;
    bool nmsDividedByUnion_;
    bool createSample_; // if true get the feature in the feature pyramid
    bool useOverlapLoss_;
    bool createRootSample2x_;
    int pad_;
    bool computeTNodeScores_;
    bool clipPt_;
};

namespace detail {

/// Symmetry for flipping features
static const int HOGSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
    29, 30, 27, 28, // Texture
    31 // Truncation
};

static const int HOGLBPSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
    29, 30, 27, 28, // Texture
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // Uniform LBP
    41 // Truncation
};

static const int HOGCOLORSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
    29, 30, 27, 28, // Texture
    31, 32, 33, 34, 35, 36, // Color
    37 // Truncation
};

static const int HOGLBPColorSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
    29, 30, 27, 28, // Texture
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // Uniform LBP
    41, 42, 43, 44, 45, 46, // Color
    47 // Truncation
};

static const int HOGSimpleSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    20, 21, 18, 19 // Texture
};

static const int HOGSimpleLBPSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    20, 21, 18, 19, // Texture
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31 // Uniform LBP
};

static const int HOGSimpleColorSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    20, 21, 18, 19, // Texture
    22, 23, 24, 25, 26, 27 // Color
};

static const int HOGSimpleLBPColorSymmetry[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
    20, 21, 18, 19, // Texture
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, // Uniform LBP
    32, 33, 34, 35, 36, 37 // Color
};

} // namespace detail

} // namespace RGM


#endif // RGM_STRUCT_HPP_
