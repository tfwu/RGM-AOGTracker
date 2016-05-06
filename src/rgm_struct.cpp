#include "rgm_struct.hpp"

namespace RGM {

// ------ GrammarParam ------

void GrammarParam::init() {
    gType_ = UNKNOWN_GRAMMAR;
    regMethod_ = REG_L2;
    name_ = "UNKNOWN";
    note_ = "UNKNOWN";
    year_ = "UNKNOWN";
    isZero_ = true;
    isSingleObjModel_ = true;
    isSpecific_ = false;
    isLRFlip_ = false;
    sharedLRFlip_ = false;
}

template<class Archive>
void GrammarParam::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(gType_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(note_);
    ar & BOOST_SERIALIZATION_NVP(year_);
    ar & BOOST_SERIALIZATION_NVP(isZero_);
    ar & BOOST_SERIALIZATION_NVP(isSingleObjModel_);
    ar & BOOST_SERIALIZATION_NVP(isSpecific_);
    ar & BOOST_SERIALIZATION_NVP(isLRFlip_);
    ar & BOOST_SERIALIZATION_NVP(sharedLRFlip_);
    ar & BOOST_SERIALIZATION_NVP(regMethod_);
}

INSTANTIATE_BOOST_SERIALIZATION(GrammarParam);


// ------ AOGridParam ------

AOGridParam::AOGridParam() : gridWidth_(0), gridHeight_(0), minSize_(0),
    controlSideLength_(true), allowGridTermNode_(false),
    allowOverlap_(false), allowGap_(false), ratio_(0.5F),
    countConfig_(false), betaRule_(0), betaImprovement_(0.01F) {
}

string AOGridParam::name() const {
    stringstream ss;
    ss << "AOGrid_" << inputWindow_.width() << inputWindow_.height()
       << gridWidth_ << gridHeight_
       << minSize_ << controlSideLength_ << allowOverlap_ << allowGap_
       << ratio_ << allowGridTermNode_;

    return ss.str();
}

template<class Archive>
void AOGridParam::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(inputWindow_);
    ar & BOOST_SERIALIZATION_NVP(gridWidth_);
    ar & BOOST_SERIALIZATION_NVP(gridHeight_);
    ar & BOOST_SERIALIZATION_NVP(minSize_);
    ar & BOOST_SERIALIZATION_NVP(controlSideLength_);
    ar & BOOST_SERIALIZATION_NVP(allowGridTermNode_);
    ar & BOOST_SERIALIZATION_NVP(allowOverlap_);
    ar & BOOST_SERIALIZATION_NVP(allowGap_);
    ar & BOOST_SERIALIZATION_NVP(ratio_);
    ar & BOOST_SERIALIZATION_NVP(countConfig_);
    ar & BOOST_SERIALIZATION_NVP(cellWidth_);
    ar & BOOST_SERIALIZATION_NVP(cellHeight_);
    ar & BOOST_SERIALIZATION_NVP(cellWidthLast_);
    ar & BOOST_SERIALIZATION_NVP(cellHeightLast_);
    ar & BOOST_SERIALIZATION_NVP(betaRule_);
    ar & BOOST_SERIALIZATION_NVP(betaImprovement_);
}

INSTANTIATE_BOOST_SERIALIZATION(AOGridParam);


// ------ FeatureParam ------

FeatureParam::FeatureParam()  : type_(UNSPECIFIED_FEATURE), cellSize_(8),
    padx_(0), pady_(0), octave_(0), interval_(10), extraOctave_(false),
    partOctave_(true), scaleBase_(2.0F), featureBias_(10.0F), minLevelSz_(5),
    useTrunc_(true) {
}


bool FeatureParam::isValid() const {
    bool notValid = (padx_ < 0) || (pady_ < 0) || (octave_ < 0) ||
                    (interval_ < 1) || (!extraOctave_ && cellSize_ / 2 < 2) ||
                    (extraOctave_ && cellSize_ / 4 < 2) || (minLevelSz_ < 5) ||
                    (useTrunc_ && type_ >= HOG_SIMPLE);

    return !notValid;
}

template<class Archive>
void FeatureParam::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(cellSize_);
    ar & BOOST_SERIALIZATION_NVP(padx_);
    ar & BOOST_SERIALIZATION_NVP(pady_);
    ar & BOOST_SERIALIZATION_NVP(octave_);
    ar & BOOST_SERIALIZATION_NVP(interval_);
    ar & BOOST_SERIALIZATION_NVP(extraOctave_);
    ar & BOOST_SERIALIZATION_NVP(partOctave_);
    ar & BOOST_SERIALIZATION_NVP(featureBias_);
    ar & BOOST_SERIALIZATION_NVP(minLevelSz_);
    ar & BOOST_SERIALIZATION_NVP(useTrunc_);
}

INSTANTIATE_BOOST_SERIALIZATION(FeatureParam);


// ------ TrainParam ------

void TrainParam::init() {
    flipPositives_ = true;
    useDifficultPos_ = false;
    flipModel_ = true;
    flipSharingParam_ = true;
    C_ = 0.001;
    numFP_ = 0;

    cacheExampleLimit_ = 24000;
    numNegUsedSmall_ = 200;
    numNegUsedLarge_ = 2000;
    fgOverlap_ = 0.7F;
    maxBgOverlap_ = 0.5;

    dataSplit_ = SPLIT_ASPECT_RATIO;
    minClusters_ = 2;
    maxClusters_ = 6;
    dataMetric_ = EUCLIDEAN;
    partConfig_ = AOG_SEARCH;
    partCount_ = 8;
    partWidth_ = 6;
    partHeigth_ = 6;
}


} // namespace RGM
