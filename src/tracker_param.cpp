#include "tracker_param.hpp"

namespace RGM {

TrackerParam::TrackerParam() :
    useRootOnly_(false), notUpdateAOGStruct_(false), regularCheckAOGStruct_(false),
    partScale_(0), fgOverlap_(0.7), maxBgOverlap_(0.1),
    maxNumEx_(4000), maxNumExInput_(4000), maxMemoryMB_(6000),
    C_(0.001F), allowOverlapInAOGrid_(true),
    betaType_(0), betaImprovement_(0.3),
    numFramesUsedInTrain_(100),
    searchROI_(3), numCand_(10), nms_(true), nmsInput_(true),
    nmsDividedByUnion_(true), useOverlapLoss_(false),
    showResult_(false), showDemo_(false) {

}

void TrackerParam::readConfig(const string & configFilename) {
    CvFileStorage * fs = cvOpenFileStorage(configFilename.c_str(), 0,
                                               CV_STORAGE_READ);
    if ( fs == NULL ) return;

    useRootOnly_ = cvReadIntByName(fs, 0, "useRootOnly", 0);
    notUpdateAOGStruct_ = cvReadIntByName(fs, 0, "notUpdateAOGStruct", 0);
    regularCheckAOGStruct_ = cvReadIntByName(fs, 0, "regularCheckAOGStruct", 0);
    if ( useRootOnly_ ) {
        notUpdateAOGStruct_ = true;
        regularCheckAOGStruct_ = false;
    }

    fgOverlap_ = cvReadRealByName(fs, 0, "fgOverlap", 0.7F);
    maxBgOverlap_ = cvReadRealByName(fs, 0, "maxBgOverlap", 0.1F);
    maxNumEx_ = cvReadIntByName(fs, 0, "cacheExampleLimit", 4000);
    maxNumExInput_ = maxNumEx_;
    maxMemoryMB_ = cvReadRealByName(fs, 0, "CacheMByteLimit", 6000);
    C_ = cvReadRealByName(fs, 0, "C", 0.001F);
    allowOverlapInAOGrid_ = cvReadIntByName(fs, 0, "allowOverlap", 1);
    betaType_ = cvReadIntByName(fs, 0, "betaType", 0);
    betaImprovement_ = cvReadRealByName(fs, 0, "betaImprovement", 0.3F);
    numFramesUsedInTrain_ = cvReadIntByName(fs, 0, "numFramesUsedInTraining", 100);

    searchROI_ = cvReadRealByName(fs, 0, "searchROI", 3.0F);
    numCand_ = cvReadIntByName(fs, 0, "numCandInDet", 10);

    nms_ = cvReadIntByName(fs, 0, "nms", 1);
    nmsInput_ = nms_;
    nmsDividedByUnion_ = cvReadIntByName(fs, 0, "nmsDividedByUnion", 1);
    useOverlapLoss_ = cvReadIntByName(fs, 0, "useOverlapLoss", 0);

    genericAOGFile_ = cvReadStringByName(fs, 0, "genericAOG", "");

    showResult_ = cvReadIntByName(fs, 0, "showResult", 0);
    showDemo_ = cvReadIntByName(fs, 0, "showDemo", 0);    
    showClearTmp_ = cvReadIntByName(fs, 0, "showClearTmp", 1);

    cvReleaseFileStorage(&fs);
}

} // namespace RGM
