#include "tracker.hpp"
#include "util/UtilString.hpp"
#include "util/UtilMath.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilFile.hpp"
#include "timer.hpp"

namespace RGM {

#define RGM_TRACKER_SHOW_DETAIL

template<int Dimension, int DimensionG>
AOGTracker::AOGTracker_(const string &configFilename) :
    configFilename_(configFilename), learner_(NULL), count_(0), useFixedDetThr_(false),
    thrTolerance_(0.5F) {
    RGM_CHECK(readConfig(), error);

    if(runMode_ >= 0) {
#ifdef RGM_USE_MPI
        runMPI();
#else
        run();
#endif
    }
}

template<int Dimension, int DimensionG>
AOGTracker::~AOGTracker_() {
    if(learner_ != NULL) {
        delete learner_;
        learner_ = NULL;
    }
}

template<int Dimension, int DimensionG>
bool AOGTracker::readConfig() {       
    CvFileStorage * fs = cvOpenFileStorage(configFilename_.c_str(), 0,
                                           CV_STORAGE_READ);
    if ( fs == NULL ) return false;

    // parameters
    featParam_.type_ = static_cast<featureType>(cvReadIntByName(fs, 0, "FeatType", 4));
    featParam_.cellSize_ = cvReadIntByName(fs, 0, "CellSize", 4);
    if ( featParam_.cellSize_ != 8 && featParam_.cellSize_ != 4) {
        featParam_.cellSize_  = 4;
    }
    featParam_.extraOctave_ = false;
    featParam_.interval_ = cvReadIntByName(fs, 0, "Interval", 6);
    featParam_.featureBias_ = cvReadRealByName(fs, 0, "Bias", 10.0F);
    featParam_.useTrunc_ = false;

    //    featParam_.octave_ = TrackerConst::NbOctave;

    useFixedRandSeed_ = cvReadIntByName(fs, 0, "useFixedRandSeed", 1);
    runBaseLine_ = cvReadIntByName(fs, 0, "runBaseLineTracker", 0);
    runMode_ = cvReadIntByName(fs, 0, "runMode", -1); // -1: none, 0: regular, 1: TB-100

    cvReleaseFileStorage(&fs);

    param_.readConfig(configFilename_);

    if ( runMode_ < 0 ) {
        param_.showResult_ = false;
        param_.showDemo_ = false;
    }

    if (!param_.showDemo_ && !param_.showResult_ )
        param_.showClearTmp_ = true;

    // source
    if(runMode_ >= 0)
        return data_.readConfig(configFilename_);

    return true;
}

template<int Dimension, int DimensionG>
void AOGTracker::run() {

    RGM_LOG(normal, "=============== Run benchmark evaluation ===============");

    if(!collectResults(true))  return;

    Timers timer;

    if(runMode_ > 0)
        data_.getAllSource();

    // debug
    //    //    string saveDir = data_.rootDir() + "visDataset";
    //    string saveDir = data_.rootDir() + "visInput";
    //    FileUtil::VerifyDirectoryExists(saveDir);
    //    //                data_.visDataset(saveDir);
    //    data_.visualizeInput(saveDir);
    //    return;

    for(int i = 0; i < data_.numSequences(); ++i) {
        //        RGM_LOG(normal, "=============== Processing " + NumToString_<int>(i + 1) + " out of " +
        //                NumToString_<int>(data_.numSequences()) + " videos ==========");
        std::cerr << "=============== Processing " + NumToString_<int>(i + 1) + " out of " +
                     NumToString_<int>(data_.numSequences()) + " videos =========="
                  << std::endl ; fflush(stdout);


        double timeConsumed = 0;
        double speed = 0;
        if(!data_.setCurSeqIdx(i, true)) {
            collectResults(false);
            continue;
        }

        Timers::Task * curTimer = timer(data_.resultFileName());
        curTimer->Start();

        init();
        if(!initAOG()) {
            release();
            curTimer->Stop();
            RGM_LOG(warning, "***************** Failed to track " + data_.resultFileName());
            continue;
        }

        runAOGTracker();

        curTimer->Stop();
        timeConsumed = curTimer->ElapsedSeconds();
        speed = data_.numFrames() / timeConsumed;

        string speedFilename = data_.cacheDir() + data_.resultFileName() + "_speed.txt";
        std::ofstream ofs(speedFilename.c_str(), std::ios::out);
        if(ofs.is_open()) {
            ofs << curTimer->ElapsedSeconds() << ", " << speed << std::endl;
            ofs.close();
        }

        collectResults(false);

        //        RGM_LOG(normal, "=============== Time usage: "
        //                + NumToString_<double>(curTimer->ElapsedSeconds()) +
        //                ", speed (FPS) " + NumToString_<double>(speed));

        std::cerr << "=============== Time usage: " +
                     NumToString_<double>(curTimer->ElapsedSeconds()) +
                     ", speed (FPS) " + NumToString_<double>(speed)
                  << std::endl ; fflush(stdout);
    }

    timer.clear();
}

template<int Dimension, int DimensionG>
void AOGTracker::init() {    
    init(data_.cacheDir(), data_.resultFileName(),
         data_.getStartFrameImg(), data_.getInputBox(), data_.numFrames());
}

template<int Dimension, int DimensionG>
void AOGTracker::init(const string &cacheDir, const string &seqName,
                      cv::Mat img, const Rectangle & inputBbox, int numFrames) {

    // set the seed for rand for reproducibility if specified
    const int seed = 3;
    srand((useFixedRandSeed_ ? seed : time(NULL)));

    curFrameImg_ = img;
    inputBbox_ = inputBbox;
    inputBbox_.clip(img.cols, img.rows);

    // debug
    //    cv::rectangle(curFrameImg_, inputBbox_.cvRect(), cv::Scalar(0, 0, 255), 2);
    //    cv::imshow("debug", curFrameImg_);
    //    cv::waitKey(0);

    // Usually the annotated bounding box is too tight
    //    int maxWd = curFrameImg_.cols;
    //    int maxHt = curFrameImg_.rows;

    // select cell size
    //    int minsz = std::min<int>(inputBbox_.width(), inputBbox_.height());
    //    float factor = float(minsz) / featParam_.cellSize_;
    //    if(factor > 10) {
    //        featParam_.cellSize_ = 8;
    //    } else if(factor < 3) {
    //        featParam_.cellSize_ = 4;
    //    }

    //    if(inputBbox_.width() / featParam_.cellSize_ < 3 ||
    //            inputBbox_.height() / featParam_.cellSize_ < 3) {
    //        displacement_ = inputBbox_.expand(std::min<int>(featParam_.cellSize_ * 2, 8),
    //                                          &maxWd, &maxHt);
    //    } else {
    displacement_ = Rectangle();
    //    }

    results_.clear();
    results_.getOutput().resize(numFrames);

    count_ = 0;

    allFrameIdx_.clear();
    allFrameIdx_.push_back(count_);

    if(param_.showDemo_) {
        showWindName_ = "AOGTracker";

        //        cv::namedWindow(showWindName_, cv::WINDOW_AUTOSIZE);
        //        cv::moveWindow(showWindName_, 10, 10);

        // write output video
        resultFrame_.create(1180, 1920, curFrameImg_.type());
        string saveName = cacheDir + seqName + ".avi";
        resultVideo_.release();
        resultVideo_.open(saveName, CV_FOURCC('D', 'I', 'V', 'X'),
                          TrackerConst::FPS, resultFrame_.size(), true); //cv::VideoWriter::fourcc('D', 'I', 'V', 'X')
        if(!resultVideo_.isOpened()) {
            RGM_LOG(warning, "can not write result video");
            param_.showDemo_ = false;
        }

        float factor = std::min<float>(1.0F, std::min<float>(500.0F / curFrameImg_.cols,
                                                             1000.0F / curFrameImg_.rows));
        frameROI_ = cv::Rect(0, 0, curFrameImg_.cols * factor,
                             curFrameImg_.rows * factor);

        frameShow_.create(frameROI_.height, frameROI_.width, curFrameImg_.type());

        objROI_ = cv::Rect(0, frameROI_.height + 5,
                           std::min<int>(frameROI_.width, 200),
                           std::min<int>(resultFrame_.rows - frameROI_.height - 5, 200));
        objShow_.create(objROI_.height, objROI_.width, curFrameImg_.type());
    }
}

template<int Dimension, int DimensionG>
bool AOGTracker::initAOG() {
    initAOG(data_.cacheDir(), data_.curFrameIdx(), data_.numFrames(),
            data_.resultFileName());
}

template<int Dimension, int DimensionG>
bool AOGTracker::initAOG(const string &cacheDir, int curFrameIdx, int numFrames,
                         const string &seqName) {
    // check size
    int minArea = featParam_.cellSize_ == 8 ? 3000 : 1000;
    int maxArea = featParam_.cellSize_ == 8 ? 5000 : 4000;
    modelArea_ = std::max<int>(minArea,
                               std::min<int>(maxArea, inputBbox_.area()));

    const int maxROISz = TrackerConst::maxROISz;

    OneFrameResult & res(results_.getOutput()[count_]);
    res.img_ = curFrameImg_.clone();
    if ( !runBaseLine_ ) {
        cv::cvtColor(res.img_, res.grayImg_, cv::COLOR_BGR2GRAY);
        //        res.blurGrayImg();
    }
    res.bbox_ = inputBbox_;
    int x = std::max<int>(0, inputBbox_.xcenter() - maxROISz / 2);
    int y = std::max<int>(0, inputBbox_.ycenter() - maxROISz / 2);
    res.roi_  = Rectangle( x, y,
                           std::min<int>(maxROISz, curFrameImg_.cols - x),
                           std::min<int>(maxROISz, curFrameImg_.rows - y) );

    res.workingScale_ = sqrt(static_cast<Scalar>(modelArea_) / inputBbox_.area());
    res.predictScale_ = res.workingScale_;

    ptNameFrmt_ = boost::format("pt_%d");

    string tmpDir = cacheDir + "tmp" + FILESEP;
    FileUtil::VerifyDirectoryExists(tmpDir);

    string initialAOGFilename = tmpDir + "initialAOG.bin";
    string emptyAOGFilename = tmpDir + "emptyAOG.bin";

    int wd = ROUND(inputBbox_.width() * res.workingScale_ / featParam_.cellSize_);
    int ht = ROUND(inputBbox_.height() * res.workingScale_ / featParam_.cellSize_);
    RGM_CHECK_GT(wd, 0);
    RGM_CHECK_GT(ht, 0);

    // create AOGrid
    AOGridParam gridParam;

    param_.maxNumEx_ = std::max<int>(2000,
                                     std::min<int>(param_.maxNumExInput_,
                                                   numFrames * 100));

    // inference of specific AOG
    sDetParam_.nmsOverlap_ = param_.fgOverlap_;
    sDetParam_.nmsDividedByUnion_ = param_.nmsDividedByUnion_;
    sDetParam_.createSample_ = false;
    sDetParam_.createRootSample2x_ = false;
    sDetParam_.computeTNodeScores_ = false;
    sDetParam_.clipPt_ = false;
    sDetParam_.useOverlapLoss_ = param_.useOverlapLoss_;

    Scalar initFgOverlap = 0.5F;

    // select partScale: 0 or 1
    int i = 0;
    bool suc = false;

    string saveAOGDir = ( (param_.showResult_ || param_.showDemo_) ? tmpDir : "");

    for(; i < 2; ++i) {
        param_.nms_ = param_.nmsInput_;
        sDetParam_.useNMS_ = param_.nms_;

        param_.partScale_ = i;
        int ds = std::pow(2, param_.partScale_);
        gridParam.inputWindow_ = Rectangle(0, 0, wd * ds, ht * ds);
        int gridWd = ROUND(gridParam.inputWindow_.width() / 6.0F);
        int gridHt = ROUND(gridParam.inputWindow_.height() / 6.0F);
        gridParam.gridWidth_ = std::max<int>(3, std::min<int>(4, gridWd));
        gridParam.gridHeight_ = std::max<int>(3, std::min<int>(4, gridHt));        
        gridParam.minSize_ = 1;
        gridParam.controlSideLength_ = true;
        gridParam.allowOverlap_ = (inputBbox_.area() < maxArea); // param_.allowOverlapInAOGrid_; //
        gridParam.allowGap_ = false;
        gridParam.ratio_ = 0.5F;
        gridParam.countConfig_ = false;
        gridParam.allowGridTermNode_ = false;

        //0: error rate, 1: use variance of T-nodes
        gridParam.betaRule_ = param_.betaType_;
        gridParam.betaImprovement_ = param_.betaImprovement_;

        featParam_.padx_ = 1; //wd;
        featParam_.pady_ = 1; //ht;
        featParam_.minLevelSz_ = std::max<int>(wd, ht);
        featParam_.partOctave_ = param_.partScale_;

        // create AOGrammar
        sAOG_.clear();
        vector<pair<int, int> > rootSz(1, std::make_pair(wd, ht));
        sAOG_.getIsLRFlip() = false;
        sAOG_.getFeatParam() = featParam_;
        sAOG_.getIsSpecific() = true;
        sAOG_.initRoots(rootSz);

        sAOG_.getGridAOG().resize(1, new AOGrid(gridParam));
        sAOG_.getGridAOG()[0]->turnOnOff(true);

        sAOG_.getIsSingleObjModel() = true;
        sAOG_.getIsZero() = true;
        sAOG_.getIsSpecific() = true;

        sAOG_.createRGM(param_.partScale_, false);

        sAOG_.getName() = seqName;
        sAOG_.getYear() = "AOGTracker";
        sAOG_.getGParam().gType_ = GRAMMAR; // STARMIXTURE; //
        sAOG_.getGParam().regMethod_ =  REG_L2; // REG_MAX; //

        sAOG_.setOnOff(true);

        // save the empty AOG
        sAOG_.save(emptyAOGFilename);

        // init patch work
        maxWd_ = std::min<int>(maxROISz, curFrameImg_.cols) *
                res.workingScale_ * featParam_.scaleBase_;
        maxHt_ = std::min<int>(maxROISz, curFrameImg_.rows) *
                res.workingScale_ * featParam_.scaleBase_;

        int maxHt = (maxHt_ + sAOG_.minCellSize() - 1) / sAOG_.minCellSize() +
                sAOG_.pady() * 2;
        int maxWd = (maxWd_ + sAOG_.minCellSize() - 1) / sAOG_.minCellSize() +
                sAOG_.padx() * 2;

        if(!Patchwork::InitFFTW((maxHt + 15) & ~15, (maxWd + 15) & ~15)) {
            RGM_LOG(error, "**************** Could not initialize the Patchwork class.");
            return false;
        }

        // train AOG

        // init learner
        if(learner_ != NULL) {
            delete learner_;
            learner_ = NULL;
        }
        learner_ = new ParameterLearner(sAOG_, param_.maxNumEx_, true,
                                        param_.maxMemoryMB_);

        nodeOnOffHistory_.clear();
        edgeOnOffHistory_.clear();

        trainLoss_ = learner_->train(results_, count_,
                                     param_.numFramesUsedInTrain_,
                                     TrackerConst::NbRelabel, param_.fgOverlap_,
                                     param_.maxBgOverlap_,
                                     param_.nms_, param_.nmsDividedByUnion_,
                                     param_.useOverlapLoss_,
                                     param_.maxNumEx_, param_.C_,
                                     param_.partScale_, true, param_.useRootOnly_,
                                     nodeOnOffHistory_, edgeOnOffHistory_,
                                     saveAOGDir);

        sDet_.setGrammar(sAOG_, sDetParam_);

        // check the model quality
        res.pts_.clear();
        sDetParam_.thresh_ = (useFixedDetThr_ ? -1.002F : std::min<Scalar>(-1.002F, sAOG_.thresh() - thrTolerance_));
        vector<ParseTree> pts;
        TLP(count_, param_.maxNumEx_, pts);
        if(pts.size() == 0) {
            suc = false;
            continue;
        }

        for(int i = 0; i < pts.size() && i < param_.numCand_; ++i) {
            res.pts_.push_back(pts[i]);
            sDet_.computeIntrackability(res.pts_[i]);
        }

        res.shiftPts();

        res.best_ = 0;
        const ParseInfo *info = res.pts_[0].rootParseInfo();
        Rectangle predictBbox(ROUND(info->x()), ROUND(info->y()),
                              ROUND(info->width()), ROUND(info->height()));

        Intersector_<int> inter(inputBbox_, initFgOverlap, true);

        if(!inter(predictBbox)) {  //|| (info->goodness_ < 3)
            param_.nms_ = !param_.nms_;
            sDetParam_.useNMS_ = param_.nms_;

            nodeOnOffHistory_.clear();
            edgeOnOffHistory_.clear();

            trainLoss_ = learner_->train(results_, count_,
                                         param_.numFramesUsedInTrain_,
                                         TrackerConst::NbRelabel, param_.fgOverlap_,
                                         param_.maxBgOverlap_,
                                         param_.nms_, param_.nmsDividedByUnion_,
                                         param_.useOverlapLoss_,
                                         param_.maxNumEx_, param_.C_,
                                         param_.partScale_, true, param_.useRootOnly_,
                                         nodeOnOffHistory_, edgeOnOffHistory_,
                                         saveAOGDir);

            // check the model quality
            res.pts_.clear();
            sDetParam_.thresh_ = (useFixedDetThr_ ? -1.002F : std::min<Scalar>(-1.002F, sAOG_.thresh() - thrTolerance_));
            pts.clear();
            TLP(count_, param_.maxNumEx_, pts);
            if(pts.size() == 0) {
                suc = false;
                continue;
            }

            for(int i = 0; i < pts.size() && i < param_.numCand_; ++i) {
                res.pts_.push_back(pts[i]);
                sDet_.computeIntrackability(res.pts_[i]);
            }

            res.shiftPts();

            res.best_ = 0;
            const ParseInfo *info = res.pts_[0].rootParseInfo();
            Rectangle predictBbox(ROUND(info->x()), ROUND(info->y()),
                                  ROUND(info->width()), ROUND(info->height()));

            Intersector_<int> inter(inputBbox_, initFgOverlap, true);

            if(!inter(predictBbox)) {
                suc = false;
            } else {
                // get hard negs
                Scalar candThr = sAOG_.thresh() - thrTolerance_;
                getHardNeg(res, pts, candThr);

                suc = true;
                break;
            }

        } else {
            // get hard negs
            Scalar candThr = sAOG_.thresh() - thrTolerance_;
            getHardNeg(res, pts, candThr);
            suc = true;
            break;
        }
    }

    if(!suc) {
        RGM_LOG(error, "**************** initial learning of the AOG failed.");
        return false;
    }

    param_.maxNumEx_ = learner_->szPtPool();

    res.isValid_ = true;

    sTau_ = sAOG_.thresh();

    trackabilities_.clear();
    trackabilities_.reserve(numFrames);
    //trackabilities_.push_back(res.pts_[0].rootParseInfo()->goodness_);

    sAOG_.save(initialAOGFilename);

    if(param_.showResult_ || param_.showDemo_) {
        modelName_ = "initialAOG_Img";
        sAOG_.visualize(tmpDir, curFrameImg_(inputBbox_.cvRect()),
                        modelName_);

        modelName_ = "initialAOG";
        sAOG_.visualize(tmpDir, modelName_);
    }

    if(param_.showDemo_) {
        resultFrame_ = cv::Scalar::all(255);

        Mat origShow = curFrameImg_.clone();
        cv::rectangle(origShow, inputBbox_.cvRect(), cv::Scalar(0, 0, 255), 3);
        cv::resize(origShow, frameShow_, cv::Size(frameShow_.cols, frameShow_.rows));
        frameShow_.copyTo(resultFrame_(frameROI_));

        string initialAOGImg = tmpDir + modelName_ + ".png";
        showResult_ = cv::imread(initialAOGImg, cv::IMREAD_COLOR);
        float factor = std::min<float>(1.0F, std::min<float>(float(resultFrame_.cols - frameShow_.cols) / showResult_.cols,
                                                             (float)resultFrame_.rows / showResult_.rows));
        modelROI_ = cv::Rect(frameShow_.cols, 0, showResult_.cols * factor,
                             showResult_.rows * factor);

        modelROI_ = cv::Rect(frameShow_.cols, 0, showResult_.cols * factor,
                             showResult_.rows * factor);

        Mat tmpModel;
        cv::resize(showResult_, tmpModel, cv::Size(modelROI_.width, modelROI_.height));
        tmpModel.copyTo(resultFrame_(modelROI_));

        //        cv::imshow(showWindName_, resultFrame_);
        //        cv::waitKey(2);

        if(resultVideo_.isOpened()) {
            for(int i = 0; i < TrackerConst::FPS; ++i)
                resultVideo_ << resultFrame_;
        }
    }

    preValidIdx_ = count_;

    // show
    showResult(cacheDir, curFrameIdx);

    // show score maps
    if ( param_.showResult_ || param_.showDemo_ ) {
        string scoreMapsDir = tmpDir + "scoreMaps_00001" + FILESEP;
        FileUtil::VerifyDirectoryExists(scoreMapsDir);
        sDet_.visScoreMaps(scoreMapsDir, &res.pts_[res.best_]);
    }

    // init generic AOG
    if(FileUtil::exists(param_.genericAOGFile_)) {
        RGM_CHECK(gAOG_.read(param_.genericAOGFile_), error);

        gDetParam_.useNMS_ = true;
        gDetParam_.nmsOverlap_ = 0.5F;
        gDetParam_.nmsDividedByUnion_ = false;
        gDetParam_.clipPt_ = true;
        gDetParam_.thresh_ = -0.6; //std::min<Scalar>(gAOG_.thresh(), -0.6);

        gDet_.setGrammar(gAOG_, gDetParam_);
    }

    return true;
}

template<int Dimension, int DimensionG>
void AOGTracker::runAOGTracker() {

    int numFrames = data_.numFrames();
    int numValid = 0, numIntrackable = 0;
    bool showScoreMaps = false;
    while(count_ < numFrames) {
        Mat img = data_.getNextFrameImg();
        if ( img.empty() ) break;
        runAOGTracker(img, data_.cacheDir(), data_.curFrameIdx(),
                      numValid, numIntrackable, showScoreMaps);
    }

    finalizeResult();

    if ( param_.showClearTmp_ ) {
        string tmpDir = data_.cacheDir() + "tmp" + FILESEP;
        boost::filesystem::remove_all(boost::filesystem::path(tmpDir));
    }
}

template<int Dimension, int DimensionG>
Rectangle AOGTracker::runAOGTracker(cv::Mat img, const string & cacheDir, int curFrameIdx,
                                    int &numValid, int &numIntrackable,
                                    bool &showScoreMaps) {

    const int minObjSz = featParam_.cellSize_ * 2;

    string tmpDir = cacheDir + "tmp" + FILESEP;
    string initialAOGFilename = tmpDir + "initialAOG.bin";
    string emptyAOGFilename = tmpDir + "emptyAOG.bin";
    string tmpAOGFilename = tmpDir + "tmpAOG.bin";
    string saveAOGDir = ( (param_.showResult_ || param_.showDemo_) ? tmpDir : "");

    Scalar maxBgOverlap = param_.maxBgOverlap_;
    int modelSz = std::max<int>(sAOG_.maxDetectWindow().width(),
                                sAOG_.maxDetectWindow().height()) *
            sAOG_.cellSize();

    const int maxROISz = TrackerConst::maxROISz;

    Scalar trackabilityMean, trackabilityStd;

    Scalar trackability = 0, thrTrackabilityHigh = 5.0F, thrTrackabilityLow = 2.5;
    Scalar score;

    count_++;

    if(count_ > TrackerConst::TimeCheckAOGStructure) {
        MathUtil_<Scalar>::calcMeanStd(trackabilities_, trackabilityMean, trackabilityStd);
        thrTrackabilityHigh = std::min<Scalar>(5.0F, trackabilityMean + trackabilityStd);
        thrTrackabilityLow = std::max<Scalar>(2.5F, trackabilityMean - 3 * trackabilityStd);
    }
    Scalar sTauMean = sTau_ / count_;

    curFrameImg_ = img;
    allFrameIdx_.push_back(count_);

    bool fromGlobalSearch = ((count_ - preValidIdx_) > TrackerConst::TimeCheckAOGStructure); // false;
    bool inPredictableRange = ((count_ - preValidIdx_) < 3);

    // init the result
    OneFrameResult &preRes(results_.getOutput()[preValidIdx_]); // [count_-1]);
    OneFrameResult &curRes(results_.getOutput()[count_]);
    curRes.best_ = -1;
    curRes.pts_.clear();
    curRes.hardNegs_.clear();
    curRes.isValid_ = false;
    curRes.img_ = curFrameImg_.clone();
    curRes.bbox_ = preRes.bbox_;
    curRes.workingScale_ = preRes.predictScale_;
    curRes.predictScale_ = preRes.predictScale_;
    curRes.computeROI(modelSz, param_.searchROI_);
    curRes.computeWorkingScale(modelSz, maxWd_, maxHt_);

    // run detection
    Scalar candThr = std::min<Scalar>(0.0F, sAOG_.thresh() - thrTolerance_);
    sDetParam_.thresh_ = (useFixedDetThr_ ? -1.002F : std::min<Scalar>(candThr, -1.002F));
    Scalar maxNum  = std::max<Scalar>(learner_->numPtsToFill(), param_.maxNumEx_ / 2);

    std::vector<ParseTree> pts;
    bool detState = false;

    // search in the ROI
    if(!fromGlobalSearch) {
        TLP(count_, maxNum, pts);
        detState = (pts.size() > 0 /* && (pts[0].score() >= candThr)*/);
    }

    // search in the whole image
    if(!detState) {
        fromGlobalSearch = true;
        // search in the whole frame
        curRes.workingScale_ = results_.getOutput()[0].workingScale_;
        curRes.predictScale_ = curRes.workingScale_;
        int x = std::max<int>(0, curRes.bbox_.xcenter() - maxROISz / 2);
        int y = std::max<int>(0, curRes.bbox_.ycenter() - maxROISz / 2);
        curRes.roi_ = Rectangle( x, y,
                                 std::min<int>(maxROISz, curFrameImg_.cols - x),
                                 std::min<int>(maxROISz, curFrameImg_.rows - y) );
        if(curRes.pyr_ != NULL) {
            delete curRes.pyr_;
            curRes.pyr_ = NULL;
        }

        TLP(count_, maxNum, pts);
        detState = (pts.size() > 0  /*&& (pts[0].score() >= candThr)*/);
    }

    // run LK
    bool lkState = false;
    if ( !runBaseLine_ ) {
        if(preRes.grayImg_.empty() ||
                preRes.grayImg_.cols != curFrameImg_.cols ||
                preRes.grayImg_.rows != curFrameImg_.rows) {
            cv::cvtColor(preRes.img_, preRes.grayImg_, cv::COLOR_BGR2GRAY);
            //            preRes.blurGrayImg();
        }
        if(curRes.grayImg_.empty()) {
            cv::cvtColor(curFrameImg_, curRes.grayImg_, cv::COLOR_BGR2GRAY);
            //            curRes.blurGrayImg();
        }
        lkState = lk_.runLK1(preRes.grayImg_, preRes.bbox_, curRes.grayImg_);
    }

    // get top numCand
    vector<int> idxInPts;
    for(int i = 0; i < pts.size() &&
        curRes.pts_.size() < param_.numCand_; ++i) {
        if(pts[i].rootParseInfo()->width() <= minObjSz ||
                pts[i].rootParseInfo()->height() <= minObjSz) continue;
        sDet_.computeIntrackability(pts[i]);        
        curRes.pts_.push_back(pts[i]);
        idxInPts.push_back(i);
    }

    // check the scale changes valid range: [1/4, 4]
    vector<int> candIdx;
    if(detState) {
        if(inPredictableRange || count_ < 10) {
            Scalar refArea = preRes.bbox_.area();
            for(int i = 0; i < curRes.pts_.size(); ++i) {
                const ParseInfo * info = curRes.pts_[i].rootParseInfo();
                Scalar scaleChange = info->area() / refArea;
                if(count_ < 10) {
                    if(scaleChange < 0.49F || scaleChange > 2.89F) continue;
                } else {
                    if(scaleChange < 0.25F || scaleChange > 4.0F) continue;
                }
                candIdx.push_back(i);
            }
        } else if(fromGlobalSearch) {
            Scalar refArea = preRes.bbox_.area();
            for(int i = 0; i < curRes.pts_.size(); ++i) {
                const ParseInfo * info = curRes.pts_[i].rootParseInfo();
                Scalar scaleChange = info->area() / refArea;
                if(scaleChange < 0.0625F || scaleChange > 16.0F) continue;
                candIdx.push_back(i);
            }
        } else {
            for(int i = 0; i < curRes.pts_.size(); ++i) {
                candIdx.push_back(i);
            }
        }
    }

    if(candIdx.size() == 0)
        detState = false;

    curRes.shiftPts();
    curRes.computeCandConsistency(NULL);

    trackability = 0;
    score = -100.0F;

    int bestPtIdx = -1;

    if (runBaseLine_) {
        if ( detState ) {
            curRes.best_ = candIdx[0];
            curRes.isValid_ = true;

            const ParseInfo * info = curRes.pts_[curRes.best_].rootParseInfo();
            curRes.bbox_ = Rectangle(ROUND(info->x()), ROUND(info->y()),
                                     ROUND(info->width()), ROUND(info->height()));

            curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
            curRes.predictScale_ /= 2.0F;

            bestPtIdx = idxInPts[curRes.best_];
        }
    } else {
        // if TLP failed
        if(!detState) {
            if(lkState) {
                curRes.lkbox_ = lk_.predictBbox();
                curRes.bbox_ = lk_.predictBbox();
                curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
                curRes.predictScale_ /= 2.0F;

                curRes.isValid_ = inPredictableRange;
                if(curRes.isValid_)
                    preValidIdx_ = count_;
            }
        } else {
            int best = candIdx[0];
            trackability = curRes.pts_[best].rootParseInfo()->goodness_;
            score = curRes.pts_[best].score();
            Scalar scoreTolerance = std::abs(score) * 0.01F;

            if(pts.size() == 1 /*&& score > sAOG_.thresh()*/) {
                if(lkState) {
                    curRes.lkbox_ = lk_.predictBbox();
                }
                curRes.best_ = best;
                curRes.isValid_ = true;

                const ParseInfo * info = curRes.pts_[curRes.best_].rootParseInfo();
                curRes.bbox_ = Rectangle(ROUND(info->x()), ROUND(info->y()),
                                         ROUND(info->width()), ROUND(info->height()));

                curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
                curRes.predictScale_ /= 2.0F;

                bestPtIdx = idxInPts[curRes.best_];
            } else {
                // integration
                if(lkState) {
                    Rectangle lkPrediction(lk_.predictBbox());
                    curRes.lkbox_ = lkPrediction;

                    // compute overlap with lk
                    bool isStrict = (((count_ < 10) || inPredictableRange));
                    Intersector_<int> inter(lkPrediction,
                                            (isStrict ? 0.5F : 0.3F),
                                            true);
                    vector<int> selCandIdx;
                    for(int c = 0; c < candIdx.size(); ++c) {
                        int d = candIdx[c];
                        const ParseInfo * info = curRes.pts_[d].rootParseInfo();
                        Rectangle box(ROUND(info->x()), ROUND(info->y()),
                                      ROUND(info->width()), ROUND(info->height()));

                        if(inter(box)) {
                            selCandIdx.push_back(d);
                        }
                    }

                    // if no overlap
                    if(selCandIdx.size() == 0) {
                        best = -1;
                        bool trustLK = false;
                        if((trackability >= thrTrackabilityHigh) &&
                                (score >= sAOG_.thresh()) &&
                                (!inPredictableRange || curRes.consistency_ >= 0.4F)) {  //
                            // trust detection
                            best = candIdx[0];
                        } else if((trackability < thrTrackabilityLow) || inPredictableRange) {
                            // trust lk
                            trustLK = true;
                            best = -1;
                            float ov;
                            for(int d = 0; d < pts.size(); ++d) {
                                if(pts[d].score() < candThr) break;
                                const ParseInfo * info = pts[d].rootParseInfo();
                                Rectangle box(ROUND(info->x() + curRes.roi_.x()),
                                              ROUND(info->y() + curRes.roi_.y()),
                                              ROUND(info->width()), ROUND(info->height()));

                                if(inter(box, &ov)) {
                                    best = d;
                                    break;
                                }
                            }
                        }

                        if(best != -1) {
                            if(trustLK) {
                                curRes.pts_.push_back(curRes.pts_[candIdx[0]]);
                                curRes.pts_[candIdx[0]] = pts[best];
                                sDet_.computeIntrackability(curRes.pts_[candIdx[0]]);
                                // shift
                                for(int j = 0; j < curRes.pts_[candIdx[0]].parseInfoSet().size(); ++j) {
                                    ParseInfo * info = curRes.pts_[candIdx[0]].getParseInfoSet()[j];
                                    info->setX(info->x() + curRes.roi_.x());
                                    info->setY(info->y() + curRes.roi_.y());                                    
                                }

                                curRes.computeCandConsistency(NULL);                                

                                bestPtIdx = best;
                            } else {
                                bestPtIdx = idxInPts[best];
                            }

                            curRes.best_ = candIdx[0];
                            curRes.isValid_ = true;

                            const ParseInfo * info = curRes.pts_[curRes.best_].rootParseInfo();
                            curRes.bbox_ = Rectangle(ROUND(info->x()), ROUND(info->y()),
                                                     ROUND(info->width()), ROUND(info->height()));

                            curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
                            curRes.predictScale_ /= 2.0F;


                        } else {
                            curRes.best_ = -1;
                            curRes.isValid_ = false;
                        }
                    } else {
                        Scalar selScore = curRes.pts_[selCandIdx[0]].rootParseInfo()->score_;
                        if((trackability >= thrTrackabilityHigh) &&
                                (score >= sAOG_.thresh()) && (selScore < sAOG_.thresh()) &&
                                (!inPredictableRange)) {
                            // trust detection
                            curRes.best_ = candIdx[0];
                        } else {
                            curRes.best_ = selCandIdx[0];
                        }
                        curRes.isValid_ = true;

                        const ParseInfo * info = curRes.pts_[curRes.best_].rootParseInfo();
                        curRes.bbox_ = Rectangle(ROUND(info->x()), ROUND(info->y()),
                                                 ROUND(info->width()), ROUND(info->height()));

                        curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
                        curRes.predictScale_ /= 2.0F;

                        bestPtIdx = idxInPts[curRes.best_];
                    }
                } else {
                    // lk failed,
                    // which is usually caused by sudden large motion or occlusion
                    best = candIdx[0];
                    bool doubleCheck = inPredictableRange;
                    bool doubleCheck1 = (score >= sAOG_.thresh());
                    float ov, maxov = 0;

                    if(trackability >= thrTrackabilityHigh &&
                            score >= sAOG_.thresh()) {
                        maxov = 1;
                    } else if(doubleCheck || doubleCheck1) {
                        // find the one overlapped with prev
                        Rectangle ref = results_.getOutput()[count_ - 1].bbox_;
                        Intersector_<int> inter(ref, param_.fgOverlap_, param_.nmsDividedByUnion_);

                        for(int c = 0; c < candIdx.size(); ++c) {
                            int d = candIdx[c];
                            const ParseInfo * info = curRes.pts_[d].rootParseInfo();                            
                            Rectangle box(ROUND(info->x()), ROUND(info->y()),
                                          ROUND(info->width()), ROUND(info->height()));

                            inter(box, &ov);
                            if(ov > maxov) {
                                maxov = ov;
                                best = d;
                            }
                            if(maxov >= 0.5F) break;
                        }
                    }

                    if(maxov >= 0.5F) {
                        curRes.best_ = best;
                        curRes.isValid_ = true;
                    } else if(curRes.consistency_ > 0.4F  && (score >= sAOG_.thresh() || maxov >= 0.3F)) { //score >= 0.5F, sAOG_.thresh(),  0.4
                        curRes.best_ = candIdx[0];
                        curRes.isValid_ = true;
                    } else {
                        curRes.best_ = -1;
                        curRes.isValid_ = false;
                    }

                    if(curRes.best_ != -1) {
                        const ParseInfo * info = curRes.pts_[curRes.best_].rootParseInfo();
                        curRes.bbox_ = Rectangle(ROUND(info->x()), ROUND(info->y()),
                                                 ROUND(info->width()), ROUND(info->height()));

                        curRes.predictScale_ += sqrt((Scalar)modelArea_ / curRes.bbox_.area());
                        curRes.predictScale_ /= 2.0F;

                        bestPtIdx = idxInPts[curRes.best_];
                    }
                }
            }
        }
    }

    if(curRes.best_ != -1) {
        numValid++;
        preValidIdx_ = count_;

        trackabilities_.push_back(curRes.pts_[curRes.best_].rootParseInfo()->goodness_);

        if ( bestPtIdx != 0 ) {
            ParseTree tmpPt = pts[0];
            pts[0] = pts[bestPtIdx];
            pts[bestPtIdx] = tmpPt;
        }

        trackability = curRes.pts_[curRes.best_].rootParseInfo()->goodness_;
        score = curRes.pts_[curRes.best_].score();

        if(trackability < thrTrackabilityLow)
            numIntrackable++;

        // get hard negs
        getHardNeg(curRes, pts, candThr);
    } else {
        numIntrackable++;
    }

    // run generic AOG if avaliable (not implemented yet)

    // show
    showResult(cacheDir, curFrameIdx);

    // show score maps
    if ( (param_.showResult_ || param_.showDemo_) && showScoreMaps && curRes.best_!=-1) {
        string scoreMapsDir = tmpDir + "scoreMaps_" + NumToString_<int>(count_+1, 5) + FILESEP;
        FileUtil::VerifyDirectoryExists(scoreMapsDir);
        sDet_.visScoreMaps(scoreMapsDir, &curRes.pts_[curRes.best_]);

        Mat img = curRes.img_.clone();
        curRes.pts_[curRes.best_].showDetection(img, false, cv::Scalar(0, 0, 255), true);
        img = img(curRes.roi_.cvRect());

        string saveName = scoreMapsDir + "img.png";
        cv::imwrite(saveName, img);

        showScoreMaps = false;
    }

    // online learning
    if(!param_.notUpdateAOGStruct_ &&
            ( (param_.regularCheckAOGStruct_ && numValid >= TrackerConst::TimeCheckAOGStructure * 2) ||
             (curRes.best_ != -1 && trackability < thrTrackabilityLow &&
              score < sAOG_.thresh() &&
              numValid >= TrackerConst::TimeCheckAOGStructure &&
              numIntrackable > 5))) {

        sAOG_.save(tmpAOGFilename);
        learner_->copyPool();

        sAOG_.read(emptyAOGFilename);

        bool suc = false;

        trainLoss_ = learner_->train(results_, count_,
                                     param_.numFramesUsedInTrain_,
                                     TrackerConst::NbRelabel, param_.fgOverlap_,
                                     param_.maxBgOverlap_,
                                     param_.nms_, param_.nmsDividedByUnion_,
                                     param_.useOverlapLoss_,
                                     param_.maxNumEx_, param_.C_,
                                     param_.partScale_, true, param_.useRootOnly_,
                                     nodeOnOffHistory_, edgeOnOffHistory_,
                                     saveAOGDir);

        if(trainLoss_ != -std::numeric_limits<Scalar>::infinity()) {
            suc = true;            
        }

        if(suc) {
            if(param_.showResult_ || param_.showDemo_) {
                modelName_ = "AOG_" + NumToString_<int>(count_+1, 5);
                sAOG_.visualize(tmpDir, modelName_);
            }

            if(count_ == TrackerConst::TimeCheckAOGStructure) {
                sAOG_.save(initialAOGFilename);
            }
        } else {
            learner_->restorePool();
            learner_->initMarginBoundPruning();
            sAOG_.read(tmpAOGFilename);
            RGM_LOG(error, "**************** Updating AOG failed.");
        }

        numValid = 0;
        numIntrackable = 0;

        sTau_ += sAOG_.thresh();

        showScoreMaps = true;

    } else if(curRes.best_ != -1) {
        trainLoss_ = learner_->trainOnline(results_.getOutput()[count_], pts,
                                           param_.maxNumEx_,
                                           param_.fgOverlap_,
                                           maxBgOverlap,
                                           param_.nmsDividedByUnion_,
                                           true,
                                           param_.partScale_,
                                           param_.C_);

        sTau_ += sAOG_.thresh();
    }

    release();

    if(!learner_->checkPtPool()) {
        RGM_LOG(error, "************ Parse tree pool has errors.");
    }

    return curRes.bbox_;
}

template<int Dimension, int DimensionG>
void AOGTracker::TLP(int count, Scalar maxNum, vector<ParseTree> &pts) {
    RGM_CHECK((count >= 0 && count <= count_), error);

    OneFrameResult &res(results_.getOutput()[count]);
    res.computeROIFeatPyr(sAOG_.featParam());

    pts.clear();
    sDet_.release();
    sDet_.runDetection(sDetParam_.thresh_, *res.pyr_, maxNum, pts);

    //    if ( pts.size() > 0 ) {
    //        pts[0].showDetection(res.img_(res.roi_.cvRect()), true);
    //    }
}

template<int Dimension, int DimensionG>
void AOGTracker::runGenericAOG(int count, vector<ParseTree_<DimensionG> > &pts) {
    if(gAOG_.empty())
        return;

    // get feature pyr
    OneFrameResult &res(results_.getOutput()[count]);
    RGM_CHECK(!res.img_.empty(), error);

    Mat img = res.img_(res.roi_.cvRect());
    //    Scalar scale = res.workingScale_ * 2;

    //    cv::Size dstSz(res.roi_.width() * scale,  res.roi_.height() * scale);
    //    RGM_CHECK_GT(dstSz.area(), 0);

    //    cv::resize(res.img_(res.roi_.cvRect()), img, dstSz, 0, 0, RGM_IMG_RESIZE);

    gAOG_.getInterval() = 10;

    FeaturePyr_<DimensionG> pyr(img, gAOG_.featParam());

    //    pyr.adjustScales(scale);

    Scalar maxNum = std::numeric_limits<Scalar>::infinity();
    gDet_.runDetection(gDetParam_.thresh_, pyr, maxNum, pts);

    //    for ( int i = 0; i < pts.size(); ++i ) {
    //        pts[i].showDetection(img, true);
    //    }

}

template<int Dimension, int DimensionG>
void AOGTracker::finalizeResult() {
    bool hasDisplacement = (displacement_.x() != 0 || displacement_.y() != 0 ||
            displacement_.width() != 0 || displacement_.height() != 0);

    if(hasDisplacement) {
        for(int i = 1; i < results_.getOutput().size(); ++i) {
            OneFrameResult &res(results_.getOutput()[i]);

            Scalar scale = sqrt(static_cast<Scalar>(res.bbox_.area()) /
                                (sAOG_.maxDetectWindow().area() *
                                 sAOG_.cellSize() * sAOG_.cellSize()));

            int dx = ROUND(displacement_.x() * res.workingScale_ * scale);
            int dy = ROUND(displacement_.y() * res.workingScale_ * scale);
            int dwd = ROUND(displacement_.width() * res.workingScale_ * scale);
            int dht = ROUND(displacement_.height() * res.workingScale_ * scale);

            res.bbox_.setX(res.bbox_.x() + dx);
            res.bbox_.setY(res.bbox_.y() + dy);
            res.bbox_.setWidth(res.bbox_.width() + dwd);
            res.bbox_.setHeight(res.bbox_.height() + dht);
        }
    }

    // save to txt file
    string strResult = data_.cacheDir() + data_.resultFileName() + ".txt";

    std::ofstream ofs(strResult.c_str(), std::ios::out);
    RGM_CHECK(ofs.is_open(), error);

    for(int i = 0; i < results_.getOutput().size(); ++i) {
        OneFrameResult &res(results_.getOutput()[i]);
        ofs << res.bbox_.x() << " " << res.bbox_.y() << " "
            << res.bbox_.width() << " " << res.bbox_.height() << std::endl;
    }

    ofs.close();
}

template<int Dimension, int DimensionG>
void AOGTracker::showResult(const std::string & cacheDir, int frameIdx) {

    if(!param_.showResult_ && !param_.showDemo_) return;

    //    for ( int i = 0; i <= count_; ++i ) {
    //        TrackerResult::OneFrameResult &res(results_.getOutput()[i]);
    //        if ( res.isValid() )
    //            std::cout << i << ":"
    //                      << res.pts_[0].rootNode()->parseInfo(&res.pts_[0])->score_
    //                    << ", ";
    //    }
    //    std::cout << std::endl;

    string tmpDir = cacheDir + "tmp" + FILESEP;
    FileUtil::VerifyDirectoryExists(tmpDir);

    OneFrameResult &res(results_.getOutput()[count_]);

    cv::Mat origShow = curFrameImg_.clone();

    cv::putText(origShow,
                string("#" + NumToString_<int>(count_+1)),
                cv::Point(10, 20),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                cv::Scalar(200, 200, 250), 1, 16); //cv::LINE_AA

    // result image
    cv::Scalar color(0, 0, 255);

    if ( res.best_ != -1 ) {
        color = cv::Scalar(0, 0, 255);
        res.pts_[res.best_].showDetection(origShow, false, color, true);
    } else {
        cv::rectangle(origShow, res.bbox_.cvRect(), color, 2);
    }

    if ( param_.showDemo_ ) {
        int margin = 20;
        cv::Rect roi(res.bbox_.x()  - margin,
                     res.bbox_.y()  - margin,
                     res.bbox_.width() + margin*2, res.bbox_.height() + margin*2);
        cv::Mat crop = OpencvUtil::subarray(origShow, roi, 0);

        objShow_ = cv::Scalar::all(255);
        if ( crop.cols > objROI_.width || crop.rows > objROI_.height ) {
            Scalar factor = std::min<Scalar>((float)objROI_.width/crop.cols,
                                             (float)objROI_.height/crop.rows);

            cv::Size dsz(crop.cols*factor, crop.rows*factor);
            cv::Rect roi((objShow_.cols - dsz.width)/2, (objShow_.rows - dsz.height)/2,
                         dsz.width, dsz.height);
            cv::resize(crop, objShow_(roi), dsz, 0, 0, RGM_IMG_RESIZE);
        } else {
            cv::Rect roi((objShow_.cols - crop.cols)/2, (objShow_.rows - crop.rows)/2,
                         crop.cols, crop.rows);
            crop.copyTo(objShow_(roi));
        }

        int wd = 60;
        cv::Mat ncrop;
        if ( crop.cols > wd ) {
            cv::resize(crop, ncrop, cv::Size(wd, float(crop.rows)/crop.cols * wd), 0, 0, RGM_IMG_RESIZE);
        } else {
            ncrop = crop;
        }
        cv::putText(ncrop,
                    string("#" + NumToString_<int>(count_+1)),
                    cv::Point(2, 10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    cv::Scalar(200, 200, 250), 1, 16); //cv::LINE_AA

        string resultImgName = tmpDir + NumToString_<int>(frameIdx, 5)  + "_patch.png";
        cv::imwrite(resultImgName, ncrop);
    }



    cv::rectangle(origShow, res.roi_.cvRect(), cv::Scalar(0, 0, 0), 2);

    int x = std::max<int>(origShow.cols / 2, origShow.cols - 90);

    cv::putText(origShow,
                "Tau:" + NumToString_<float>(sAOG_.thresh()),
                cv::Point(x - 20, 15),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                cv::Scalar(0, 0, 255), 1, 16); //cv::LINE_AA

    cv::putText(origShow,
                "C:" + NumToString_<float>(res.consistency_),
                cv::Point(x - 20, 30),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                cv::Scalar(0, 0, 255), 1, 16); //cv::LINE_AA

    for(int i = 0; i < res.pts_.size(); ++i) {
        color = rgbTable[rgbTableSz - 1 - i]; //cv::Scalar(100, 100, 100);
        bool showPart = false;
        if(i == res.best_) {
            color = cv::Scalar(0, 0, 255);
            showPart = true;
        }

        if ( i != res.best_ )
            res.pts_[i].showDetection(origShow, false, color, showPart);

        cv::putText(origShow,
                    "S:" + NumToString_<float>(res.pts_[i].score()),
                    cv::Point(x, 50 + i * 40),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    color, 1, 16); //cv::LINE_AA

        cv::putText(origShow,
                    "T:" + NumToString_<float>(res.pts_[i].rootParseInfo()->goodness_),
                    cv::Point(x, 50 + i * 40 + 20),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    color, 1, 16); //cv::LINE_AA
    }

    if(res.lkbox_.width() > 0 && res.lkbox_.height() > 0) {
        color = cv::Scalar(255, 0, 255);
        cv::rectangle(origShow, res.lkbox_.cvRect(), color, 2);
    }

    color = cv::Scalar(100, 100, 100);
    for(int i = 0; i < res.hardNegs_.size(); ++i) {
        cv::rectangle(origShow, res.hardNegs_[i].cvRect(), color, 1);
    }


    string resultImgName = tmpDir + NumToString_<int>(frameIdx, 5) + ".png";
    if(param_.showResult_) {
        cv::imwrite(resultImgName, origShow);
    }

    if(!param_.showDemo_) {
        return;
    }

    ptNameFrmt_ % frameIdx;

    // score map
    std::map<int, Mat> maps;
    if(res.pts_.size() > 0 && res.best_ != -1) {
        sDet_.getPtScoreMaps(res.pts_[res.best_], maps, featParam_.cellSize_ * 100);
    }

    // model
    if(res.best_ != -1) {
        sAOG_.visualize1(tmpDir, ptNameFrmt_.str(),
                         &res.pts_[res.best_], &maps, true, count_ > 0,
                &nodeOnOffHistory_, &edgeOnOffHistory_);
    } else {
        sAOG_.visualize1(tmpDir, ptNameFrmt_.str(),
                         NULL, NULL, false, count_ > 0,
                         &nodeOnOffHistory_, &edgeOnOffHistory_);
    }

    showResult_ = cv::imread(tmpDir + ptNameFrmt_.str() + ".png", cv::IMREAD_COLOR);

    // show
    resultFrame_ = cv::Scalar::all(255);

    objShow_.copyTo(resultFrame_(objROI_));

    cv::resize(origShow, frameShow_, cv::Size(frameShow_.cols, frameShow_.rows));
    frameShow_.copyTo(resultFrame_(frameROI_));

    float factor = std::min<float>(1.0F, std::min<float>(float(resultFrame_.cols - frameShow_.cols) / showResult_.cols,
                                                         (float)resultFrame_.rows / showResult_.rows));
    modelROI_ = cv::Rect(frameShow_.cols, 0, showResult_.cols * factor,
                         showResult_.rows * factor);

    Mat tmpModel;
    cv::resize(showResult_, tmpModel, cv::Size(modelROI_.width, modelROI_.height));
    tmpModel.copyTo(resultFrame_(modelROI_));

    //    cv::imshow(showWindName_, resultFrame_);
    //    cv::waitKey(2);

    if(resultVideo_.isOpened()) {
        if(count_ == 0) {
            for(int i = 0; i < TrackerConst::FPS; ++i)
                resultVideo_ << resultFrame_;
        } else {
            resultVideo_ << resultFrame_;
        }
    }
}

template<int Dimension, int DimensionG>
void AOGTracker::release() {
    vector<int> frameIdx =
            results_.getFrameIdxForTrain(count_, param_.numFramesUsedInTrain_);

    std::sort(frameIdx.begin(), frameIdx.end());

    vector<int> diff;

    std::set_symmetric_difference(allFrameIdx_.begin(),  allFrameIdx_.end(),
                                  frameIdx.begin(), frameIdx.end(),
                                  std::back_inserter(diff));

    for(int i = 0; i < diff.size(); ++i) {
        results_.getOutput()[diff[i]].release();
    }
}

#ifdef RGM_USE_MPI

template<int Dimension, int DimensionG>
void AOGTracker::runMPI() {

    // Get the number of processes
    int ntasks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if(ntasks == 1) {
        run();
        return;
    }

    if(runMode_ > 0 )
        data_.getAllSource();

    if(myrank == 0) {
        std::cout << "=============== Run benchmark evaluation ==============="
                  << std::endl ; fflush(stdout);

        int quit = -1;
        if(!collectResults(true)) {
            for(int t = 1; t < ntasks; ++t)
                MPI_Send(&quit, 1, MPI_INT, t, 0, MPI_COMM_WORLD);
            return;
        }

        vector<int> allDataIdx;
        for(int i = 0; i < data_.numSequences(); ++i) {
            if(data_.setCurSeqIdx(i, true)) {
                allDataIdx.push_back(i);
            } else {
                collectResults(false);
            }
        }

        std::cout << "[0] Total number of sequences: " << allDataIdx.size()
                  << std::endl ; fflush(stdout);

        int d = 0, usedTasks = 1;
        for(; d < allDataIdx.size() && usedTasks < ntasks; ++d, ++usedTasks) {
            int idx = allDataIdx[d];
            MPI_Send(&idx, 1, MPI_INT, usedTasks, 0, MPI_COMM_WORLD);
            std::cout << "[0] sends sequence " << idx << " to [" << usedTasks << "]"
                      << std::endl ; fflush(stdout);
        }

        for(int t = usedTasks; t < ntasks; ++t)
            MPI_Send(&quit, 1, MPI_INT, t, 0, MPI_COMM_WORLD);

        MPI_Status status;
        int dataIdx = 0;

        while(d < allDataIdx.size()) {
            // recieve results
            MPI_Recv(&dataIdx,      /* message buffer */
                     1,                 /* one data item */
                     MPI_INT,           /* of type double real */
                     MPI_ANY_SOURCE,    /* receive from any sender */
                     MPI_ANY_TAG,       /* any type of message */
                     MPI_COMM_WORLD,    /* default communicator */
                     &status);          /* info about the received message */

            int worker = status.MPI_SOURCE;

            // write results
            data_.setCurSeqIdx(dataIdx);
            collectResults(false);

            std::cout << "[0] recieves results from [" << worker << "] of sequence "
                      << data_.resultFileName() << std::endl ; fflush(stdout);

            // send next work
            int idx = allDataIdx[d];
            MPI_Send(&idx, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            std::cout << "[0] sends sequence " << idx << " to [" << worker << "]"
                      << std::endl ; fflush(stdout);
            d++;
        }

        for(int t = 1; t < usedTasks; ++t) {
            // recieve results
            MPI_Recv(&dataIdx,      /* message buffer */
                     1,                 /* one data item */
                     MPI_FLOAT,           /* of type double real */
                     MPI_ANY_SOURCE,    /* receive from any sender */
                     MPI_ANY_TAG,       /* any type of message */
                     MPI_COMM_WORLD,    /* default communicator */
                     &status);          /* info about the received message */

            int worker = status.MPI_SOURCE;

            std::cout << "[0] recieves results from [" << worker << "] of sequence "
                      << dataIdx << std::endl ; fflush(stdout);

            MPI_Send(&quit, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);

            // write results
            data_.setCurSeqIdx(dataIdx);
            collectResults(false);
        }
    } else {
        int d = 0;
        float speed = 0;
        Timers timer;
        while(true) {
            speed = 0;
            MPI_Recv(&d, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(d < 0)  break;

            if(!data_.setCurSeqIdx(d)) {
                std::cout << "[" << myrank << "] sequence " << d << " "
                          << data_.resultFileName() << " done already"
                          << std::endl ; fflush(stdout);
                MPI_Send(&d, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                continue;
            } else {
                std::cout << "[" << myrank << "] recieves sequence " << d << ": "
                          << data_.resultFileName() << std::endl ; fflush(stdout);
            }                      

            Timers::Task *curTimer = timer(data_.curSequenceName());
            curTimer->Start();

            init();
            if(!initAOG()) {
                release();
                curTimer->Stop();
                std::cout << "[" << myrank << "]: ****** Failed to track sequence "
                          << d << " " << data_.resultFileName()
                          << std::endl ; fflush(stdout);
                MPI_Send(&d, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                continue;
            }

            runAOGTracker();

            curTimer->Stop();
            speed =  data_.numFrames() / curTimer->ElapsedSeconds();

            // save results
            string speedFilename = data_.cacheDir() + data_.resultFileName() + "_speed.txt";
            std::ofstream ofs(speedFilename.c_str(), std::ios::out);
            if(ofs.is_open()) {
                ofs << curTimer->ElapsedSeconds() << ", " << speed << std::endl;
                ofs.close();
            }

            MPI_Send(&d, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            std::cout << "[" << myrank << "] sends results of sequence "
                      << d << " " << data_.resultFileName() << " speed: " << speed
                      << std::endl ; fflush(stdout);
        }

        timer.clear();
    }
}

#endif

template<int Dimension, int DimensionG>
bool AOGTracker::collectResults(bool firstTime) {
    if(runMode_ > 0) {
        string resultFileOPE  = data_.rootDir() + "AOGTracker_Result_OPE_" +
                data_.note() + ".txt";
        std::ofstream ofsOPE(resultFileOPE.c_str(), (firstTime ? std::ios::out : std::ios::app));

        string resultFileTRE  = data_.rootDir() + "AOGTracker_Result_TRE_" +
                data_.note() + ".txt";
        std::ofstream ofsTRE(resultFileTRE.c_str(), (firstTime ? std::ios::out : std::ios::app));

        string resultFileSRE  = data_.rootDir() + "AOGTracker_Result_SRE_" +
                data_.note() + ".txt";
        std::ofstream ofsSRE(resultFileSRE.c_str(), (firstTime ? std::ios::out : std::ios::app));

        if(!ofsOPE.is_open() || !ofsTRE.is_open() || !ofsSRE.is_open()) {
            std::cout << "Failed to open files to write results in " << data_.rootDir()
                      << std::endl ; fflush(stdout);

            ofsOPE.close();
            ofsTRE.close();
            ofsSRE.close();

            return false;
        }

        if(firstTime) {
            ofsOPE.close();
            ofsTRE.close();
            ofsSRE.close();
            return true;
        }

        string resultFilename = data_.cacheDir() + data_.resultFileName() + ".txt";
        if(!FileUtil::exists(resultFilename)) {
            ofsOPE.close();
            ofsTRE.close();
            ofsSRE.close();
            return true;
        }

        float timeConsumed = 0;
        float speed = 0;
        string speedFilename = data_.cacheDir() + data_.resultFileName() + "_speed.txt";
        std::ifstream ifs(speedFilename.c_str(), std::ios::in);
        string format("%f, %f");
        if(ifs.is_open()) {
            string  line;
            std::getline(ifs, line);
            sscanf(line.c_str(), format.c_str(), &timeConsumed, &speed);
            ifs.close();
        }

        std::ofstream *ofs = NULL;
        if(resultFilename.find("_OPE_", 0) != string::npos) {
            ofs = &ofsOPE;
        } else if(resultFilename.find("_TRE_", 0) != string::npos) {
            ofs = &ofsTRE;
        } else if(resultFilename.find("_SRE_", 0) != string::npos) {
            ofs = &ofsSRE;
        }

        if(ofs != NULL) {
            *ofs << data_.curSequenceName() << "\t"
                 << resultFilename << "\t" << data_.gtFile() << "\t"
                 << data_.dataStartFrameIdx0() << "\t"
                 << data_.dataStartFrameIdx() << "\t" << data_.dataEndFrameIdx() << "\t"
                 << timeConsumed << "\t" << speed << "\t"
                 << data_.dataShiftType() << std::endl;

            if(ofs == &ofsOPE) {
                ofsTRE << data_.curSequenceName() << "\t"
                       << resultFilename << "\t" << data_.gtFile() << "\t"
                       << data_.dataStartFrameIdx0() << "\t"
                       << data_.dataStartFrameIdx() << "\t" << data_.dataEndFrameIdx() << "\t"
                       << timeConsumed << "\t" << speed << "\t"
                       << data_.dataShiftType() << std::endl;
            }
        }

        ofsOPE.close();
        ofsTRE.close();
        ofsSRE.close();
    } else {
        string resultFile  = data_.rootDir() + "AOGTracker_Result_" +
                data_.note() + ".txt";
        std::ofstream ofs(resultFile.c_str(), (firstTime ? std::ios::out : std::ios::app));
        if(!ofs.is_open()) {
            std::cerr << "Failed to open files to write results in " << data_.rootDir()
                      << std::endl ; fflush(stdout);
            return false;
        }

        if(firstTime) return true;

        string resultFilename = data_.cacheDir() + data_.resultFileName() + ".txt";
        if(!FileUtil::exists(resultFilename)) {
            ofs.close();
            return true;
        }

        float timeConsumed = 0;
        float speed = 0;
        string speedFilename = data_.cacheDir() + data_.resultFileName() + "_speed.txt";
        std::ifstream ifs(speedFilename.c_str(), std::ios::in);
        string format("%f, %f");
        if(ifs.is_open()) {
            string  line;
            std::getline(ifs, line);
            sscanf(line.c_str(), format.c_str(), &timeConsumed, &speed);
            ifs.close();
        }

        ofs << data_.curSequenceName() << "\t"
            << resultFilename << "\t" << data_.gtFile() << "\t"
            << data_.dataStartFrameIdx0() << "\t"
            << data_.dataStartFrameIdx() << "\t" << data_.dataEndFrameIdx() << "\t"
            << timeConsumed << "\t" << speed << "\t"
            << data_.dataShiftType() << std::endl;

        ofs.close();
    }

    return true;
}

template<int Dimension, int DimensionG>
void AOGTracker::getHardNeg(OneFrameResult_<Dimension> &curRes,
                            vector<ParseTree_<Dimension> > &pts, Scalar candThr) {
    curRes.hardNegs_.clear();
    if(param_.showResult_ || param_.showDemo_) {
        vector<bool> isKept(pts.size(), true);
        for(int d = 0; d < pts.size();) {
            const ParseInfo *infoi = pts[d].rootParseInfo();
            Rectangle_<Scalar> boxi((infoi->x()), (infoi->y()),
                                    (infoi->width()), (infoi->height()));

            Intersector_<Scalar> inter(boxi, (d == 0 ? param_.maxBgOverlap_ : 0.5F), true);

            for(int j = d + 1; j < pts.size(); ++j) {
                if(!isKept[j]) continue;
                const ParseInfo *infoj = pts[j].rootParseInfo();
                Rectangle_<Scalar> boxj((infoj->x()), (infoj->y()),
                                        (infoj->width()), (infoj->height()));

                isKept[j]  = !inter(boxj);
            }

            ++d;
            while(!isKept[d]) ++d;
        }
        isKept[0] = false;

        for(int d = 1; d < isKept.size() &&
            curRes.hardNegs_.size() < param_.numCand_; ++d) {
            if(isKept[d]) {
                const ParseInfo *info = pts[d].rootParseInfo();
                if(info->score_ < candThr) break;
                curRes.hardNegs_.push_back(Rectangle(ROUND(info->x() + curRes.roi_.x()),
                                                     ROUND(info->y() + curRes.roi_.y()),
                                                     ROUND(info->width()), ROUND(info->height())));
            }
        }
    }
}

// Instantiation
template class AOGTracker_<22, 32>;
template class AOGTracker_<22, 42>;
template class AOGTracker_<22, 48>;

template class AOGTracker_<28, 32>;
template class AOGTracker_<28, 42>;
template class AOGTracker_<28, 48>;

template class AOGTracker_<32, 32>;
template class AOGTracker_<32, 42>;
template class AOGTracker_<32, 48>;

template class AOGTracker_<38, 32>;
template class AOGTracker_<38, 42>;
template class AOGTracker_<38, 48>;

template class AOGTracker_<42, 32>;
template class AOGTracker_<42, 42>;
template class AOGTracker_<42, 48>;

template class AOGTracker_<48, 32>;
template class AOGTracker_<48, 42>;
template class AOGTracker_<48, 48>;


void RunAOGTracker(std::string &configFilename) {
    CvFileStorage * fs = cvOpenFileStorage(configFilename.c_str(), 0,
                                           CV_STORAGE_READ);
    if ( fs == NULL ) {
        std::cerr << "Not found " << configFilename << std::endl ; fflush(stdout);
        return;
    }

    int sFeatType = cvReadIntByName(fs, 0, "FeatType", 4);
    int gFeatType = cvReadIntByName(fs, 0, "GFeatType", 0);

    cvReleaseFileStorage(&fs);

    // get feature type and dimension
    const int sDimension = FeatureDim[sFeatType];
    const int gDimension = FeatureDim[gFeatType];

    switch(sDimension) {
    case 22: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<22, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<22, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<22, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    case 28: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<28, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<28, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<28, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    case 32: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<32, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<32, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<32, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    case 42: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<42, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<42, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<42, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    case 38: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<38, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<38, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<38, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    case 48: {
        switch(gDimension) {
        case 32: {
            AOGTracker_<48, 32>  tracker(configFilename);
            break;
        }
        case 42: {
            AOGTracker_<48, 42>  tracker(configFilename);
            break;
        }
        case 48: {
            AOGTracker_<48, 48>  tracker(configFilename);
            break;
        }
        default: {
            std::cerr << "wrong feature type" ; fflush(stdout);
            break;
        }
        }
        break;
    }
    default:
        std::cerr << "wrong feature type";
        break;
    }

}


} // namespace RGM
