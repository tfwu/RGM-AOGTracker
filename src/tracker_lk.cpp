#include "tracker_lk.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilMath.hpp"

namespace RGM {

LKTracker::LKTracker() :
    curIdx_(1), nextIdx_(0) {
    img_.resize(2);
    pyr_.resize(2);
}

bool LKTracker::runLK(Mat curImg, Rectangle &inputBbox, Mat nextImg,
                      Rectangle * nextBbox) {
    RGM_CHECK(!curImg.empty() && !nextImg.empty(), error);

    std::swap(curIdx_, nextIdx_);

    if(inputBbox.width() < MarginBetweenPoints * 3 ||
            inputBbox.height() < MarginBetweenPoints * 3)
        return false;

    if(nextBbox != NULL) {
        if(nextBbox->width() < MarginBetweenPoints * 3 ||
                nextBbox->height() < MarginBetweenPoints * 3)
            return false;
    }

    vector<cv::Point2f> templatPts, targetPts, fwdbckPts;
    templatPts = OpencvUtil_<int>::getTrackPoints(inputBbox.cvRect(),
                                                  NbRowTrackPoints,
                                                  NbColTrackPoints,
                                                  MarginBetweenPoints);
    int nPts = templatPts.size();
    if(nPts == 0)
        return false;

    if(nextBbox == NULL) {
        targetPts = templatPts;
    } else {
        targetPts = OpencvUtil_<int>::getTrackPoints(nextBbox->cvRect(),
                                                     NbRowTrackPoints,
                                                     NbColTrackPoints,
                                                     MarginBetweenPoints);
        if(targetPts.size() != templatPts.size()) {
            targetPts = templatPts;
        }
    }

    fwdbckPts = templatPts;

    const float fwdbckErrThr = std::max<int>(5, templatPts.size() * 0.1);

    int Winsize = 10;

    vector<float> ncc(nPts, 0.f);
    vector<float> fb(nPts, 0.f);
    vector<unsigned char>  statusFwd, statusBckwd;
    vector<float> errFwd, errBckwd;

    cv::Size sz(SzWin, SzWin);
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                              20, 0.03);


    if(img_[curIdx_].cols != curImg.cols ||
            img_[curIdx_].rows != curImg.rows) {
        img_[curIdx_] = curImg;
        cv::buildOpticalFlowPyramid(img_[curIdx_], pyr_[curIdx_], sz, NbLevel);
    }

    img_[nextIdx_] = nextImg;
    cv::buildOpticalFlowPyramid(img_[nextIdx_], pyr_[nextIdx_], sz, NbLevel);

    cv::calcOpticalFlowPyrLK(pyr_[curIdx_], pyr_[nextIdx_],
                             templatPts, targetPts, statusFwd, errFwd,
                             sz, NbLevel,
                             termcrit,
                             cv::OPTFLOW_USE_INITIAL_FLOW |
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    cv::calcOpticalFlowPyrLK(pyr_[nextIdx_], pyr_[curIdx_],
                             targetPts, fwdbckPts, statusBckwd, errBckwd,
                             sz, NbLevel,
                             termcrit,
                             cv::OPTFLOW_USE_INITIAL_FLOW |
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    OpencvUtil::NCC(img_[curIdx_], img_[nextIdx_], templatPts, targetPts, statusFwd,
                    ncc, Winsize, cv::TM_CCOEFF_NORMED);

    OpencvUtil::euclideanDistance(templatPts, fwdbckPts, fb);

    //get median of fb and ncc
    vector<float> validfb, validncc;
    vector<cv::Point2f> validTemplatePts, validTargetPts;

    for(int i = 0; i < nPts; i++) {
        if(statusFwd[i] == (unsigned char)1) {
            validTemplatePts.push_back(templatPts[i]);
            validTargetPts.push_back(targetPts[i]);
            validfb.push_back(fb[i]);
            validncc.push_back(ncc[i]);
        }
    }

    float medFB = MathUtil_<float>::median(validfb);
    float medNCC = MathUtil_<float>::median(validncc);

    vector<cv::Point2f>().swap(templatPts);
    vector<cv::Point2f>().swap(targetPts);

    for(int i = 0; i < validfb.size(); ++i) {
        if(validfb[i] <= medFB && validncc[i] >= medNCC) {
            templatPts.push_back(validTemplatePts[i]);
            targetPts.push_back(validTargetPts[i]);
        }
    }

    if(validfb.size() == 0)
        return false;
    if(medFB > fwdbckErrThr)
        return false; //too unstable predictions

    // prediction
    int numPts = templatPts.size();

    vector<float> dx(numPts, 0), dy(numPts, 0);

    for(int i = 0; i < numPts; ++i) {
        dx[i] = targetPts[i].x - templatPts[i].x;
        dy[i] = targetPts[i].y - templatPts[i].y;
    }

    float dxMed = MathUtil_<float>::median(dx);
    float dyMed = MathUtil_<float>::median(dy);

    vector<float> d1 = MathUtil_<float>::pdist(templatPts);
    vector<float> d2 = MathUtil_<float>::pdist(targetPts);

    vector<float> d12;
    for(int i = 0; i < d1.size(); ++i) {
        d12.push_back(d2[i] / d1[i]);
    }

    float s = MathUtil_<float>::median(d12);

    float shiftx = 0.5 * (s - 1) * inputBbox.width();
    float shifty = 0.5 * (s - 1) * inputBbox.height();

    predictBbox_.setX(inputBbox.x() - shiftx + dxMed) ;
    predictBbox_.setY(inputBbox.y() - shifty + dyMed);
    predictBbox_.setWidth(inputBbox.width() + 2 * shiftx);
    predictBbox_.setHeight(inputBbox.height() + 2 * shifty);

    float scaleChange = ((float)predictBbox_.area() / inputBbox.area());
    if(scaleChange > 2 || scaleChange < 0.5F)
        return false;

#if 0
    Mat bigImg(img_[curIdx_].rows + img_[nextIdx_].rows,
               img_[curIdx_].cols + img_[nextIdx_].cols,
               img_[curIdx_].type(),
               cv::Scalar::all(0));
    img_[curIdx_].copyTo(bigImg(cv::Rect(0, 0, img_[curIdx_].cols,
                                         img_[curIdx_].rows)));
    img_[nextIdx_].copyTo(bigImg(cv::Rect(img_[curIdx_].cols,
                                          img_[curIdx_].rows,
                                          img_[nextIdx_].cols,
                                          img_[nextIdx_].rows)));

    for(int i = 0; i < templatPts.size(); ++i) {
        cv::circle(bigImg, cv::Point(templatPts[i].x, templatPts[i].y), 1,
                   cv::Scalar::all(128), 2);
        cv::circle(bigImg, cv::Point(targetPts[i].x + img_[curIdx_].cols,
                                     templatPts[i].y + img_[curIdx_].rows), 1,
                   cv::Scalar::all(0), 2);
        cv::line(bigImg, cv::Point(templatPts[i].x, templatPts[i].y),
                 cv::Point(targetPts[i].x + img_[curIdx_].cols,
                           templatPts[i].y + img_[curIdx_].rows),
                 cv::Scalar::all(255), 1);
    }
    cv::imshow("LK", bigImg);
    cv::waitKey(0);
#endif

    return true;

}

bool LKTracker::runLK1(Mat curImg, Rectangle &inputBbox, Mat nextImg,
                      Rectangle * nextBbox) {
    RGM_CHECK(!curImg.empty() && !nextImg.empty(), error);

    if(inputBbox.width() < MarginBetweenPoints * 3 ||
            inputBbox.height() < MarginBetweenPoints * 3)
        return false;

    if(nextBbox != NULL) {
        if(nextBbox->width() < MarginBetweenPoints * 3 ||
                nextBbox->height() < MarginBetweenPoints * 3)
            return false;
    }

    vector<cv::Point2f> templatPts, targetPts, fwdbckPts;
    templatPts = OpencvUtil_<int>::getTrackPoints(inputBbox.cvRect(),
                                                  NbRowTrackPoints,
                                                  NbColTrackPoints,
                                                  MarginBetweenPoints);
    int nPts = templatPts.size();
    if(nPts == 0)
        return false;

    if(nextBbox == NULL) {
        targetPts = templatPts;
    } else {
        targetPts = OpencvUtil_<int>::getTrackPoints(nextBbox->cvRect(),
                                                     NbRowTrackPoints,
                                                     NbColTrackPoints,
                                                     MarginBetweenPoints);
        if(targetPts.size() != templatPts.size()) {
            targetPts = templatPts;
        }
    }

    fwdbckPts = templatPts;

    const float fwdbckErrThr = 10; //std::max<int>(5, templatPts.size() * 0.1); //std::min<int>(10, templatPts.size() * 0.1);

    int Winsize = 10;

    vector<float> ncc(nPts, 0.f);
    vector<float> fb(nPts, 0.f);
    vector<unsigned char>  statusFwd, statusBckwd;
    vector<float> errFwd, errBckwd;

    cv::Size sz(SzWin, SzWin);
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                              20, 0.03);

    img_[curIdx_] = curImg;
    cv::buildOpticalFlowPyramid(img_[curIdx_], pyr_[curIdx_], sz, NbLevel);

    img_[nextIdx_] = nextImg;
    cv::buildOpticalFlowPyramid(img_[nextIdx_], pyr_[nextIdx_], sz, NbLevel);

    cv::calcOpticalFlowPyrLK(pyr_[curIdx_], pyr_[nextIdx_],
                             templatPts, targetPts, statusFwd, errFwd,
                             sz, NbLevel,
                             termcrit,
                             cv::OPTFLOW_USE_INITIAL_FLOW |
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    cv::calcOpticalFlowPyrLK(pyr_[nextIdx_], pyr_[curIdx_],
                             targetPts, fwdbckPts, statusBckwd, errBckwd,
                             sz, NbLevel,
                             termcrit,
                             cv::OPTFLOW_USE_INITIAL_FLOW |
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    OpencvUtil::NCC(img_[curIdx_], img_[nextIdx_], templatPts, targetPts, statusFwd,
                    ncc, Winsize, cv::TM_CCOEFF_NORMED);

    OpencvUtil::euclideanDistance(templatPts, fwdbckPts, fb);

    //get median of fb and ncc
    vector<float> validfb, validncc;
    vector<cv::Point2f> validTemplatePts, validTargetPts;

    for(int i = 0; i < nPts; i++) {
        if(statusFwd[i] == (unsigned char)1) {
            validTemplatePts.push_back(templatPts[i]);
            validTargetPts.push_back(targetPts[i]);
            validfb.push_back(fb[i]);
            validncc.push_back(ncc[i]);
        }
    }

    float medFB = MathUtil_<float>::median(validfb);
    float medNCC = MathUtil_<float>::median(validncc);

    vector<cv::Point2f>().swap(templatPts);
    vector<cv::Point2f>().swap(targetPts);

    for(int i = 0; i < validfb.size(); ++i) {
        if(validfb[i] <= medFB && validncc[i] >= medNCC) {
            templatPts.push_back(validTemplatePts[i]);
            targetPts.push_back(validTargetPts[i]);
        }
    }

    if(validfb.size() == 0)
        return false;
    if(medFB > fwdbckErrThr)
        return false; //too unstable predictions

    // prediction
    int numPts = templatPts.size();

    vector<float> dx(numPts, 0), dy(numPts, 0);

    for(int i = 0; i < numPts; ++i) {
        dx[i] = targetPts[i].x - templatPts[i].x;
        dy[i] = targetPts[i].y - templatPts[i].y;
    }

    float dxMed = MathUtil_<float>::median(dx);
    float dyMed = MathUtil_<float>::median(dy);

    vector<float> d1 = MathUtil_<float>::pdist(templatPts);
    vector<float> d2 = MathUtil_<float>::pdist(targetPts);

    vector<float> d12;
    for(int i = 0; i < d1.size(); ++i) {
        d12.push_back(d2[i] / d1[i]);
    }

    float s = MathUtil_<float>::median(d12);

    float shiftx = 0.5 * (s - 1) * inputBbox.width();
    float shifty = 0.5 * (s - 1) * inputBbox.height();

    predictBbox_.setX(inputBbox.x() - shiftx + dxMed) ;
    predictBbox_.setY(inputBbox.y() - shifty + dyMed);
    predictBbox_.setWidth(inputBbox.width() + 2 * shiftx);
    predictBbox_.setHeight(inputBbox.height() + 2 * shifty);

    float scaleChange = ((float)predictBbox_.area() / inputBbox.area());
    if(scaleChange > 4 || scaleChange < 0.25F)
        return false;

#if 0
    Mat bigImg(img_[curIdx_].rows + img_[nextIdx_].rows,
               img_[curIdx_].cols + img_[nextIdx_].cols,
               img_[curIdx_].type(),
               cv::Scalar::all(0));
    img_[curIdx_].copyTo(bigImg(cv::Rect(0, 0, img_[curIdx_].cols,
                                         img_[curIdx_].rows)));
    img_[nextIdx_].copyTo(bigImg(cv::Rect(img_[curIdx_].cols,
                                          img_[curIdx_].rows,
                                          img_[nextIdx_].cols,
                                          img_[nextIdx_].rows)));

    for(int i = 0; i < templatPts.size(); ++i) {
        cv::circle(bigImg, cv::Point(templatPts[i].x, templatPts[i].y), 1,
                   cv::Scalar::all(128), 2);
        cv::circle(bigImg, cv::Point(targetPts[i].x + img_[curIdx_].cols,
                                     templatPts[i].y + img_[curIdx_].rows), 1,
                   cv::Scalar::all(0), 2);
//        cv::line(bigImg, cv::Point(templatPts[i].x, templatPts[i].y),
//                 cv::Point(targetPts[i].x + img_[curIdx_].cols,
//                           templatPts[i].y + img_[curIdx_].rows),
//                 cv::Scalar::all(255), 1);
    }
    cv::imshow("LK", bigImg);
    cv::waitKey(0);
#endif

    return true;

}

}
