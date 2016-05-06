#include "boost/random.hpp"

#include "tracker_result.hpp"
#include "util/UtilOpencv.hpp"

namespace RGM {

template<int Dimension>
OneFrameResult::OneFrameResult_() :
    best_(-1), consistency_(0.0F), workingScale_(1.0F), predictScale_(1.0),
    pyr_(NULL), isValid_(false) {

}

template<int Dimension>
OneFrameResult::~OneFrameResult_() {
    release();
}

template<int Dimension>
OneFrameResult & OneFrameResult::operator =(const OneFrameResult &res) {
    if(this == &res) {
        return *this;
    }

    release();

    pts_ = res.pts_;
    best_ = res.best_;
    isValid_ = res.isValid_;
    consistency_ = res.consistency_;

    img_ = res.img_;
    roi_ = res.roi_;
    bbox_ = res.bbox_;
    hardNegs_ = res.hardNegs_;

    grayImg_ = res.grayImg_;

    lkbox_ = res.lkbox_;

    workingScale_ = res.workingScale_;
    predictScale_ = res.predictScale_;

    if(pyr_ != NULL) {
        delete pyr_;
        pyr_ = NULL;
    }
    if(res.pyr_ != NULL) {
        pyr_ = new FeaturePyr(*res.pyr_);
    }

    warpedPos_ = res.warpedPos_;
    warpedPosFeat_ = res.warpedPosFeat_;
    warpedPosFeatX_ = res.warpedPosFeatX_;

    warpedBbox_ = res.warpedBbox_;
    for(int i = 0; i < warpedPyrs_.size(); ++i) {
        if(warpedPyrs_[i] != NULL)   {
            delete warpedPyrs_[i];
            warpedPyrs_[i] = NULL;
        }
    }
    warpedPyrs_.resize(res.warpedPyrs_.size(), NULL);
    for(int i = 0; i < res.warpedPyrs_.size(); ++i) {
        warpedPyrs_[i] = new FeaturePyr(*res.warpedPyrs_[i]);
    }

    return *this;
}

template<int Dimension>
void OneFrameResult::shiftPts() {
    // put pts in the original frame coordinate
    for(int k = 0; k < pts_.size(); ++k) {
        ParseTree &pt(pts_[k]);
        for(int j = 0; j < pt.parseInfoSet().size(); ++j) {
            ParseInfo * info = pt.getParseInfoSet()[j];
            info->setX(info->x() + roi_.x());
            info->setY(info->y() + roi_.y());
            //info->clipBbox(img_.cols, img_.rows);
        }
    }
}

template<int Dimension>
void OneFrameResult::computeROIFeatPyr(const FeatureParam &featParam) {

    RGM_CHECK(!img_.empty(), error);

    Mat img;
    Scalar scale = workingScale_ * featParam.scaleBase_;

    cv::Size dstSz(roi_.width() * scale,  roi_.height() * scale);
    RGM_CHECK_GT(dstSz.area(), 0);

    if((pyr_ != NULL) &&
            (pyr_->imgWd() == dstSz.width && pyr_->imgHt() == dstSz.height)) {
        pyr_->getValidLevels().assign(pyr_->validLevels().size(), true);
        return;
    }

    if(pyr_ != NULL) {
        delete pyr_;
        pyr_ = NULL;
    }

    cv::resize(img_(roi_.cvRect()), img, dstSz, 0, 0, RGM_IMG_RESIZE);

    pyr_ = new FeaturePyr(img, featParam);

    pyr_->adjustScales(scale);
}

template<int Dimension>
void OneFrameResult::computeWarpedFeatPyr(const FeatureParam &featParam) {

    if(warpedPyrs_.size() == warpedPos_.size()) {
        for(int i = 0; i < warpedPyrs_.size(); ++i) {
            warpedPyrs_[i]->getValidLevels().assign(warpedPyrs_[i]->validLevels().size(), true);
        }
        return;
    }

    RGM_CHECK(!img_.empty(), error);
    RGM_CHECK_GT(warpedPos_.size(), 0);

    for(int i = 0; i < warpedPyrs_.size(); ++i) {
        if(warpedPyrs_[i] != NULL)   {
            delete warpedPyrs_[i];
            warpedPyrs_[i] = NULL;
        }
    }
    warpedPyrs_.resize(warpedPos_.size(), NULL);
    warpedImgs_.resize(warpedPos_.size());

    cv::Rect roi = bbox_.cvRect();
    int sz = std::max<int>(roi.width, roi.height);
    const int maxROISz = TrackerConst::maxROISz;
    int padx = std::min<int>(sz,  (std::min<int>(maxROISz, img_.cols) - roi.width) / 2);
    int pady = std::min<int>(sz, (std::min<int>(maxROISz, img_.rows) - roi.height) / 2);

    roi.x -= padx;
    roi.y -= pady;
    roi.width += 2 * padx;
    roi.height += 2 * pady;

    warpedBbox_ = Rectangle(padx, pady, bbox_.width(), bbox_.height());

    Mat roiImg = OpencvUtil::subarray(img_, roi, 1);

    cv::Rect warpROI(bbox_.xcenter() - roi.x - warpedPos_[0].cols / 2,
            bbox_.ycenter() - roi.y - warpedPos_[0].rows / 2,
            warpedPos_[0].cols, warpedPos_[0].rows);

    Scalar scale = workingScale_ * featParam.scaleBase_;
    cv::Size dstSz(roiImg.cols * scale,  roiImg.rows * scale);
    RGM_CHECK_GT(dstSz.area(), 0);

#pragma omp parallel for
    for(int i = 0; i < warpedPos_.size(); ++i) {
        warpedImgs_[i] = roiImg.clone();
        warpedPos_[i].copyTo(warpedImgs_[i](warpROI));
        Mat img, imgf;
        cv::resize(warpedImgs_[i], img, dstSz, 0, 0, RGM_IMG_RESIZE);

        //        cv::imshow("debug1", warpedPos_[i]);
        //        cv::waitKey(0);
        //        cv::rectangle(warpedImgs_[i], warpedBbox_.cvRect(), cv::Scalar(0, 0, 255), 2);
        //        cv::imshow("debug", warpedImgs_[i]);
        //        cv::waitKey(0);

        img.convertTo(imgf,
                      CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                  img.channels()));

        warpedPyrs_[i] = new FeaturePyr(imgf, featParam);
        warpedPyrs_[i]->adjustScales(scale);
    }
}

template<int Dimension>
int OneFrameResult::prepareWarpPos(int modelWd, int modelHt, bool isInit,
                                   const FeatureParam & featParam) {

    const int NbWarpedPos = (isInit ? TrackerConst::NbWarpedPosInit :
                                      TrackerConst::NbWarpedPosUpdate);

    bool valid = (warpedPos_.size() == NbWarpedPos &&
                  warpedPosFeat_.size() == NbWarpedPos &&
                  warpedPosFeat_[0].rows() == modelHt &&
            warpedPosFeat_[0].cols() == modelWd);

    bool validx = (featParam.partOctave_ == 0 || (featParam.partOctave_ == 1 &&
                                                  (warpedPosFeatX_.size() == NbWarpedPos &&
                                                   warpedPosFeatX_[0].rows() == modelHt * 2 &&
                                                  warpedPosFeatX_[0].cols() == modelWd * 2)));

    if(valid && validx) {
        return warpedPosFeat_.size();
    }

    FeatureParam fParam = featParam;
    fParam.padx_ = 0;
    fParam.pady_ = 0;
    int cellSz = fParam.cellSize_;

    if(!valid) {
        warpParam param;
        if(!isInit) {
            param.update();
        }
        param.num_ = NbWarpedPos;

        cv::Rect cropROI = bbox_.cvRect();

        int pady = ROUND((Scalar)cropROI.height / modelHt);
        int padx = ROUND((Scalar)cropROI.width  / modelWd);

        cropROI.x -= padx;
        cropROI.y -= pady;
        cropROI.width += padx * 2;
        cropROI.height += pady * 2;

        warpedPos_ = OpencvUtil::warp(img_, cropROI, param);

        int cellSz = fParam.cellSize_;
        cv::Size patchSz((modelWd + 2) * cellSz, (modelHt + 2) * cellSz);
        cv::Size patchSzX((modelWd * 2 + 2) * cellSz, (modelHt * 2 + 2) * cellSz);

        Cell bg;
        bg.setZero();

        warpedPosFeat_.resize(NbWarpedPos);
        if(fParam.partOctave_) {
            warpedPosFeatX_.resize(NbWarpedPos);
        }

#pragma omp parallel for
        for(int i = 0; i < NbWarpedPos; ++i) {
            Mat patch;
            Mat patchf;

            if(fParam.partOctave_) {
                cv::resize(warpedPos_[i], patch, patchSzX, 0, 0, RGM_IMG_RESIZE);
                patch.convertTo(patchf,
                                CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                            patch.channels()));

                FeaturePyr::computeFeature(patchf, fParam, bg,
                                           warpedPosFeatX_[i]);

            }

            cv::resize(warpedPos_[i], patch, patchSz, 0, 0, RGM_IMG_RESIZE);
            patch.convertTo(patchf,
                            CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                        patch.channels()));

            FeaturePyr::computeFeature(patchf, fParam, bg, warpedPosFeat_[i]);
        }

        if(fParam.partOctave_) {
            RGM_CHECK_EQ(warpedPosFeatX_[0].rows(), modelHt * 2);
            RGM_CHECK_EQ(warpedPosFeatX_[0].cols(), modelWd * 2);
        }

        RGM_CHECK_EQ(warpedPosFeat_[0].rows(), modelHt);
        RGM_CHECK_EQ(warpedPosFeat_[0].cols(), modelWd);

    } else if(!validx) {
        cv::Size patchSzX((modelWd * 2 + 2) * cellSz, (modelHt * 2 + 2) * cellSz);

        Cell bg;
        bg.setZero();

        warpedPosFeat_.resize(NbWarpedPos);
        if(fParam.partOctave_) {
            warpedPosFeatX_.resize(NbWarpedPos);
        }

#pragma omp parallel for
        for(int i = 0; i < NbWarpedPos; ++i) {
            Mat patch;
            Mat patchf;

            cv::resize(warpedPos_[i], patch, patchSzX, 0, 0, RGM_IMG_RESIZE);
            patch.convertTo(patchf,
                            CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                        patch.channels()));

            FeaturePyr::computeFeature(patchf, fParam, bg, warpedPosFeatX_[i]);
        }

        RGM_CHECK_EQ(warpedPosFeatX_[0].rows(), modelHt * 2);
        RGM_CHECK_EQ(warpedPosFeatX_[0].cols(), modelWd * 2);
    }

    return warpedPosFeat_.size();
}

template<int Dimension>
int OneFrameResult::prepareShiftPos(int modelWd, int modelHt,
                                    const FeatureParam & featParam) {

    const int NbWarpedPos = 9; // shift around each corner + orig

    bool valid = (warpedPos_.size() == NbWarpedPos &&
                  warpedPosFeat_.size() == NbWarpedPos &&
                  warpedPosFeat_[0].rows() == modelHt &&
            warpedPosFeat_[0].cols() == modelWd);

    bool validx = (featParam.partOctave_ == 0 || (featParam.partOctave_ == 1 &&
                                                  (warpedPosFeatX_.size() == NbWarpedPos &&
                                                   warpedPosFeatX_[0].rows() == modelHt * 2 &&
                                                  warpedPosFeatX_[0].cols() == modelWd * 2)));

    if(valid && validx) {
        return warpedPosFeat_.size();
    }

    FeatureParam fParam = featParam;
    fParam.padx_ = 0;
    fParam.pady_ = 0;
    int cellSz = fParam.cellSize_;
    int shift = cellSz / 2;

    if(!valid) {
        cv::Rect cropROI = bbox_.cvRect();

        int pady = ROUND((Scalar)cropROI.height / modelHt);
        int padx = ROUND((Scalar)cropROI.width  / modelWd);

        cropROI.x -= padx;
        cropROI.y -= pady;
        cropROI.width += padx * 2;
        cropROI.height += pady * 2;

        warpedPos_.resize(NbWarpedPos);

        int i = 0;
        warpedPos_[i++] = OpencvUtil::subarray(img_, cropROI, 1);

        cv::Rect roi = cropROI;
        roi.x -= shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.x += shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.y -= shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.y += shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.x -= shift;
        roi.y -= shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.x -= shift;
        roi.y += shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.x += shift;
        roi.y -= shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        roi = cropROI;
        roi.x += shift;
        roi.y += shift;
        warpedPos_[i++] = OpencvUtil::subarray(img_, roi, 1);

        int cellSz = fParam.cellSize_;
        cv::Size patchSz((modelWd + 2) * cellSz, (modelHt + 2) * cellSz);
        cv::Size patchSzX((modelWd * 2 + 2) * cellSz, (modelHt * 2 + 2) * cellSz);

        Cell bg;
        bg.setZero();

        warpedPosFeat_.resize(NbWarpedPos);
        if(fParam.partOctave_) {
            warpedPosFeatX_.resize(NbWarpedPos);
        }

#pragma omp parallel for
        for(int i = 0; i < NbWarpedPos; ++i) {
            Mat patch;
            Mat patchf;

            if(fParam.partOctave_) {
                cv::resize(warpedPos_[i], patch, patchSzX, 0, 0, RGM_IMG_RESIZE);
                patch.convertTo(patchf,
                                CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                            patch.channels()));

                FeaturePyr::computeFeature(patchf, fParam, bg,
                                           warpedPosFeatX_[i]);

            }

            cv::resize(warpedPos_[i], patch, patchSz, 0, 0, RGM_IMG_RESIZE);
            patch.convertTo(patchf,
                            CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                        patch.channels()));

            FeaturePyr::computeFeature(patchf, fParam, bg, warpedPosFeat_[i]);
        }

        if(fParam.partOctave_) {
            RGM_CHECK_EQ(warpedPosFeatX_[0].rows(), modelHt * 2);
            RGM_CHECK_EQ(warpedPosFeatX_[0].cols(), modelWd * 2);
        }

        RGM_CHECK_EQ(warpedPosFeat_[0].rows(), modelHt);
        RGM_CHECK_EQ(warpedPosFeat_[0].cols(), modelWd);

    } else if(!validx) {
        cv::Size patchSzX((modelWd * 2 + 2) * cellSz, (modelHt * 2 + 2) * cellSz);

        Cell bg;
        bg.setZero();

        warpedPosFeat_.resize(NbWarpedPos);
        if(fParam.partOctave_) {
            warpedPosFeatX_.resize(NbWarpedPos);
        }

#pragma omp parallel for
        for(int i = 0; i < NbWarpedPos; ++i) {
            Mat patch;
            Mat patchf;

            cv::resize(warpedPos_[i], patch, patchSzX, 0, 0, RGM_IMG_RESIZE);
            patch.convertTo(patchf,
                            CV_MAKETYPE(cv::DataDepth<Scalar>::value,
                                        patch.channels()));

            FeaturePyr::computeFeature(patchf, fParam, bg, warpedPosFeatX_[i]);
        }

        RGM_CHECK_EQ(warpedPosFeatX_[0].rows(), modelHt * 2);
        RGM_CHECK_EQ(warpedPosFeatX_[0].cols(), modelWd * 2);
    }

    return warpedPosFeat_.size();
}

template<int Dimension>
void OneFrameResult::computeROI(int modelSz, Scalar searchROI) {
    const int maxROISz = TrackerConst::maxROISz;
    int searchSz = std::min<int>(maxROISz, std::max<int>(modelSz,
                                                         std::max<int>(bbox_.width(),
                                                                       bbox_.height()))  * searchROI);
    roi_.setX(std::max<int>(0, bbox_.xcenter() - searchSz / 2));
    roi_.setY(std::max<int>(0, bbox_.ycenter() - searchSz / 2));
    roi_.setWidth(std::min<int>(roi_.x() + searchSz, img_.cols) -
                  roi_.x());
    roi_.setHeight(std::min<int>(roi_.y() + searchSz, img_.rows) -
                   roi_.y());


//    int searchWd = std::min<int>(maxROISz, std::max<int>(modelSz,
//                                                         bbox_.width())  * searchROI);
//    int searchHt = std::min<int>(maxROISz, std::max<int>(modelSz,
//                                                         bbox_.height())  * searchROI);

//    roi_.setX(std::max<int>(0, bbox_.xcenter() - searchWd / 2));
//    roi_.setY(std::max<int>(0, bbox_.ycenter() - searchHt / 2));
//    roi_.setWidth(std::min<int>(roi_.x() + searchWd, img_.cols) -
//                  roi_.x());
//    roi_.setHeight(std::min<int>(roi_.y() + searchHt, img_.rows) -
//                   roi_.y());

}

template<int Dimension>
void OneFrameResult::computeWorkingScale(int modelSz, int maxWd, int maxHt) {
    Scalar mins = static_cast<Scalar>(modelSz) /
            (2 * std::min(roi_.width(), roi_.height()));
    Scalar maxs = std::min<Scalar>(
                static_cast<Scalar>(maxWd) / (2 * roi_.width()),
                static_cast<Scalar>(maxHt) / (2 * roi_.height()));

    workingScale_ = std::min<Scalar>(maxs, std::max<Scalar>(workingScale_, mins));
    predictScale_ = workingScale_;
}

template<int Dimension>
Scalar OneFrameResult::computeCandConsistency(const Rectangle *predictBox) {

    consistency_ = 0;

    //    if ( pts_.size() < 2 ) return 1.0F;

    //    const ParseInfo * info = pts_[0].rootParseInfo();
    //    Rectangle_<Scalar> ref(info->x(), info->y(), info->width(), info->height());
    //    Intersector_<Scalar> inter(ref, 0.5F, true);

    //    float ov;
    //    for ( int i = 1; i < pts_.size(); ++i ) {
    //        const ParseInfo * infoi = pts_[i].rootParseInfo();
    //        Rectangle_<Scalar> box(infoi->x(), infoi->y(), infoi->width(), infoi->height());
    //        inter(box, &ov);
    //        consistency_ += ov;
    //    }



    if ( pts_.size() < 2 ) return 1.0F;

    // compute the bounding hull
    int x1 = 10000;
    int y1 = 10000;
    int x2 = 0;
    int y2 = 0;

    for ( int i = 0; i < pts_.size(); ++i ) {
        const ParseInfo * info = pts_[i].rootParseInfo();
        x1 = std::min<int>(ROUND(info->x()), x1);
        y1 = std::min<int>(ROUND(info->y()), y1);
        x2 = std::max<int>(ROUND(info->right()), x2);
        y2 = std::max<int>(ROUND(info->bottom()), y2);
    }

    Rectangle hull(x1, y1, x2-x1+1, y2-y1+1);
    Intersector_<int> inter(hull, 0.5F, true);

    Scalar ov;

    for ( int i = 0; i < pts_.size(); ++i ) {
        const ParseInfo * info = pts_[i].rootParseInfo();
        Rectangle box(ROUND(info->x()), ROUND(info->y()),
                      ROUND(info->width()), ROUND(info->height()));

        //        if ( !inter(box, &ov) ) {
        //            consistency_ = 0;
        //            return 0.0F;
        //        }

        inter(box, &ov);

        consistency_ += ov;
    }

    consistency_ /= (pts_.size()); // 1 + pts_.size()

    ov = 0;
    if ( predictBox != NULL ) {
        inter(*predictBox, &ov);
    }

    return ov;
}

template<int Dimension>
void OneFrameResult::blurGrayImg() {
//    // blur
//    Scalar sigma = 2.0F;
//    cv::Size ksz(6*sigma + 1, 6*sigma+1);

//    Mat imgF, imgFB;
//    grayImg_.convertTo(imgF, CV_32FC1);
//    cv::GaussianBlur(imgF, imgFB, ksz, sigma, sigma);
//    imgFB.convertTo(grayImg_, CV_8UC1);
}

template<int Dimension>
void OneFrameResult::release() {
//    img_.release();
    grayImg_.release();
    if(pyr_ != NULL) {
        delete pyr_;
        pyr_ = NULL;
    }
    warpedPos_.clear();
    warpedPosFeat_.clear();
    warpedPosFeatX_.clear();

    warpedImgs_.clear();
    for(int i = 0; i < warpedPyrs_.size(); ++i) {
        if(warpedPyrs_[i] != NULL)   {
            delete warpedPyrs_[i];
            warpedPyrs_[i] = NULL;
        }
    }
}

INSTANTIATE_CLASS_(OneFrameResult_);



template<int Dimension>
void TrackerResult::clear() {
    vector<OneFrameResult>().swap(output_);
    start_ = 0;
    //    pos_.clear();
    //    neg_.clear();
}

template<int Dimension>
vector<int> TrackerResult::getFrameIdxForTrain(int count,
                                               int numFrameUsedToTrain) {
    vector<int> frameIdx;
    if(numFrameUsedToTrain == 1) {
        // use the current frame only
        RGM_CHECK(getOutput()[count].isValid_, error);
        frameIdx.push_back(count);

    } else {
        frameIdx.push_back(start_); // alway use the first frame
        int numAdded = 1;
        int reserved = ((count != start_  && getOutput()[count].isValid_) ? 1 : 0);

        int i = start_ + 1;
        for(; i < count && numAdded < TrackerConst::NbFirstFramesKept &&
            numAdded <= numFrameUsedToTrain - reserved; ++i) {
            if(getOutput()[i].isValid_) {
                frameIdx.push_back(i);
                numAdded++;
            }
        }
        for(int j = count - 1; j > i &&
            numAdded <= numFrameUsedToTrain - reserved; --j) {
            if(getOutput()[j].isValid_) {
                frameIdx.push_back(j);
                ++numAdded;
            }
        }
        if(reserved) {
            frameIdx.push_back(count);
        }
    }

    //    std::cout << "frameIdx: ";
    //    for ( int i = 0; i < frameIdx.size(); ++i ) {
    //        std::cout << frameIdx[i] << " ";
    //    }
    //    std::cout << std::endl;

    return frameIdx;
}

INSTANTIATE_CLASS_(TrackerResult_);

}
