#include "tracker_data.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilString.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilOpencv.hpp"

namespace RGM {

#define RGM_USE_RECT

TrackerData::TrackerData() :
    curSeqIdx_(0), curFrameIdx_(0), clearPreviousResults_(false) {

}

bool TrackerData::readConfig(const string &configFilename) {
    CvFileStorage * fs = cvOpenFileStorage(configFilename.c_str(), 0,
                                           CV_STORAGE_READ);
    if ( fs == NULL ) return false;

    rootDir_ = cvReadStringByName(fs, 0, "RootDir", "");
    if (!FileUtil::exists(rootDir_) ) {
        std::cerr << "Not found RootDir" << std::endl ; fflush(stdout);
        return false;
    }
    FileUtil::VerifyTheLastFileSep(rootDir_);

    string seqNames = cvReadStringByName(fs, 0, "SequenceNames", "");
    vector<string> allSeqNames;
    if(seqNames.empty()) {
        FileUtil::GetSubFolderList(allSeqNames, rootDir_);
        std::sort(allSeqNames.begin(), allSeqNames.end());
    } else {
        boost::split(allSeqNames, seqNames, boost::is_any_of(","));
    }
    RGM_CHECK_GT(allSeqNames.size(), 0);

    imgFolder_ = cvReadStringByName(fs, 0, "ImageFolder", "");
    imgExt_ = cvReadStringByName(fs, 0, "ImageExtName", ".jpg");

    string strStartEndFrameIdxSpec = cvReadStringByName(fs, 0, "StartEndFrameIdxSpec",
                                                        "David 300 770 Football1 1 74 Freeman3 1 460 Freeman4 1 283");
    std::map<string, pair<int, int> > startEndFrameIdxSpec;
    if(!strStartEndFrameIdxSpec.empty()) {
        vector<string> tmp;
        boost::split(tmp, strStartEndFrameIdxSpec, boost::is_any_of(" "));
        RGM_CHECK_EQ(tmp.size() % 3, 0);
        for(int i = 0; i < tmp.size(); i += 3) {
            RGM_CHECK_EQ(startEndFrameIdxSpec.find(tmp[i]), startEndFrameIdxSpec.end());
            int startIdx = boost::lexical_cast<int>(tmp[i + 1]);
            int endIdx = boost::lexical_cast<int>(tmp[i + 2]);
            startEndFrameIdxSpec.insert(std::make_pair(tmp[i], std::make_pair(startIdx, endIdx)));
        }
    }

    string gtFileBaseName = cvReadStringByName(fs, 0, "GroundTruthFileBaseName", "");
    if ( gtFileBaseName.empty() ) {
        std::cerr << "Not specify GroundTruthFileBaseName" << std::endl ; fflush(stdout);
        return false;
    }

    clearPreviousResults_ = cvReadIntByName(fs, 0, "clearPreviousResults", 0);
    note_ = cvReadStringByName(fs, 0, "note", "test");

    curSeqIdx_   = 0;

    // check the sequences
    source_.clear();
    for(int i = 0; i < allSeqNames.size(); ++i) {
        string seq = allSeqNames[i] + FILESEP;

        Source src;
        src.name_ = allSeqNames[i];

        string imgPath = rootDir_ + seq + imgFolder_ + FILESEP;
        vector<string> imgFiles;
        FileUtil::GetFileList(imgFiles, imgPath, imgExt_);
        if(imgFiles.size() == 0) {
            seq += seq;
            imgPath = rootDir_ + seq + imgFolder_ + FILESEP;
            FileUtil::GetFileList(imgFiles, imgPath, imgExt_);

            if(imgFiles.size() == 0) {
                RGM_LOG(warning, "Not found images in " + imgPath);
                continue;
            }
        }

        std::sort(imgFiles.begin(), imgFiles.end());

        int len = imgFiles[0].length() - imgExt_.length();

        string baseName = imgFiles[0].substr(0, len);
        int imgStart = boost::lexical_cast<int>(baseName);
        baseName = imgFiles.back().substr(0, len);
        int imgEnd = boost::lexical_cast<int>(baseName);

        src.imgNameFormat_ = "%0" + NumToString_<int>(len) + "d" + imgExt_;
        src.imgFileFormat_ = boost::format(imgPath + src.imgNameFormat_);

        // check if there are specifications
        int specStartIdx = -1;
        int specEndIdx = -1;
        if(startEndFrameIdxSpec.find(src.name_) !=
                startEndFrameIdxSpec.end()) {
            specStartIdx = startEndFrameIdxSpec[src.name_].first;
            specEndIdx = startEndFrameIdxSpec[src.name_].second;
        }

        // get frame idx
        src.startFrameIdx_ = std::max<int>(specStartIdx, imgStart);
        src.endFrameIdx_ = (specEndIdx == -1 ? imgEnd : std::min<int>(imgEnd, specEndIdx)) + 1;
        src.startFrameIdx0_ = src.startFrameIdx_;
        if(src.endFrameIdx_ - src.startFrameIdx_ < 2) {
            RGM_LOG(warning, "not enough images for " + seq);
            continue;
        }

        // check gt file
        src.gtFile_ = rootDir_ + seq + gtFileBaseName + ".txt";
        if(FileUtil::exists(src.gtFile_)) {
            src.resultFilename_ = src.name_ + "_AOGTracker_Result";
            src.instanceIdx_ = 0;

            // read gt
            vector<Rectangle > gtBboxes;
            readGroundTruthBboxes(src.gtFile_, gtBboxes, false);
            if(gtBboxes.size() == 0) {
                RGM_LOG(warning, "Not found ground-truth " + src.name_);
                continue;
            }

            src.inputBbox_ = gtBboxes[0];

            source_.push_back(src);
        }

        boost::format gtFrmt(rootDir_ + seq + gtFileBaseName + ".%d.txt");
        boost::format resultFrmt(src.name_ + "_AOGTracker_Result.%d");
        for(int j = 1; j < 10; ++j) {
            gtFrmt % j;
            src.gtFile_ = gtFrmt.str();
            if(FileUtil::exists(src.gtFile_)) {
                resultFrmt % j;
                src.resultFilename_ = resultFrmt.str();
                src.instanceIdx_ = j;

                // read gt
                vector<Rectangle > gtBboxes;
                readGroundTruthBboxes(src.gtFile_, gtBboxes, false);
                if(gtBboxes.size() == 0) {
                    RGM_LOG(warning, "Not found ground-truth " + src.name_);
                    continue;
                }

                src.inputBbox_ = gtBboxes[0];

                source_.push_back(src);
            }
        }
    }

    sourceOrig_ = source_;

    numSplit_ = cvReadIntByName(fs, 0, "numSplit", 20);
    omitFrameIdxSpecDir_ = cvReadStringByName(fs, 0, "omitFrameIdxSpecDir", "");

    shiftTypes_.clear();
    string strShiftTypes = cvReadStringByName(fs, 0, "shiftTypes",
                                              "left,right,up,down,topLeft,topRight,bottomLeft,bottomRight,scale_8,scale_9,scale_11,scale_12");
    boost::split(shiftTypes_, strShiftTypes, boost::is_any_of(","));

    cvReleaseFileStorage(&fs);

    return true;
}

void TrackerData::getAllSource() {
    getSourceForOPE();
    vector<Source> srcOPE = source_;

    getSourceForTRE();
    vector<Source> srcTRE = source_;

    getSourceForSRE();
    vector<Source> srcSRE = source_;

    source_.resize(srcOPE.size() + srcTRE.size() + srcSRE.size());

    std::copy(srcOPE.begin(), srcOPE.end(), source_.begin());
    std::copy(srcTRE.begin(), srcTRE.end(), source_.begin() + srcOPE.size());
    std::copy(srcSRE.begin(), srcSRE.end(), source_.begin() + srcOPE.size() + srcTRE.size());
}

void TrackerData::getSourceForOPE() {
    source_ = sourceOrig_;
    for(int i = 0; i < source_.size(); ++i) {
        source_[i].protocol_ = "OPE";
        source_[i].resultFilename_ += "_" + source_[i].protocol_ +
                "_" + NumToString_<int>(source_[i].startFrameIdx_) +
                "_" + NumToString_<int>(source_[i].endFrameIdx_);
    }
}

void TrackerData::getSourceForTRE() {
    // split source
    source_.clear();

    const int minNum = 20;

    FileUtil::VerifyTheLastFileSep(omitFrameIdxSpecDir_);

    for(int i = 0; i < sourceOrig_.size(); ++i) {
        Source src = sourceOrig_[i];
        src.protocol_ = "TRE";
        string resultName = src.resultFilename_;
        int origStartFrameIdx = src.startFrameIdx_;

        // read gt
        vector<Rectangle> gtBboxes;
        readGroundTruthBboxes(src.gtFile_, gtBboxes);
        if(gtBboxes.size() == 0) {
            RGM_LOG(warning, "Not found ground-truth " + src.name_);
            continue;
        }

        // get frame index to be omitted due to occl / out-of-view
        vector<int> notValidIdx;
        string filename = omitFrameIdxSpecDir_ + src.name_ + ".txt";
        if(!FileUtil::exists(filename)) {
            string name1 = src.name_;
            boost::to_lower(name1);
            filename = omitFrameIdxSpecDir_ + name1 + ".txt";
            if(!FileUtil::exists(filename)) {
                filename = omitFrameIdxSpecDir_ + name1 + "-" +
                        boost::lexical_cast<string>(src.instanceIdx_) + ".txt";
            }
        }
        std::ifstream ifs(filename.c_str(), std::ios::in);
        string line;
        int t1, t2;
        if(ifs.is_open()) {
            while(ifs) {
                std::getline(ifs, line);
                sscanf(line.c_str(), "%d %d", &t1, &t2);
                for(int j = t1; j <= t2; ++j) notValidIdx.push_back(j);
            }
            ifs.close();

            std::sort(notValidIdx.begin(), notValidIdx.end());
        } /*else {
            std::cerr << "No omit spec for " << src.name_ << std::endl;
        }*/

        for(int j = 0; j < gtBboxes.size(); ++j) {
            if(gtBboxes[j].x() <= 0 || gtBboxes[j].y() <= 0 ||
                    gtBboxes[j].width() <= 0 || gtBboxes[j].height() <= 0) {
                notValidIdx.push_back(src.startFrameIdx_ + j);
            }
        }

        uniqueVector_<int>(notValidIdx);

        // get valid index
        vector<int> allIdx, validIdx;
        for(int j = src.startFrameIdx_; j <= src.endFrameIdx_; ++j)
            allIdx.push_back(j);

        if(notValidIdx.size() > 0) {
            std::set_symmetric_difference(allIdx.begin(), allIdx.end(),
                                          notValidIdx.begin(), notValidIdx.end(),
                                          std::back_inserter(validIdx));
        } else {
            validIdx = allIdx;
        }

        if(validIdx.size() == 0) continue;

        int endSegIdx = -1;
        for(int j = validIdx.size() - 1; j >= 0; --j) {
            if(src.endFrameIdx_ - validIdx[j] + 1 >= minNum) {
                endSegIdx = j;
                break;
            }
        }

        // get sampled start idx
        vector<int> startIdx;
        float interval = float(endSegIdx + 1) / (numSplit_ - 1);
        if(interval < 2) {
            startIdx.push_back(validIdx[0]);
        } else {
            for(float j = 0; j < endSegIdx; j += interval) {
                startIdx.push_back(validIdx[floor(j)]);
            }
            startIdx.push_back(validIdx[endSegIdx]);
        }

        for(int j = 0; j < startIdx.size(); ++j) {
            src.startFrameIdx_ = startIdx[j];
            if(src.startFrameIdx_ == 1) continue;    // OPE covered this
            src.inputBbox_ = gtBboxes[startIdx[j] - origStartFrameIdx];

            src.resultFilename_ = resultName + "_" + src.protocol_ +
                    "_" + NumToString_<int>(src.startFrameIdx_) +
                    "_" + NumToString_<int>(src.endFrameIdx_);;

            source_.push_back(src);
        }
    }

}

void TrackerData::getSourceForSRE() {
    // shift the input bbox
    source_.clear();

    float shiftRatio = 0.1F;

    for(int i = 0; i < sourceOrig_.size(); ++i) {
        Source src = sourceOrig_[i];
        src.protocol_ = "SRE";
        string resultName = src.resultFilename_;

        src.imgFileFormat_ % src.startFrameIdx_;
        Mat img = cv::imread(src.imgFileFormat_.str());
        int imgWd = img.cols;
        int imgHt = img.rows;

        int x = src.inputBbox_.x();
        int y = src.inputBbox_.y();
        int wd = src.inputBbox_.width();
        int ht = src.inputBbox_.height();
        int brx = src.inputBbox_.right();
        int bry = src.inputBbox_.bottom();
        float ctx = x + wd / 2.0F;
        float cty = y + ht / 2.0F;

        int xx, yy, wd1, ht1, brx1, bry1;
        float ratio, w, h;
        Rectangle bbox;

        for(int j = 0; j < shiftTypes_.size(); ++j) {
            src.shiftType_ = shiftTypes_[j];
            src.resultFilename_ = resultName + "_" + src.protocol_ + "_" + src.shiftType_ +
                    "_" + NumToString_<int>(src.startFrameIdx_) +
                    "_" + NumToString_<int>(src.endFrameIdx_);

            if(std::strcmp(shiftTypes_[j].c_str(), "left") == 0) {
                xx = x - ceil(shiftRatio * wd);
                bbox = Rectangle(xx, y, wd, ht);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "right") == 0) {
                xx = x + ceil(shiftRatio * wd);
                bbox = Rectangle(xx, y, wd, ht);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "up") == 0) {
                yy = y - ceil(shiftRatio * ht);
                bbox = Rectangle(x, yy, wd, ht);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "down") == 0) {
                yy = y + ceil(shiftRatio * ht);
                bbox = Rectangle(x, yy, wd, ht);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "topLeft") == 0) {
                xx = ROUND(x - shiftRatio * wd);
                yy = ROUND(y - shiftRatio * ht);
                wd1 = brx - xx;
                ht1 = bry - yy;
                bbox = Rectangle(xx, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "topRight") == 0) {
                brx1 = ROUND(brx + shiftRatio * wd);
                yy = ROUND(y - shiftRatio * ht);
                wd1 = brx1 - x;
                ht1 = bry - yy;
                bbox = Rectangle(x, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "bottomLeft") == 0) {
                xx = ROUND(x - shiftRatio * wd);
                bry1 = ROUND(bry + shiftRatio * ht);
                wd1 = brx - xx;
                ht1 = bry1 - y + 1;
                bbox = Rectangle(xx, y, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "bottomRight") == 0) {
                brx1 = ROUND(brx + shiftRatio * wd);
                bry1 = ROUND(bry + shiftRatio * ht);
                wd1 = brx1 - x;
                ht1 = bry1 - y;
                bbox = Rectangle(x, y, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "scale_8") == 0) {
                ratio = 0.8F;
                w = ratio * wd;
                h = ratio * ht;
                xx = ROUND(ctx - w / 2.0F);
                yy = ROUND(cty - h / 2.0F);
                wd1 = ROUND(w);
                ht1 = ROUND(h);
                bbox = Rectangle(xx, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "scale_9") == 0) {
                ratio = 0.9F;
                w = ratio * wd;
                h = ratio * ht;
                xx = ROUND(ctx - w / 2.0F);
                yy = ROUND(cty - h / 2.0F);
                wd1 = ROUND(w);
                ht1 = ROUND(h);
                bbox = Rectangle(xx, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "scale_11") == 0) {
                ratio = 1.1F;
                w = ratio * wd;
                h = ratio * ht;
                xx = ROUND(ctx - w / 2.0F);
                yy = ROUND(cty - h / 2.0F);
                wd1 = ROUND(w);
                ht1 = ROUND(h);
                bbox = Rectangle(xx, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else if(std::strcmp(shiftTypes_[j].c_str(), "scale_12") == 0) {
                ratio = 1.2F;
                w = ratio * wd;
                h = ratio * ht;
                xx = ROUND(ctx - w / 2.0F);
                yy = ROUND(cty - h / 2.0F);
                wd1 = ROUND(w);
                ht1 = ROUND(h);
                bbox = Rectangle(xx, yy, wd1, ht1);
                if(bbox.clip(imgWd, imgHt)) {
                    src.inputBbox_ = bbox;
                    source_.push_back(src);
                }
            } else {
                RGM_LOG(warning, "wrong shift type " + shiftTypes_[j]);
            }
        }
    }
}

bool TrackerData::setCurSeqIdx(int idx, bool delOldResults) {
    curSeqIdx_ = idx;
    RGM_CHECK_GE(curSeqIdx_, 0);
    RGM_CHECK_LT(curSeqIdx_, source_.size());

    Source &src(source_[curSeqIdx_]);

    curFrameIdx_ = src.startFrameIdx_;

    // output
    cacheDir_ = rootDir_ + src.name_ + FILESEP + note_ + "_cache_" +
            NumToString_<int>(src.instanceIdx_);
    if(!src.protocol_.empty()) cacheDir_ += "_" + src.protocol_;
    if(!src.shiftType_.empty()) cacheDir_ += "_" + src.shiftType_;
    cacheDir_ += "_" + NumToString_<int>(src.startFrameIdx_) + "_" +
            NumToString_<int>(src.endFrameIdx_) + "_" +  FILESEP;

    if(clearPreviousResults_ && delOldResults)
        boost::filesystem::remove_all(boost::filesystem::path(cacheDir_));

    FileUtil::VerifyDirectoryExists(cacheDir_);

    // check if done already
    string resultName = cacheDir_ + src.resultFilename_ + ".txt";
    if(FileUtil::exists(resultName)) {
        RGM_LOG(normal, src.name_ + " done alreay!");
        return false;
    }
    outputImgFileFormat_ = boost::format(cacheDir_ + FILESEP + src.imgNameFormat_);

    return true;
}

int TrackerData::numFrames() {
    Source &src(source_[curSeqIdx_]);
    int num = src.endFrameIdx_ - src.startFrameIdx_;
    RGM_CHECK_GT(num, 1);

    return num;
}

string TrackerData::outputImgFilename(int frameIdx) {
    outputImgFileFormat_ % frameIdx;
    return outputImgFileFormat_.str();
}

//void TrackerData::readGroundTruthBboxes() {
//    readGroundTruthBboxes(gtFile(), gtFormat_, gtBboxes_);
//    RGM_CHECK_GE(gtBboxes_.size(), numFrames());
//}

void TrackerData::readGroundTruthBboxes(const string &gtFile,
                                        vector<Rectangle > &gtBboxes,
                                        bool readAll) {
    DEFINE_RGM_LOGGER;

    gtBboxes.clear();

    std::ifstream ifs(gtFile.c_str(), std::ios::in);
    if(!ifs.is_open()) {
        RGM_LOG(warning,  "can not read " + gtFile);
        return;
    }

    string line;
    while(ifs) {
        std::getline(ifs, line);
        if(!line.empty()) break;
    }
    if(line.empty()) return;

    std::size_t pos = line.find("\r");
    if(pos != string::npos) {
        line = line.substr(0, pos);
    }

    // identify the format
    bool isReal = (line.find(".") != string::npos);
    string token;
    if(line.find(",") != string::npos) {
        token = ",";
    } else if(line.find("\t") != string::npos) {
        token = "\t";
    } else {
        token = " ";
    }

    vector<string> strXYs;
    boost::split(strXYs, line, boost::is_any_of(token));

    bool isRect = (strXYs.size() == 4);

    // get the first annotation
    vector<Scalar> xys(strXYs.size());
    for(int i = 0; i < strXYs.size(); ++i) {
        xys[i] = boost::lexical_cast<Scalar>(strXYs[i]);
    }

    if(isRect) {
        gtBboxes.push_back(Rectangle(ROUND(xys[0]), ROUND(xys[1]), ROUND(xys[2]), ROUND(xys[3])));
    } else {
#ifndef RGM_USE_RECT
        int count = xys.size() / 2;
        assert(count >= 3);

        Scalar x = 0, y = 0, cx = 0, cy = 0;
        Scalar x1 = FLT_MAX, x2 = FLT_MIN, y1 = FLT_MAX, y2 = FLT_MIN;
        for(int i = 0; i < xys.size(); i += 2) {
            x = xys[i];
            y = xys[i + 1];
            cx += x;
            cy += y;
            x1 = std::min<Scalar>(x1, x);
            x2 = std::max<Scalar>(x2, x);
            y1 = std::min<Scalar>(y1, y);
            y2 = std::max<Scalar>(y2, y);
        }
        cx /= count;
        cy /= count;

        Scalar A1 = sqrt(std::pow<Scalar>(xys[0] - xys[2], 2) +
                std::pow<Scalar>(xys[1] - xys[3], 2)) *
                sqrt(std::pow<Scalar>(xys[2] - xys[4], 2) +
                std::pow<Scalar>(xys[3] - xys[5], 2)) ;
        Scalar A2 = (x2 - x1) * (y2 - y1);
        Scalar s = sqrt(A1 / A2);
        Scalar wd = s * (x2 - x1) + 1;
        Scalar ht = s * (y2 - y1) + 1;

        Scalar ltx = (cx - wd / 2.0F);
        Scalar lty = (cy - ht / 2.0F);
        gtBboxes.push_back(Rectangle(ROUND(ltx), ROUND(lty), ROUND(wd), ROUND(ht)));
#else
        Scalar x1 = FLT_MAX, x2 = FLT_MIN, y1 = FLT_MAX, y2 = FLT_MIN;
        for(int i = 0; i < xys.size(); i += 2) {
            x1 = std::min<Scalar>(x1, xys[i]);
            x2 = std::max<Scalar>(x2, xys[i]);
            y1 = std::min<Scalar>(y1, xys[i + 1]);
            y2 = std::max<Scalar>(y2, xys[i + 1]);
        }
        gtBboxes.push_back(Rectangle(ROUND(x1), ROUND(y1), ROUND(x2 - x1), ROUND(y2 - y1)));
#endif

    }
    if(!readAll) {
        ifs.close();
        return;
    }

    while(ifs) {
        std::getline(ifs, line);
        if(line.empty()) continue;

        pos = line.find("\r");
        if(pos != string::npos) {
            line = line.substr(0, pos);
        }

        boost::split(strXYs, line, boost::is_any_of(token));
        if(strXYs.size() == 1) {
            if(token.compare("\t") == 0)
                boost::split(strXYs, line, boost::is_any_of(" "));
            else
                boost::split(strXYs, line, boost::is_any_of("\t"));
        }
        for(int i = 0; i < strXYs.size(); ++i) {
            xys[i] = boost::lexical_cast<Scalar>(strXYs[i]);
        }

        if(isRect) {
            gtBboxes.push_back(Rectangle(ROUND(xys[0]), ROUND(xys[1]), ROUND(xys[2]), ROUND(xys[3])));
        } else {
#ifndef RGM_USE_RECT
            int count = xys.size() / 2;
            assert(count >= 3);

            Scalar x = 0, y = 0, cx = 0, cy = 0;
            Scalar x1 = FLT_MAX, x2 = FLT_MIN, y1 = FLT_MAX, y2 = FLT_MIN;
            for(int i = 0; i < xys.size(); i += 2) {
                x = xys[i];
                y = xys[i + 1];
                cx += x;
                cy += y;
                x1 = std::min<Scalar>(x1, x);
                x2 = std::max<Scalar>(x2, x);
                y1 = std::min<Scalar>(y1, y);
                y2 = std::max<Scalar>(y2, y);
            }
            cx /= count;
            cy /= count;

            Scalar A1 = sqrt(std::pow<Scalar>(xys[0] - xys[2], 2) +
                    std::pow<Scalar>(xys[1] - xys[3], 2)) *
                    sqrt(std::pow<Scalar>(xys[2] - xys[4], 2) +
                    std::pow<Scalar>(xys[3] - xys[5], 2)) ;
            Scalar A2 = (x2 - x1) * (y2 - y1);
            Scalar s = sqrt(A1 / A2);
            Scalar wd = s * (x2 - x1) + 1;
            Scalar ht = s * (y2 - y1) + 1;

            Scalar ltx = (cx - wd / 2.0F);
            Scalar lty = (cy - ht / 2.0F);
            gtBboxes.push_back(Rectangle(ROUND(ltx), ROUND(lty), ROUND(wd), ROUND(ht)));
#else
        Scalar x1 = FLT_MAX, x2 = FLT_MIN, y1 = FLT_MAX, y2 = FLT_MIN;
        for(int i = 0; i < xys.size(); i += 2) {
            x1 = std::min<Scalar>(x1, xys[i]);
            x2 = std::max<Scalar>(x2, xys[i]);
            y1 = std::min<Scalar>(y1, xys[i + 1]);
            y2 = std::max<Scalar>(y2, xys[i + 1]);
        }
        gtBboxes.push_back(Rectangle(ROUND(x1), ROUND(y1), ROUND(x2 - x1), ROUND(y2 - y1)));
#endif
        }
    }

    ifs.close();
}

Mat TrackerData::getStartFrameImg() {
    Source &src(source_[curSeqIdx_]);
    curFrameIdx_ = src.startFrameIdx_;
    return getFrameImg(curFrameIdx_);
}

Mat TrackerData::getNextFrameImg() {
    return getFrameImg(++curFrameIdx_);
}

Mat TrackerData::getFrameImg(int frameIdx) {
    Source &src(source_[curSeqIdx_]);

    Mat img;
    if(frameIdx >= src.endFrameIdx_)
        return img;

    src.imgFileFormat_ % frameIdx;

    img = cv::imread(src.imgFileFormat_.str(), cv::IMREAD_COLOR);
    if(img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        //        std::cout << "Not a color image" << std::endl ; fflush(stdout);
    }
    //    RGM_CHECK(!img.empty(), error);

    // blur
    //    Scalar sigma = 2.0F;
    //    cv::Size ksz(6*sigma + 1, 6*sigma+1);

    //    Mat imgF;
    //    img.convertTo(imgF, CV_32FC3);
    //    cv::GaussianBlur(imgF, imgF, ksz, sigma, sigma);
    //    imgF.convertTo(img, CV_8UC3);

    return img;
}

void TrackerData::visualizeInput(std::string &saveDir) {
    if(!FileUtil::exists(saveDir)) return;

    FileUtil::VerifyTheLastFileSep(saveDir);

    int wd = 100;

    for(int i = 0; i < source_.size(); ++i) {
        Source &src(source_[i]);

        src.imgFileFormat_ % src.startFrameIdx_;
        cv::Mat img = cv::imread(src.imgFileFormat_.str(), cv::IMREAD_COLOR);

        cv::rectangle(img, src.inputBbox_.cvRect(), cv::Scalar(0, 0, 255), 5);
        cv::Mat small;
        cv::resize(img, small, cv::Size(wd, float(img.rows)/img.cols * wd), 0, 0, RGM_IMG_RESIZE);

        cv::putText(small,
                    string("#" + NumToString_<int>(src.startFrameIdx_)),
                    cv::Point(3, 10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                    cv::Scalar(0, 0, 255), 1, 16); //cv::LINE_AA

        string saveName = saveDir + src.resultFilename_ + ".png";
        cv::imwrite(saveName, small);
    }

}

void TrackerData::visDataset(std::string &saveDir) {
    if(!FileUtil::exists(saveDir)) return;

    FileUtil::VerifyTheLastFileSep(saveDir);

    int wd = 200;

    for(int i = 0; i < source_.size(); ++i) {

        Source &src(source_[i]);

        src.imgFileFormat_ % src.startFrameIdx_;
        cv::Mat img = cv::imread(src.imgFileFormat_.str(), cv::IMREAD_COLOR);
        cv::rectangle(img, src.inputBbox_.cvRect(), cv::Scalar(0, 0, 255), 4);

        cv::Mat small;
        cv::resize(img, small, cv::Size(wd, float(img.rows)/img.cols * wd), 0, 0, RGM_IMG_RESIZE);
        cv::putText(small,
                    src.name_,
                    cv::Point(10, 10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    cv::Scalar(0, 0, 250), 1, 16); //cv::LINE_AA

        string saveName = saveDir + src.resultFilename_ + ".png";
        cv::imwrite(saveName, small);

        vector<Rectangle> gtBboxes;
        readGroundTruthBboxes(src.gtFile_, gtBboxes);

        // get object mean image
        cv::Size dstSz(src.inputBbox_.width(), src.inputBbox_.height());
        cv::Mat objMeanImg(src.inputBbox_.height(), src.inputBbox_.width(), CV_32FC3);
        objMeanImg = cv::Scalar::all(0);

        cv::Mat tmp, tmpF;

        img(src.inputBbox_.cvRect()).convertTo(objMeanImg, CV_32FC3);

        int count = 1;
        for ( int j = src.startFrameIdx_+1;
              j < src.endFrameIdx_ && count < gtBboxes.size(); ++j, ++count ) {
            src.imgFileFormat_ % j;
            img = cv::imread(src.imgFileFormat_.str(), cv::IMREAD_COLOR);
            Rectangle &box(gtBboxes[j - src.startFrameIdx0_]);
            if ( box.width() == 0 || box.height() == 0 ) continue;
            cv::resize(OpencvUtil::subarray(img, box.cvRect(), 1), tmp, dstSz,
                       0, 0, RGM_IMG_RESIZE);
            tmp.convertTo(tmpF, CV_32FC3);

            objMeanImg += tmpF;
        }

        objMeanImg /= count;

        objMeanImg.convertTo(tmp, CV_8UC3);

        saveName = saveDir + src.resultFilename_ + "_mean.png";
        cv::imwrite(saveName, tmp);
    }

}

}
