#define TRAX

// Uncomment line below if you want to use rectangles
//#define VOT_RECTANGLE

//#define VOT_VIS

#include "vot.h"

#include "tracker.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilString.hpp"

namespace RGM_VOT {

template<int Dimension, int DimensionG>
int runAOGTracker_(string & configFile) {

    VOT vot; // Initialize the communcation

    VOTRegion region = vot.region(); // Get region and first frame
    string path = vot.frame();

    // TODO: Load the first frame and use the initialization region to initialize the tracker.
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if ( img.channels() == 1 ) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    string rootPath = RGM::FileUtil::GetParentDir(path);
    string extName = RGM::FileUtil::GetFileExtension(path);

    vector<string> allImgFiles;
    RGM::FileUtil::GetFileList(allImgFiles, rootPath, extName);
    int numFrames = allImgFiles.size();

//    std::cerr << "numFrames: " << numFrames << std::endl;

    RGM::Rectangle bbox;
#ifdef VOT_RECTANGLE
    bbox = RGM::Rectangle(ROUND(region.get_x()), ROUND(region.get_y()),
                          ROUND(region.get_width()), ROUND(region.get_height()));
#else
    int numPts = region.count();
    if(numPts < 3) {
        std::cerr << "***************** input wrong region" << std::endl;
        return -1;
    }    
// Init with an axis aligned bounding box with correct area and center coordinate
    float cx = 0, cy = 0, x1 = FLT_MAX, x2 = FLT_MIN, y1 = FLT_MAX, y2 = FLT_MIN;
    for ( int i = 0; i < numPts; ++i) {
        cx += region.get_x(i);
        cy += region.get_y(i);
        x1 = std::min<float>(x1, region.get_x(i));
        x2 = std::max<float>(x2, region.get_x(i));
        y1 = std::min<float>(y1, region.get_y(i));
        y2 = std::max<float>(y2, region.get_y(i));
    }
    cx /= numPts;
    cy /= numPts;

    float A1 = sqrt(std::pow<float>(region.get_x(0) - region.get_x(1), 2) +
                    std::pow<float>(region.get_y(0) - region.get_y(1), 2)) *
            sqrt(std::pow<float>(region.get_x(1) - region.get_x(2), 2) +
                                std::pow<float>(region.get_y(1) - region.get_y(2), 2));
    float A2 = (x2 - x1) * (y2 - y1);
    float s = sqrt(A1/A2);
    float wd = s * (x2 - x1) + 1;
    float ht = s * (y2 - y1) + 1;

    int x = ROUND(cx - wd / 2.0F);
    int y = ROUND(cy - ht / 2.0F);
    bbox = RGM::Rectangle(x, y, ROUND(wd), ROUND(ht));
//    bbox.clip(img.cols, img.rows);
#endif
//        std::cerr << bbox.x() << " " << bbox.y() << " " << bbox.width() << " " << bbox.height() << std::endl;

#ifdef VOT_VIS
    string cacheDir = rootPath + "cache" + RGM::FILESEP; ;
    int count = 1;
    while(RGM::FileUtil::exists(cacheDir)) {
        cacheDir = rootPath + "cache_" + boost::lexical_cast<string>(count++) + RGM::FILESEP;
    }
    RGM::FileUtil::VerifyDirectoryExists(cacheDir);
#else
    string cacheDir = "./";
#endif

    const string seqName("VOT");

    RGM::AOGTracker_<Dimension, DimensionG>  tracker(configFile);
    tracker.init(cacheDir, seqName, img, bbox, numFrames);
    int frameIdx = 1;
    if(!tracker.initAOG(cacheDir, frameIdx, numFrames, seqName)) {
        std::cerr << "Failed to init AOGTracker" << std::endl;
        return -1;
    }
    ++frameIdx;
#ifdef VOT_VIS
    count = 0;
    string saveName = cacheDir + RGM::NumToString_<int>(count, 5) + ".jpg";
    cv::rectangle(img, bbox.cvRect(), cv::Scalar(0, 0, 255), 2);
    cv::imwrite(saveName, img);
#endif

    int numValid = 0;
    int numIntrackable = 0;
    bool showScoreMaps = false;
    //track
    while(true) {
        path = vot.frame(); // Get the next frame
        if(path.empty()) break;  // Are we done?

        // TODO: Perform a tracking step with the image, obtain new region
        img = cv::imread(path, cv::IMREAD_COLOR);
        if ( img.channels() == 1 ) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }

        bbox = tracker.runAOGTracker(img, cacheDir, frameIdx++,
                                     numValid, numIntrackable, showScoreMaps);

#ifdef VOT_RECTANGLE
        region.set_x(bbox.x());
        region.set_y(bbox.y());
        region.set_width(bbox.width());
        region.set_height(bbox.height());
#else
        region = VOTRegion(4);

//        wd = bbox.width() / s;
//        ht = bbox.height() / s;

//        x1 = std::max<float>(bbox.xcenter() - wd / 2, 0);
//        y1 = std::max<float>(bbox.ycenter() - ht / 2, 0);
//        x2 = std::min<float>(img.cols, bbox.x() + wd) - 1;
//        y2 = std::min<float>(img.rows, bbox.y() + ht) - 1;

//        region.set(0, x1, y1);
//        region.set(1, x2, y1);
//        region.set(2, x2, y2);
//        region.set(3, x1, y2);

        region.set(0, bbox.x(),                    bbox.y());
        region.set(1, bbox.x() + bbox.width() - 1, bbox.y());
        region.set(2, bbox.x() + bbox.width() - 1, bbox.y() + bbox.height() - 1);
        region.set(3, bbox.x(),                    bbox.y() + bbox.height() - 1);

#endif

#ifdef VOT_VIS
        ++count;
        saveName = cacheDir + RGM::NumToString_<int>(count, 5) + ".jpg";
        cv::rectangle(img, bbox.cvRect(), cv::Scalar(0, 0, 255), 2);
        cv::imwrite(saveName, img);
#endif

         vot.report(region); // Report the position of the tracker
    }

    // Finishing the communication is completed automatically with the destruction
    // of the communication object (if you are using pointers you have to explicitly
    // delete the object).

    return 0;
}

}

int main(int argc, char* argv[]) {

    RGM::log_init();

    if(argc < 2) {
        std::cerr << "Usage: AOGTracker4VOT path_to_config_xml_file" << std::endl;
        return -1;
    }

    string configFile(argv[1]);            
    CvFileStorage * fs = cvOpenFileStorage(configFile.c_str(), 0,
                                               CV_STORAGE_READ);
    if ( fs == NULL ) {
        std::cerr << "Not found " << configFile << std::endl ; fflush(stdout);
        return -1;
    }

    int sFeatType = cvReadIntByName(fs, 0, "FeatType", 4);
    int gFeatType = cvReadIntByName(fs, 0, "GFeatType", 0);

    cvReleaseFileStorage(&fs);


    // get feature type and dimension
    const int sDimension = RGM::FeatureDim[sFeatType];
    const int gDimension = RGM::FeatureDim[gFeatType];

    int code = -1;
    switch(sDimension) {
        case 22: {
            switch(gDimension) {
                case 32: {
                    code = RGM_VOT::runAOGTracker_<22, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<22, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<22, 48>(configFile);
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
                    code = RGM_VOT::runAOGTracker_<28, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<28, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<28, 48>(configFile);
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
                    code = RGM_VOT::runAOGTracker_<32, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<32, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<32, 48>(configFile);
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
                    code = RGM_VOT::runAOGTracker_<42, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<42, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<42, 48>(configFile);
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
                    code = RGM_VOT::runAOGTracker_<38, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<38, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<38, 48>(configFile);
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
                    code = RGM_VOT::runAOGTracker_<48, 32>(configFile);
                    break;
                }
                case 42: {
                    code = RGM_VOT::runAOGTracker_<48, 42>(configFile);
                    break;
                }
                case 48: {
                    code = RGM_VOT::runAOGTracker_<48, 48>(configFile);
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

    return code;
}
