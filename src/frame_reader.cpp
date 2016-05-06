#include "frame_reader.h"
#include <stdexcept>

namespace RGM {

FrameReader::FrameReader(const std::string& filename) {
    capture_ = new FCapture(filename.c_str());
    frame_count_ = capture_->getProperty(FCapture::CAP_PROP_FRAME_COUNT);
    fps_         = capture_->getProperty(FCapture::CAP_PROP_FPS);
    width_       = capture_->getProperty(FCapture::CAP_PROP_FRAME_WIDTH);
    height_      = capture_->getProperty(FCapture::CAP_PROP_FRAME_HEIGHT);
}

FrameReader::~FrameReader() {
    if(capture_) {
        delete capture_;
        capture_ = 0;
    }
}

void FrameReader::open(const std::string &filename) {
    capture_ = new FCapture(filename.c_str());
    frame_count_ = capture_->getProperty(FCapture::CAP_PROP_FRAME_COUNT);
    fps_         = capture_->getProperty(FCapture::CAP_PROP_FPS);
    width_       = capture_->getProperty(FCapture::CAP_PROP_FRAME_WIDTH);
    height_      = capture_->getProperty(FCapture::CAP_PROP_FRAME_HEIGHT);
}

std::vector<cv::Mat> FrameReader::RetrieveFrames(int start_frame_id,
                                                 int batch_size, int step_size) {
    if(start_frame_id >= frame_count_)
        throw std::out_of_range("Start frame id out of range.");
    std::vector<cv::Mat> frames;
    capture_->setProperty(FCapture::CAP_PROP_POS_FRAMES, start_frame_id);
    int frame_counter = 0;
    while(frames.size() < batch_size) {
        if(step_size > 50) {
            // Reset pointer when step_size is greater than 50.
            // This number is NOT got from any benchmark.
            // May need further investigation.
            if(frames.size() != 0) {
                capture_->setProperty(FCapture::CAP_PROP_POS_FRAMES,
                                      start_frame_id + frames.size()*step_size);
            }
            capture_->grabFrame();
            capture_->retrieveFrame();
        } else {
            // Use loop to skip frames.
            int current_frame_id = start_frame_id + frame_counter;
            if(current_frame_id >= frame_count_)
                break;
            capture_->grabFrame();
            if(frame_counter % step_size != 0) {
                frame_counter++;
                continue;
            }
            capture_->retrieveFrame();
            frame_counter++;
        }
        cv::Mat img(capture_->frame.height, capture_->frame.width,
                    CV_8UC3, capture_->frame.data);
        frames.push_back(img.clone());
    }
    return frames;
}

}
