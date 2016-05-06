#ifndef _FRAME_READER_H_
#define _FRAME_READER_H_

#include "frame_capture.h"
#include <vector>
#include <opencv2/core/core.hpp>

namespace RGM {

class FCapture;

class FrameReader {
  public:
    FrameReader() : capture_(NULL) {}
    FrameReader(const std::string& filename);
    ~FrameReader();
    void open(const std::string& filename);
    std::vector<cv::Mat> RetrieveFrames(int start_frame_id,
                                        int batch_size, int step_size = 1);
    int GetWidth() {return width_;}
    int GetHeight() {return height_;}
    double GetFPS() {return fps_;}
    int GetFrameCount() {return frame_count_;}

  protected:
    int width_;
    int height_;
    int frame_count_;
    double fps_;
    FCapture* capture_;
};

}

#endif
