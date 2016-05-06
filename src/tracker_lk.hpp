#ifndef RGM_LK_TRACKER_HPP_
#define RGM_LK_TRACKER_HPP_

#include "rectangle.hpp"
#include "util/UtilLog.hpp"


namespace RGM {

/// Lucas-Kanade optical flow tracker
class LKTracker {
  public:
    static const int SzWin    = 11;
    static const int NbLevel  = 3;
    static const int NbRowTrackPoints = 10;
    static const int NbColTrackPoints = 10;
    static const int MarginBetweenPoints = 3;

    LKTracker();

    bool runLK(Mat curImg, Rectangle & inputBbox, Mat nextImg,
               Rectangle * nextBbox = NULL);

    bool runLK1(Mat curImg, Rectangle & inputBbox, Mat nextImg,
               Rectangle * nextBbox = NULL);

    const Rectangle & predictBbox() const { return predictBbox_; }

  private:
    vector<Mat> img_; // gray
    Rectangle inputBbox_;

    vector<vector<Mat> > pyr_;
    int curIdx_;
    int nextIdx_;

    Rectangle predictBbox_;

    DEFINE_RGM_LOGGER;
};

} // namespace RGM

#endif
