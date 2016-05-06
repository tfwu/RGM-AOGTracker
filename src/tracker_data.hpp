#ifndef RGM_TRACKER_DATA_HPP_
#define RGM_TRACKER_DATA_HPP_

#include "rectangle.hpp"

namespace RGM {

struct Source {
    string name_;
    string imgNameFormat_;
    boost::format imgFileFormat_;
    string gtFile_;
    string protocol_;
    Rectangle inputBbox_;
    string shiftType_;
    int startFrameIdx_;
    int endFrameIdx_; // exclusive
    int startFrameIdx0_;
    string resultFilename_;
    int instanceIdx_; // which object
};

/// Data for Tracker
class TrackerData {
  public:

    /// Default constructor
    TrackerData();

    /// Reads config.
    bool readConfig(const string &configFilename);

    /// Returns the root dir
    const string & rootDir() const { return rootDir_; }

    const string & imgFolder() const { return imgFolder_; }

    const string & imgExt() const { return imgExt_; }

    /// Gets the number of sequences
    int numSequences() { return source_.size(); }

    /// Select a sequence
    bool setCurSeqIdx(int idx, bool delOldResults=false);

    /// Gets current frame idx
    const int & curFrameIdx() const { return curFrameIdx_; }

    /// Gets start frame idx
    int curStartFrameIdx() const { return source_[curSeqIdx_].startFrameIdx_; }

    /// Gets the current sequence name
    const string & curSequenceName() const { return source_[curSeqIdx_].name_;}

    /// Gets the number of frames
    int numFrames();

    /// Gets full path format for saving results
    boost::format & outputImgFileFormat() { return outputImgFileFormat_; }
    string  outputImgFilename(int frameIdx);

    /// Returns the note
    const string & note() const { return note_; }

    /// Gets ground-truth file for evaluation
    const string & gtFile() const { return source_[curSeqIdx_].gtFile_; }
    const Rectangle & getInputBox() { return source_[curSeqIdx_].inputBbox_; }

    int dataStartFrameIdx0() const { return source_[curSeqIdx_].startFrameIdx0_; }
    int dataStartFrameIdx() const { return source_[curSeqIdx_].startFrameIdx_; }
    int dataEndFrameIdx() const { return source_[curSeqIdx_].endFrameIdx_; }
    string dataShiftType() const { return source_[curSeqIdx_].shiftType_; }

    /// Gets cache dir
    const string & cacheDir() const { return cacheDir_; }

    /// Gets result file name
    const string & resultFileName() const { return source_[curSeqIdx_].resultFilename_;}

    // Get frame image
    Mat getStartFrameImg();
    Mat getNextFrameImg();
    Mat getFrameImg(int frameIdx);

    // TRE: start from different frames
    void getSourceForTRE();
    // OPE: default specification
    void getSourceForOPE();
    // SRE: use different shifted boxes to init tracker
    void getSourceForSRE();
    // All
    void getAllSource();

    static void readGroundTruthBboxes(const string &gtFile,
                                      vector<Rectangle> &gtBboxes,
                                      bool readAll=true);

    // visualize the first frame with input box for checking each Source
    void visualizeInput(string & saveDir);

    void visDataset(string & saveDir);

  private:
    string rootDir_;
    string imgFolder_;
    string imgExt_;
    int curSeqIdx_; // which one in source_

    vector<Source> sourceOrig_;
    vector<Source> source_;        
    int            curFrameIdx_;

    bool           clearPreviousResults_;
    string         cacheDir_;
    boost::format  outputImgFileFormat_;

    string note_; // identify the results

    int isBenchmarkMode_;    
    int numSplit_;
    string omitFrameIdxSpecDir_;
    vector<string> shiftTypes_;    

    DEFINE_RGM_LOGGER;
};

}

#endif
