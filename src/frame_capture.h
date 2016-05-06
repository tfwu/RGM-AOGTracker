#ifndef _FRAME_CAPTURE_H_
#define _FRAME_CAPTURE_H_

///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <exception>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#if defined(__APPLE__)
#define AV_NOPTS_VALUE_ ((int64_t)0x8000000000000000LL)
#else
#define AV_NOPTS_VALUE_ ((int64_t)AV_NOPTS_VALUE)
#endif

namespace RGM {


class FrameImage {
public:
    unsigned char* data;
    int step;
    int width;
    int height;
    int cn;
};

class FCapture {
public:
    FCapture(const char *filename);
    ~FCapture();
    double getProperty(int) const;
    bool setProperty(int, double);
    void seek(int64_t frame_number);
    bool retrieveFrame();
    uint64_t get_total_frames() const;
    double  get_duration_sec() const;
    double  get_fps() const;
    int64_t dts_to_frame_number(int64_t dts);
    double  dts_to_sec(int64_t dts);
    double  r2d(AVRational r) const;
    bool grabFrame();

    double eps_zero;
    AVFormatContext * ic;
    AVCodec         * avcodec;
    int               video_stream;
    AVStream        * video_st;
    AVFrame         * picture;
    AVFrame           rgb_picture;
    int64_t           picture_pts;
    AVPacket          packet;
    FrameImage        frame;
    struct SwsContext *img_convert_ctx;
    int64_t frame_number, first_frame_number;
    enum {
        CAP_PROP_POS_MSEC = 0,
        CAP_PROP_POS_FRAMES = 1,
        CAP_PROP_POS_AVI_RATIO = 2,
        CAP_PROP_FRAME_WIDTH = 3,
        CAP_PROP_FRAME_HEIGHT = 4,
        CAP_PROP_FPS = 5,
        CAP_PROP_FOURCC = 6,
        CAP_PROP_FRAME_COUNT = 7
    };
};

}

#endif
