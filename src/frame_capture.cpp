#include "frame_capture.h"
#include <stdexcept>

namespace RGM {

FCapture::~FCapture() {
    if(img_convert_ctx) {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }
    if(picture) av_free(picture);
    if(video_st) {
        avcodec_close(video_st->codec);
        video_st = NULL;
    }
    if(ic) {
        avformat_close_input(&ic);
        ic = NULL;
    }
    if(rgb_picture.data[0]) {
        free(rgb_picture.data[0]);
        rgb_picture.data[0] = 0;
    }
    // free last packet if exist
    if(packet.data) {
        av_free_packet(&packet);
        packet.data = NULL;
    }
}

FCapture::FCapture(const char* filename) {
    av_register_all();
    unsigned i;
    bool valid = false;
    ic = 0;
    avcodec = NULL;
    video_stream = -1;
    video_st = NULL;
    picture = NULL;
    picture_pts = 0;
    eps_zero = 0.000025;
    img_convert_ctx = NULL;
    frame_number = 0;
    first_frame_number = -1;
    rgb_picture.data[0] = 0;
    memset(&packet, 0, sizeof(packet));
    av_init_packet(&packet);
    // Opens an input stream and reads the header
    int err = avformat_open_input(&ic, filename, NULL, NULL);
    if(err < 0) {
        throw std::runtime_error("Error opening file.");
    }
    // Get stream information
    err = avformat_find_stream_info(ic, NULL);
    if(err < 0) {
        throw std::runtime_error("Could not find codec parameters");
    }
    for(i = 0; i < ic->nb_streams; i++) {
        AVCodecContext *enc = ic->streams[i]->codec;
        //    AVCodecContext *enc = &ic->streams[i]->codec;
        enc->thread_count = 1;
        // May need to define AVMEDIA_TYPE_VIDEO to CODEC_TYPE_VIDEO
        if(AVMEDIA_TYPE_VIDEO == enc->codec_type && video_stream < 0) {
            int enc_width = enc->width;
            int enc_height = enc->height;
            AVCodec *codec = avcodec_find_decoder(enc->codec_id);
            if(!codec || avcodec_open2(enc, codec, NULL) < 0)
                exit(1);
            if(enc_width && (enc->width != enc_width)) {enc->width = enc_width;}
            if(enc_height && (enc->height != enc_height)) {
                enc->height = enc_height;
            }
            video_stream = i;
            video_st = ic->streams[i];
            picture = avcodec_alloc_frame();
            rgb_picture.data[0] = (uint8_t *)malloc(
                                      avpicture_get_size(PIX_FMT_BGR24,
                                                         enc->width, enc->height));
            // Fill out image parameters to rgb_picture
            avpicture_fill((AVPicture *)&rgb_picture, rgb_picture.data[0],
                           PIX_FMT_BGR24, enc->width, enc->height);
            frame.width = enc->width;
            frame.height = enc->height;
            frame.cn = 3;
            frame.step = rgb_picture.linesize[0];
            frame.data = rgb_picture.data[0];
            break;
        }
    }
    if(video_stream >= 0)
        valid = true;
    if(!valid) {
        throw std::runtime_error("Invalid stream");
    }
}

double FCapture::r2d(AVRational r) const {
    return r.num == 0 || r.den == 0 ? 0. : (double) r.num / (double)r.den;
}

double FCapture::get_fps() const {
    double fps = r2d(ic->streams[video_stream]->r_frame_rate);
    if(fps < eps_zero)
        double fps = r2d(ic->streams[video_stream]->avg_frame_rate);
    return av_q2d(ic->streams[video_stream]->r_frame_rate);
    return fps;
}

int64_t FCapture::dts_to_frame_number(int64_t dts) {
    double sec = dts_to_sec(dts);
    return (int64_t)(get_fps() * sec + 0.5);
}

double FCapture::dts_to_sec(int64_t dts) {
    return (double)(dts - ic->streams[video_stream]->start_time) *
           r2d(ic->streams[video_stream]->time_base);
}

bool FCapture::grabFrame() {
    bool valid = false;
    int got_picture;
    int count_errs = 0;
    const int max_number_of_attempts = 1 << 16;
    if(!ic || !video_st)  return false;
    if(ic->streams[video_stream]->nb_frames > 0 &&
            frame_number > ic->streams[video_stream]->nb_frames)
        return false;
    av_free_packet(&packet);
    picture_pts = AV_NOPTS_VALUE_;
    // get the next frame
    while(!valid) {
        int ret = av_read_frame(ic, &packet);
        if(ret == AVERROR(EAGAIN)) continue;
        if(packet.stream_index != video_stream) {
            av_free_packet(&packet);
            count_errs++;
            if(count_errs > max_number_of_attempts)
                break;
            continue;
        }
        // Decode video frame
        avcodec_decode_video2(video_st->codec, picture, &got_picture, &packet);

        if(got_picture) {
            if(picture_pts == AV_NOPTS_VALUE_)
                picture_pts = packet.pts != AV_NOPTS_VALUE_ && packet.pts != 0 ? packet.pts : packet.dts;
            frame_number++;
            valid = true;
        } else {
            count_errs++;
            if(count_errs > max_number_of_attempts)
                break;
        }
        av_free_packet(&packet);
    }
    if(valid && first_frame_number < 0)
        first_frame_number = dts_to_frame_number(picture_pts);
    return valid;
}

double FCapture::get_duration_sec() const {
    double sec = (double)ic->duration / (double)AV_TIME_BASE;
    if(sec < eps_zero)
        sec = (double)ic->streams[video_stream]->duration *
              r2d(ic->streams[video_stream]->time_base);
    return sec;
}

uint64_t FCapture::get_total_frames() const {
    //int64_t nbf = 0;
    int64_t nbf = ic->streams[video_stream]->nb_frames;
    if(nbf == 0)
        nbf = (int64_t)floor(get_duration_sec() * get_fps() + 0.5);
    return nbf;
}

// moves to the _frame_number'th frame
void FCapture::seek(int64_t _frame_number) {
    _frame_number = _frame_number < get_total_frames() ?
                    _frame_number : get_total_frames();

    int delta = 16;
    if(first_frame_number < 0 && get_total_frames() > 1)
        grabFrame();
    for(;;) {
        int64_t _frame_number_temp = _frame_number - delta > 0 ?
                                     _frame_number - delta : (int64_t)0;
        double sec = (double)_frame_number_temp / get_fps();
        int64_t time_stamp = ic->streams[video_stream]->start_time;
        double time_base = r2d(ic->streams[video_stream]->time_base);
        time_stamp += (int64_t)(sec / time_base + 0.5);
        if(get_total_frames() > 1)
            av_seek_frame(ic, video_stream, time_stamp, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(ic->streams[video_stream]->codec);
        if(_frame_number > 0) {
            grabFrame();
            if(_frame_number > 1) {
                frame_number = dts_to_frame_number(picture_pts) - first_frame_number;
                if(frame_number < 0 || frame_number > _frame_number - 1) {
                    if(_frame_number_temp == 0 || delta > INT_MAX / 4)
                        break;
                    delta = delta < 16 ? delta * 2 : delta * 3 / 2;
                    continue;
                }
                while(frame_number < _frame_number - 1)
                    if(!grabFrame())
                        break;
                frame_number++;
                break;
            } else {
                frame_number = 1;
                break;
            }
        } else {
            frame_number = 0;
            break;
        }
    }
    return;
}

bool FCapture::retrieveFrame() {
    if(!video_st || !picture->data[0])
        return false;

    avpicture_fill((AVPicture*)&rgb_picture, rgb_picture.data[0], PIX_FMT_RGB24,
                   video_st->codec->width, video_st->codec->height);

    if(img_convert_ctx == NULL ||
            frame.width != video_st->codec->width ||
            frame.height != video_st->codec->height) {
        if(img_convert_ctx)
            sws_freeContext(img_convert_ctx);

        frame.width = video_st->codec->width;
        frame.height = video_st->codec->height;

        img_convert_ctx = sws_getCachedContext(
                              NULL,
                              video_st->codec->width, video_st->codec->height,
                              video_st->codec->pix_fmt,
                              video_st->codec->width, video_st->codec->height,
                              PIX_FMT_BGR24,
                              SWS_BICUBIC,
                              NULL, NULL, NULL
                          );

        if(img_convert_ctx == NULL)
            return false;//CV_Error(0, "Cannot initialize the conversion context!");
    }

    sws_scale(
        img_convert_ctx,
        picture->data,
        picture->linesize,
        0, video_st->codec->height,
        rgb_picture.data,
        rgb_picture.linesize
    );

    // *data = frame.data;
    // *step = frame.step;
    // *width = frame.width;
    // *height = frame.height;
    // *cn = frame.cn;

    return true;
}



double FCapture::getProperty(int property_id) const {
    if(!video_st) return 0;

    switch(property_id) {
        case CAP_PROP_POS_MSEC:
            return 1000.0 * (double)frame_number / get_fps();
        case CAP_PROP_POS_FRAMES:
            return (double)frame_number;
        case CAP_PROP_POS_AVI_RATIO:
            return r2d(ic->streams[video_stream]->time_base);
        case CAP_PROP_FRAME_COUNT:
            return (double)get_total_frames();
        case CAP_PROP_FRAME_WIDTH:
            return (double)frame.width;
        case CAP_PROP_FRAME_HEIGHT:
            return (double)frame.height;
        case CAP_PROP_FPS:
            return av_q2d(video_st->r_frame_rate);
        case CAP_PROP_FOURCC:
            return (double)video_st->codec->codec_tag;
        default:
            break;
    }

    return 0;
}

bool FCapture::setProperty(int property_id, double value) {
    if(!video_st) return false;

    switch(property_id) {
        case CAP_PROP_POS_MSEC:
        case CAP_PROP_POS_FRAMES:
        case CAP_PROP_POS_AVI_RATIO: {
            switch(property_id) {
                case CAP_PROP_POS_FRAMES:
                    seek((int64_t)value);
                    break;
                case CAP_PROP_POS_MSEC:
                    seek(value / 1000.0);
                    break;
                case CAP_PROP_POS_AVI_RATIO:
                    seek((int64_t)(value * ic->duration));
                    break;
            }
            picture_pts = (int64_t)value;
        }
        break;
        default:
            return false;
    }

    return true;
}

}
