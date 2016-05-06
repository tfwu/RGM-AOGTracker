#include "feature_pyr.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilSerialization.hpp"

namespace RGM {

template<int Dimension>
FeaturePyr::~FeaturePyr_() {
    levels_.clear();    
    scales_.clear();
    validLevels_.clear();
}

template<int Dimension>
FeaturePyr::FeaturePyr_(const FeaturePyr &pyr) {
    param_ = pyr.param();
    bgmu_ = pyr.bgmu();

    levels_ = pyr.levels();    
    scales_ = pyr.scales();
    validLevels_ = pyr.validLevels();

    imgHt_ = pyr.imgHt();
    imgWd_ = pyr.imgWd();
}

template<int Dimension>
FeaturePyr::FeaturePyr_(const cv::Mat & image, const FeatureParam & param) {
    param_ = param;

    bgmu_.setZero();
    if(param.useTrunc_)
        bgmu_(Dimension - 1) = 1;

    computePyramid(image);
}

template<int Dimension>
FeaturePyr::FeaturePyr_(const cv::Mat & image, const FeatureParam & param,
                        const Cell & bgmu) : bgmu_(bgmu) {
    param_ = param;
    if(!param_.useTrunc_)
        bgmu_(Dimension - 1) = 0;

    computePyramid(image);
}

template<int Dimension>
void FeaturePyr::computePyramid(const cv::Mat & image) {
    if(image.empty() || !param_.isValid()) {
        RGM_LOG(error, "Attempting to create an empty pyramid");
        return;
    }

    imgWd_ = image.cols;
    imgHt_ = image.rows;

    // Min size of a level
    const int MinLevelSz = std::max<int>(5, param_.minLevelSz_);

    // Compute the number of scales
    // such that the smallest size of the last level is MinLevelSz (=5)
    int minSz    = min(imgWd_, imgHt_);
    int levelSz  = octave() > 0 ?
                   max(MinLevelSz,
                       static_cast<int>(minSz / (cellSize() * pow(2.0F, octave()))))
                   : MinLevelSz;

    // using double to be consistent with matlab
    const int maxScale = 1 +
                         floor(log(static_cast<double>(minSz) / double(levelSz * cellSize())) /
                               log(pow(double(2.0F), double(1.0F) / interval())));

    // Cannot compute the pyramid on images too small
    if(maxScale < interval()) {
        return;
    }

    int extraInterval = extraOctave() ? interval() : 0;
    int partInterval  = partOctave() ? interval() : 0;

    int totalLevels = maxScale + extraInterval + partInterval;

    levels_.resize(totalLevels);
    scales_.resize(totalLevels);
    validLevels_.resize(totalLevels, true);

    // Convert input image to Scalar type
    Mat imgf = OpencvUtil::floatColorImg(image);

    const Scalar sc = pow(param_.scaleBase_, 1.0F / static_cast<Scalar>(interval()));

    #pragma omp parallel for
    for(int i = 0; i < interval(); ++i) {
        Scalar scale = 1.0F / pow(sc, i);
        Mat scaled;
        if(scale == 1.0) {
            scaled = imgf;
        } else {
            cv::resize(imgf, scaled, cv::Size(imgf.cols * scale + 0.5f,
                                              imgf.rows * scale + 0.5f),
                       0.0, 0.0, RGM_IMG_RESIZE);
        }

        Mat scaled1;
        if(extraOctave()) {
            // Optional (cellSize/4) x (cellSize/4) features
            cv::resize(scaled, scaled1, cv::Size(scaled.cols * 4 + 0.5f,
                                                 scaled.rows * 4 + 0.5f),
                       0.0, 0.0, RGM_IMG_RESIZE);

            computeFeature(scaled1, param_, bgmu_, levels_[i]);
            scales_[i] = scale * 4;
        }

        if(partOctave()) {
            // First octave at twice the image resolution,
            // i.e., (cellSize/2) x (cellSize/2) features
            cv::resize(scaled, scaled1, cv::Size(scaled.cols * 2 + 0.5f,
                                                 scaled.rows * 2 + 0.5f),
                       0.0, 0.0, RGM_IMG_RESIZE);

            computeFeature(scaled1, param_, bgmu_, levels_[i + extraInterval]);
            scales_[i + extraInterval] = scale * 2;
        }

        // Second octave at the original resolution,
        // i.e., cellSize x cellSize HOG features
        int ii = i + partInterval + extraInterval;
        if(ii < totalLevels) {
            computeFeature(scaled, param_, bgmu_, levels_[ii]);
            scales_[ii] = scale;
        }

        // Remaining octaves
        for(int j = i + partInterval;  j < maxScale; j += interval()) {
            ii = j + interval() + extraInterval;
            if(ii >= totalLevels)
                break;

            cv::resize(scaled, scaled, cv::Size(scaled.cols * 0.5f + 0.5f,
                                                scaled.rows * 0.5f + 0.5f),
                       0.0, 0.0, RGM_IMG_RESIZE);
            computeFeature(scaled, param_, bgmu_, levels_[ii]);
            scales_[ii] = 0.5F * scales_[j + extraInterval];
        }
    }
}

template<int Dimension>
void FeaturePyr::computeFeature(const Mat & image, const FeatureParam & param,
                                const Cell & bgmu, Level & level) {
    if(FeatureDim[static_cast<int>(param.type_)] != Dimension) {
        std::cerr << "wrong feature type" ; fflush(stdout);
        exit(0);
    }

    if(param.type_ == HOG_DPM) {
        computeHOGDPM(image, param, bgmu, level);
    } else if(param.type_ == HOG_FFLD || param.type_ == HOG_LBP ||
              param.type_ == HOG_COLOR || param.type_ == HOG_LBP_COLOR) {
        computeHOGFFLD(image, param, bgmu, level);
    } else if(param.type_ == HOG_SIMPLE || param.type_ == HOG_SIMPLE_LBP ||
              param.type_ == HOG_SIMPLE_COLOR || param.type_ == HOG_SIMPLE_LBP_COLOR) {
        computeTrackingFeat(image, param, bgmu, level);
    } else {
        std::cerr << "Wrong feature type." ; fflush(stdout);
    }
}

template<int Dimension>
void FeaturePyr::adjustScales(Scalar multi) {
    for(int i = 0; i < scales_.size(); ++i) {
        scales_[i] *= multi;
    }
}

template<int Dimension>
const vector<int> & FeaturePyr::idxValidLevels() {
    idxValidLevels_.resize(nbValidLevels());

    for(int i = 0, j = 0; i < nbLevels(); ++i) {
        if(validLevels_[i]) {
            idxValidLevels_[j++] = i;
        }
    }

    return idxValidLevels_;
}

template<int Dimension>
int FeaturePyr::nbValidLevels() const {
    int n = 0;
    n = std::accumulate(validLevels().begin(), validLevels().end(), n);
    return n;
}

template<int Dimension>
void FeaturePyr::convolve(const Level & filter,
                          vector<Matrix> & convolutions) const {
    convolutions.resize(levels_.size());

    #pragma omp parallel for
    for(int i = 0; i < levels_.size(); ++i) {
        Convolve(levels_[i], filter, convolutions[i]);
    }
}

template<int Dimension>
void FeaturePyr::Convolve(const Level & x, const Level & y, Matrix & z) {
    // Nothing to do if x is smaller than y
    if((x.rows() < y.rows()) || (x.cols() < y.cols())) {
        z = Matrix();
        return;
    }

    z = Matrix::Zero(x.rows() - y.rows() + 1, x.cols() - y.cols() + 1);

    for(int i = 0; i < z.rows(); ++i) {
        for(int j = 0; j < y.rows(); ++j) {
            const Eigen::Map<const Matrix, Aligned, Eigen::OuterStride<Dimension> >
            mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data()), z.cols(),
                 y.cols() * Dimension);
#ifndef RGM_USE_DOUBLE
            const Eigen::Map<const Eigen::RowVectorXf, Aligned>
#else
            const Eigen::Map<const Eigen::RowVectorXd, Aligned>
#endif
            mapy(reinterpret_cast<const Scalar *>(y.row(j).data()),
                 y.cols() * Dimension);

            z.row(i).noalias() += mapy * mapx.transpose();
        }
    }
}

template<int Dimension>
typename FeaturePyr::Level FeaturePyr::Flip(const Level & level,
                                            featureType t) {
    using namespace detail;
    // Symmetric filter
    FeaturePyr::Level result(level.rows(), level.cols());

    switch(t) {
        case HOG_DPM:
        case HOG_FFLD: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGSymmetry[i]);
                    }
            break;
        }
        case HOG_LBP: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGLBPSymmetry[i]);
                    }
            break;
        }
        case HOG_COLOR: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGCOLORSymmetry[i]);
                    }
            break;
        }
        case HOG_LBP_COLOR: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGLBPColorSymmetry[i]);
                    }
            break;
        }
        case HOG_SIMPLE: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGSimpleSymmetry[i]);
                    }
            break;
        }
        case HOG_SIMPLE_LBP: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGSimpleLBPSymmetry[i]);
                    }
            break;
        }
        case HOG_SIMPLE_COLOR: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGSimpleColorSymmetry[i]);
                    }
            break;
        }
        case HOG_SIMPLE_LBP_COLOR: {
            for(int y = 0; y < level.rows(); ++y)
                for(int x = 0; x < level.cols(); ++x)
                    for(int i = 0; i < Dimension; ++i) {
                        result(y, x)(i) =
                            level(y, level.cols() - 1 - x)(HOGSimpleLBPColorSymmetry[i]);
                    }
            break;
        }
        default: {
            std::cerr << "Wrong feature type." ; fflush(stdout);
            break;
        }
    }

    return result;
}

template<int Dimension>
Eigen::Map<Matrix, Aligned> FeaturePyr::Map(Level & level) {
    return Eigen::Map<Matrix, Aligned>(level.data()->data(), level.rows(),
                                       level.cols() * Dimension);
}

template<int Dimension>
Eigen::Map<dMatrix, Aligned> FeaturePyr::dMap(dLevel & level) {
    return Eigen::Map<dMatrix, Aligned>(level.data()->data(), level.rows(),
                                        level.cols() * Dimension);
}

template<int Dimension>
const Eigen::Map<const Matrix, Aligned> FeaturePyr::Map(const Level & level) {
    return Eigen::Map<const Matrix, Aligned>(level.data()->data(), level.rows(),
                                             level.cols() * Dimension);
}

template<int Dimension>
cv::Mat_<Scalar> FeaturePyr::convertToMat(const Level & level,
                                          int startDim, int endDim) {
    //startDim = max<int>(0, startDim);
    //endDim   = min<int>(Dimension, endDim);

    if(endDim - startDim <= 0) {
        std::cerr << " Attempting to access invalid dimensions" << std::endl;
        return cv::Mat_<Scalar>();
    }

    int dim[] = { level.rows(), level.cols(), endDim - startDim };
    cv::Mat_<Scalar> m(3, dim);

    for(int d = startDim, dst = 0; d < endDim; ++d, ++dst) {
        for(int r = 0; r < level.rows(); ++r) {
            for(int c = 0; c < level.cols(); ++c) {
                m(r, c, dst) = level(r, c)(d);
            }
        }
    }

    return m;
}

template<int Dimension>
cv::Mat_<Scalar> FeaturePyr::fold(const Level & level, int featType) {
    if(featType < HOG_SIMPLE) {
        return convertToMat(level, 18, 27);
    } else {
        return convertToMat(level, 0, 17);
    }
}

template<int Dimension>
void FeaturePyr::resize(const Level & in, Scalar factor, Level & out) {
    if(factor == 1.0F) {
        out = in;
        return;
    }

    out = Level::Constant(ROUND(factor * in.rows()), ROUND(factor * in.cols()),
                          Cell::Zero());

    int dim[] = {in.rows(), in.cols(), Dimension};

    cv::Mat_<Scalar> inMat(3, dim);
    for(int r = 0; r < in.rows(); ++r)
        for(int c = 0; c < in.cols(); ++c)
            for(int d = 0; d < Dimension; ++d) {
                inMat(r, c, d) = in(r, c)(d);
            }

    cv::Mat_<Scalar> xMat = OpencvUtil_<Scalar>::resize(inMat, factor,
                                                        cv::INTER_CUBIC);

    for(int r = 0; r < out.rows(); ++r)
        for(int c = 0; c < out.cols(); ++c)
            for(int d = 0; d < Dimension; ++d) {
                out(r, c)(d) = xMat(r, c, d);
            }

}

template<int Dimension>
Mat FeaturePyr::visualize(const Level & level, int bs) {
    // Make pictures of positive and negative weights
    cv::Mat_<Scalar> w = convertToMat(level, 0, 9);

    Scalar maxVal = *(std::max_element(w.begin(), w.end()));
    Scalar minVal = *(std::min_element(w.begin(), w.end()));

    Scalar scale = max<double>(maxVal, -minVal);

    Mat pos = OpencvUtil::pictureHOG(w, bs) * 255.0F / scale;

    Mat neg;
    if(minVal < 0) {
        cv::Mat_<Scalar> minusw = w * -1.0F;
        neg = OpencvUtil::pictureHOG(minusw, bs) * 255.0F / scale;
    }

    // Put pictures together and draw
    int buff = 10;
    Mat img(pos.rows + 2 * buff, pos.cols + neg.cols + 4 * buff, pos.type(),
            cv::Scalar::all(128));

    pos.copyTo(img(cv::Rect(buff, buff, pos.cols, pos.rows)));

    if(minVal < 0) {
        neg.copyTo(img(cv::Rect(pos.cols + 2 * buff, buff, neg.cols, neg.rows)));
    }

    Mat imgShow;
    cv::normalize(img, imgShow, 255, 0.0, cv::NORM_MINMAX, CV_8UC1);
    cv::String winName("HOG");
    cv::imshow(winName, imgShow);
    cv::waitKey(0);

    return imgShow;
}

template<int Dimension>
void FeaturePyr::computeHOGDPM(const cv::Mat & image,
                               const FeatureParam & param,
                               const Cell & bgmu, Level & level) {
    assert(Dimension == 32);
    assert(param.isValid());

    const int & cellSize = param.cellSize_;

    assert((cellSize == 8 || cellSize == 4 || cellSize == 2));

    // Adapted from voc-release5/features.cc
    const Scalar UU[9] = {
        1.0000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397
    };

    const Scalar VV[9] = {
        0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420
    };

    // Get all the image members
    const int width = image.cols;
    const int height = image.rows;

    // Make sure the image is big enough
    assert(width > cellSize);
    assert(height > cellSize);

    // Memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = floor(static_cast<double>(height) / cellSize + 0.5);
    blocks[1] = floor(static_cast<double>(width)  / cellSize + 0.5);
    Matrix hist = Matrix::Zero(blocks[0], blocks[1] * 18);
    Matrix norm = Matrix::Zero(blocks[0], blocks[1]);

    // Memory for HOG features
    int out[3];
    out[0] = max(blocks[0] - 2, 0);
    out[1] = max(blocks[1] - 2, 0);
    out[2] = Dimension; // 27 + 4 + 1

    level = Level::Constant(out[0], out[1], Cell::Zero());

    int visible[2];
    visible[0] = blocks[0] * cellSize;
    visible[1] = blocks[1] * cellSize;

    for(int x = 1; x < visible[1] - 1; ++x) {
        for(int y = 1; y < visible[0] - 1; ++y) {
            const int x2 = min(x, width - 2);
            const int y2 = min(y, height - 2);

            // Use the channel with the largest gradient magnitude
            Scalar magnitude = 0;
            Scalar argDx = 0;
            Scalar argDy = 0;

            const Pixel & pixelyp = image.at<Pixel>(y2 + 1, x2);
            const Pixel & pixelym = image.at<Pixel>(y2 - 1, x2);
            const Pixel & pixelxp = image.at<Pixel>(y2, x2 + 1);
            const Pixel & pixelxm = image.at<Pixel>(y2, x2 - 1);

            for(int i = 2; i >= 0; i--) {
                const Scalar dx = pixelxp[i] - pixelxm[i];
                const Scalar dy = pixelyp[i] - pixelym[i];

                Scalar tmp = dx * dx + dy * dy;

                if(tmp > magnitude) {
                    magnitude = tmp;
                    argDx = dx;
                    argDy = dy;
                }
            }

            // Snap to one of 18 orientations
            int theta = 0;
            Scalar best = 0;
            for(int i = 0; i < 9; ++i) {
                Scalar dot = UU[i] * argDx + VV[i] * argDy;

                if(dot > best) {
                    best = dot;
                    theta = i;
                } else if(-dot > best) {
                    best = -dot;
                    theta = i + 9;
                }
            }

            // Add to 4 histograms around pixel using linear interpolation
            Scalar xp = (x + Scalar(0.5)) / (Scalar)cellSize - Scalar(0.5);
            Scalar yp = (y + Scalar(0.5)) / (Scalar)cellSize - Scalar(0.5);
            int ixp = floor(xp);
            int iyp = floor(yp);
            Scalar vx0 = xp - ixp;
            Scalar vy0 = yp - iyp;
            Scalar vx1 = 1 - vx0;
            Scalar vy1 = 1 - vy0;

            magnitude = sqrt(magnitude);

            if((ixp >= 0) && (iyp >= 0)) {
                hist(iyp, ixp * 18 + theta) += vx1 * vy1 * magnitude;
            }

            if((ixp + 1 < blocks[1]) && (iyp >= 0)) {
                hist(iyp, (ixp + 1) * 18 + theta) += vx0 * vy1 * magnitude;
            }

            if((ixp >= 0) && (iyp + 1 < blocks[0])) {
                hist(iyp + 1, ixp * 18 + theta) += vx1 * vy0 * magnitude;
            }

            if((ixp + 1 < blocks[1]) && (iyp + 1 < blocks[0])) {
                hist(iyp + 1, (ixp + 1) * 18 + theta) += vx0 * vy0 * magnitude;
            }
        }
    }

    // Compute energy in each block by summing over orientations
    for(int y = 0; y < blocks[0]; ++y) {
        for(int x = 0; x < blocks[1]; ++x) {
            Scalar sumSq = 0;

            for(int i = 0; i < 9; ++i)
                sumSq += ((hist(y, x * 18 + i) + hist(y, x * 18 + i + 9)) *
                          (hist(y, x * 18 + i) + hist(y, x * 18 + i + 9)));

            norm(y, x) = sumSq;
        }
    }

    for(int y = 0; y < out[0]; ++y) {
        for(int x = 0; x < out[1]; ++x) {
            // Normalization factors
            const Scalar n0 = 1 / sqrt(norm(y + 1, x + 1) + norm(y + 1, x + 2) +
                                       norm(y + 2, x + 1) + norm(y + 2, x + 2) +
                                       EPS);
            const Scalar n1 = 1 / sqrt(norm(y    , x + 1) + norm(y    , x + 2) +
                                       norm(y + 1, x + 1) + norm(y + 1, x + 2) +
                                       EPS);
            const Scalar n2 = 1 / sqrt(norm(y + 1, x) + norm(y + 1, x + 1) +
                                       norm(y + 2, x) + norm(y + 2, x + 1) +
                                       EPS);
            const Scalar n3 = 1 / sqrt(norm(y    , x) + norm(y    , x + 1) +
                                       norm(y + 1, x) + norm(y + 1, x + 1) +
                                       EPS);

            // Contrast-sensitive features
            Scalar t0 = 0;
            Scalar t1 = 0;
            Scalar t2 = 0;
            Scalar t3 = 0;

            for(int i = 0; i < 18; ++i) {
                const Scalar sum = hist(y + 1, (x + 1) * 18 + i);
                const Scalar h0 = min(sum * n0, Scalar(0.2));
                const Scalar h1 = min(sum * n1, Scalar(0.2));
                const Scalar h2 = min(sum * n2, Scalar(0.2));
                const Scalar h3 = min(sum * n3, Scalar(0.2));
                level(y, x)(i) = (h0 + h1 + h2 + h3) / 2.0;
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }

            // Contrast-insensitive features
            for(int i = 0; i < 9; ++i) {
                const Scalar sum = hist(y + 1, (x + 1) * 18 + i) +
                                   hist(y + 1, (x + 1) * 18 + i + 9);
                const Scalar h0 = min(sum * n0, Scalar(0.2));
                const Scalar h1 = min(sum * n1, Scalar(0.2));
                const Scalar h2 = min(sum * n2, Scalar(0.2));
                const Scalar h3 = min(sum * n3, Scalar(0.2));
                level(y, x)(i + 18) = (h0 + h1 + h2 + h3) / 2.0F;
            }

            // Texture features
            level(y, x)(27) = t0 * Scalar(0.2357);
            level(y, x)(28) = t1 * Scalar(0.2357);
            level(y, x)(29) = t2 * Scalar(0.2357);
            level(y, x)(30) = t3 * Scalar(0.2357);

            // Truncation feature
            level(y, x)(31) = 0;
        }
    }

    // Add padding
    if(param.padx_ > 0 && param.pady_ > 0) {
        // add 1 to padding because feature generation deletes a 1-cell
        // wide border around the feature map
        Level tmp = Level::Constant(level.rows() + (param.pady_ + 1) * 2,
                                    level.cols() + (param.padx_ + 1) * 2, bgmu);

        tmp.block(param.pady_ + 1, param.padx_ + 1,
                  level.rows(), level.cols()) = level;

        level.swap(tmp);
    }
}

namespace detail {
struct HOGTable {
    char bins[512][512][2];
    Scalar magnitudes[512][512][2];

    // Singleton pattern
    static const HOGTable & Singleton() {
        return Singleton_;
    }

  private:
    // Singleton pattern
    HOGTable() throw() {
        for(int dy = -255; dy <= 255; ++dy) {
            for(int dx = -255; dx <= 255; ++dx) {
                // Magnitude in the range [0, 1]
                const double magnitude = sqrt(dx * dx + dy * dy) / 255.0;

                // Angle in the range [-pi, pi]
                double angle = atan2(static_cast<double>(dy),
                                     static_cast<double>(dx));

                // Convert it to the range [9.0, 27.0]
                angle = angle * (9.0 / M_PI) + 18.0;

                // Convert it to the range [0, 18)
                if(angle >= 18.0) {
                    angle -= 18.0;
                }

                // Bilinear interpolation
                const int bin0 = angle;
                const int bin1 = (bin0 < 17) ? (bin0 + 1) : 0;
                const double alpha = angle - bin0;

                bins[dy + 255][dx + 255][0] = bin0;
                bins[dy + 255][dx + 255][1] = bin1;
                magnitudes[dy + 255][dx + 255][0] = magnitude * (1.0 - alpha);
                magnitudes[dy + 255][dx + 255][1] = magnitude * alpha;
            }
        }
    }

    // Singleton pattern
    HOGTable(const HOGTable &) throw();
    void operator=(const HOGTable &) throw();

    static const HOGTable Singleton_;
}; // struct HOGTable

const HOGTable HOGTable::Singleton_;

} // namespace detail

template<int Dimension>
void FeaturePyr::computeHOGFFLD(const cv::Mat & image,
                                const FeatureParam & param,
                                const Cell & bgmu, Level & level) {

    int cellSize = param.cellSize_;
    int padx = param.padx_;
    int pady = param.pady_;

    // Get the size of image
    const int width  = image.cols;
    const int height = image.rows;
    const int depth  = image.channels();

    // Make sure the image is big enough
    if((width < cellSize) || (height < cellSize) || (depth != 3) ||
            !param.isValid()) {
        std::cerr << "Attempting to compute an empty pyramid level" << std::endl;
        fflush(stdout);
        return;
    }

    bool zeroPadx = (padx == 0);
    bool zeroPady = (pady == 0);
    if(zeroPadx) padx = 1;
    if(zeroPady) pady = 1;

    // Resize the feature matrix
    int blockHt = (height + cellSize / 2) / cellSize;
    int blockWd = (width + cellSize / 2)  / cellSize;

    level = Level::Constant(blockHt + 2 * pady, blockWd + 2 * padx,
                            Cell::Zero());

    const Scalar invCellSize = static_cast<Scalar>(1) / cellSize;
    const Scalar tmp = 0.5F;

    for(int y = 0; y < height; ++y) {
        const int yabove = max(y - 1, 0);
        const int ybelow = min(y + 1, height - 1);

        for(int x = 0; x < width; ++x) {
            const int xright = min(x + 1, width - 1);
            const int xleft = max(x - 1, 0);

            const Pixel & pixelyp = image.at<Pixel>(ybelow, x);
            const Pixel & pixelym = image.at<Pixel>(yabove, x);
            const Pixel & pixelxp = image.at<Pixel>(y, xright);
            const Pixel & pixelxm = image.at<Pixel>(y, xleft);

            // Use the channel with the largest gradient magnitude
            int maxMagnitude = 0;
            int argDx = 255;
            int argDy = 255;

            for(int i = 0; i < depth; ++i) {
                const int dx = static_cast<int>(pixelxp[i] - pixelxm[i]);
                const int dy = static_cast<int>(pixelyp[i] - pixelym[i]);

                if(dx * dx + dy * dy > maxMagnitude) {
                    maxMagnitude = dx * dx + dy * dy;
                    argDx = dx + 255;
                    argDy = dy + 255;
                }
            }

            const char bin0 = detail::HOGTable::Singleton().bins[argDy][argDx][0];
            const char bin1 = detail::HOGTable::Singleton().bins[argDy][argDx][1];
            const Scalar magnitude0 =
                detail::HOGTable::Singleton().magnitudes[argDy][argDx][0];
            const Scalar magnitude1 =
                detail::HOGTable::Singleton().magnitudes[argDy][argDx][1];

            // Bilinear interpolation
            const Scalar xp = (x + tmp) * invCellSize + padx - 0.5f;
            const Scalar yp = (y + tmp) * invCellSize + pady - 0.5f;
            const int ixp = xp;
            const int iyp = yp;
            const Scalar xp0 = xp - ixp;
            const Scalar yp0 = yp - iyp;
            const Scalar xp1 = 1 - xp0;
            const Scalar yp1 = 1 - yp0;

            level(iyp    , ixp)(bin0) += xp1 * yp1 * magnitude0;
            level(iyp    , ixp)(bin1) += xp1 * yp1 * magnitude1;
            level(iyp    , ixp + 1)(bin0) += xp0 * yp1 * magnitude0;
            level(iyp    , ixp + 1)(bin1) += xp0 * yp1 * magnitude1;
            level(iyp + 1, ixp)(bin0) += xp1 * yp0 * magnitude0;
            level(iyp + 1, ixp)(bin1) += xp1 * yp0 * magnitude1;
            level(iyp + 1, ixp + 1)(bin0) += xp0 * yp0 * magnitude0;
            level(iyp + 1, ixp + 1)(bin1) += xp0 * yp0 * magnitude1;

            // Normalize by the number of pixels
            const Scalar normalization = 2.0 / (cellSize * cellSize);

            if(param.type_ == HOG_LBP || param.type_ == HOG_LBP_COLOR) {

                // Texture (Uniform LBP) features
                const int LBP_TABLE[256] = {
                    0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9,
                    4, 9, 9, 9, 5, 9, 6, 7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
                    2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7,
                    4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8
                };

                // Use the green channel
                const Pixel & pixel = image.at<Pixel>(y, x);
                const Scalar p = pixel[1];

                // clock-wise pixels in 8 negihborhood of (x,y)
                const int lbp = (static_cast<int>(image.at<Pixel>(yabove, xleft)[1] >= p)) |
                                (static_cast<int>(image.at<Pixel>(yabove, x)[1] >= p) << 1) |
                                (static_cast<int>(image.at<Pixel>(yabove, xright)[1] >= p) << 2) |
                                (static_cast<int>(image.at<Pixel>(y,      xright)[1] >= p) << 3) |
                                (static_cast<int>(image.at<Pixel>(ybelow, xright)[1] >= p) << 4) |
                                (static_cast<int>(image.at<Pixel>(ybelow, x)[1] >= p) << 5) |
                                (static_cast<int>(image.at<Pixel>(ybelow, xleft)[1] >= p) << 6) |
                                (static_cast<int>(image.at<Pixel>(y,      xleft)[1] >= p) << 7);


                // Bilinear interpolation
                level(iyp    , ixp)(LBP_TABLE[lbp] + 31) += xp1 * yp1 * normalization;
                level(iyp    , ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp1 * normalization;
                level(iyp + 1, ixp)(LBP_TABLE[lbp] + 31) += xp1 * yp0 * normalization;
                level(iyp + 1, ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp0 * normalization;
            }

            if(param.type_ == HOG_COLOR || param.type_ == HOG_LBP_COLOR) {
                const int startBin = (param.type_ == HOG_COLOR ? 31 : 41);
                // Color features
                const Pixel & pixel = image.at<Pixel>(y, x);
                const Scalar r = pixel[2] * static_cast<Scalar>(1.0F / 255.0F);
                const Scalar g = pixel[1] * static_cast<Scalar>(1.0F / 255.0F);
                const Scalar b = pixel[0] * static_cast<Scalar>(1.0F / 255.0F);

                const Scalar minRGB = min(r, min(g, b));
                const Scalar maxRGB = max(r, max(g, b));
                const Scalar chroma = maxRGB - minRGB;

                if(chroma > 0.05F) {
                    Scalar hue = 0.0F;

                    if(r == maxRGB) {
                        hue = (g - b) / chroma;
                    } else if(g == maxRGB) {
                        hue = (b - r) / chroma + 2;
                    } else {
                        hue = (r - g) / chroma + 4;
                    }

                    if(hue < 0.0F) {
                        hue += 6.0F;
                    }

                    if(hue >= 6.0F) {
                        hue = 0.0F;
                    }

                    const Scalar saturation = chroma / maxRGB;

                    // Bilinear interpolation
                    const int bin0 = hue;
                    const int bin1 = (bin0 < 5) ? (bin0 + 1) : 0; // (hue0 < 5) ? (hue0 + 1) : 0;
                    const Scalar alpha = hue - bin0;
                    const Scalar magnitude0 = saturation * normalization * (1 - alpha);
                    const Scalar magnitude1 = saturation * normalization * alpha;

                    level(iyp    , ixp)(bin0 + startBin) += xp1 * yp1 * magnitude0;
                    level(iyp    , ixp)(bin1 + startBin) += xp1 * yp1 * magnitude1;
                    level(iyp    , ixp + 1)(bin0 + startBin) += xp0 * yp1 * magnitude0;
                    level(iyp    , ixp + 1)(bin1 + startBin) += xp0 * yp1 * magnitude1;
                    level(iyp + 1, ixp)(bin0 + startBin) += xp1 * yp0 * magnitude0;
                    level(iyp + 1, ixp)(bin1 + startBin) += xp1 * yp0 * magnitude1;
                    level(iyp + 1, ixp + 1)(bin0 + startBin) += xp0 * yp0 * magnitude0;
                    level(iyp + 1, ixp + 1)(bin1 + startBin) += xp0 * yp0 * magnitude1;
                }
            }
        }
    }

    // Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
    for(int y = 0; y < level.rows(); ++y) {
        for(int x = 0; x < level.cols(); ++x) {
            Scalar sumSq = 0;

            for(int i = 0; i < 9; ++i)
                sumSq += (level(y, x)(i) + level(y, x)(i + 9)) *
                         (level(y, x)(i) + level(y, x)(i + 9));

            level(y, x)(Dimension - 1) = sumSq;
        }
    }

    // Compute the four normalization factors then normalize and clamp everything
    for(int y = pady; y < level.rows() - pady; ++y) {
        for(int x = padx; x < level.cols() - padx; ++x) {
            const Scalar n0 = 1 / sqrt(level(y - 1, x - 1)(Dimension - 1) +
                                       level(y - 1, x)(Dimension - 1) +
                                       level(y    , x - 1)(Dimension - 1) +
                                       level(y    , x)(Dimension - 1) +
                                       EPS);
            const Scalar n1 = 1 / sqrt(level(y - 1, x)(Dimension - 1) +
                                       level(y - 1, x + 1)(Dimension - 1) +
                                       level(y    , x)(Dimension - 1) +
                                       level(y    , x + 1)(Dimension - 1) +
                                       EPS);
            const Scalar n2 = 1 / sqrt(level(y    , x - 1)(Dimension - 1) +
                                       level(y    , x)(Dimension - 1) +
                                       level(y + 1, x - 1)(Dimension - 1) +
                                       level(y + 1, x)(Dimension - 1) +
                                       EPS);
            const Scalar n3 = 1 / sqrt(level(y    , x)(Dimension - 1) +
                                       level(y    , x + 1)(Dimension - 1) +
                                       level(y + 1, x)(Dimension - 1) +
                                       level(y + 1, x + 1)(Dimension - 1) +
                                       EPS);

            // Contrast-insensitive features
            for(int i = 0; i < 9; ++i) {
                const Scalar sum = level(y, x)(i) + level(y, x)(i + 9);
                const Scalar h0 = min(sum * n0, static_cast<Scalar>(0.2));
                const Scalar h1 = min(sum * n1, static_cast<Scalar>(0.2));
                const Scalar h2 = min(sum * n2, static_cast<Scalar>(0.2));
                const Scalar h3 = min(sum * n3, static_cast<Scalar>(0.2));
                level(y, x)(i + 18) = (h0 + h1 + h2 + h3) *
                                      static_cast<Scalar>(0.5);
            }

            // Contrast-sensitive features
            Scalar t0 = 0;
            Scalar t1 = 0;
            Scalar t2 = 0;
            Scalar t3 = 0;

            for(int i = 0; i < 18; ++i) {
                const Scalar sum = level(y, x)(i);
                const Scalar h0 = min(sum * n0, static_cast<Scalar>(0.2));
                const Scalar h1 = min(sum * n1, static_cast<Scalar>(0.2));
                const Scalar h2 = min(sum * n2, static_cast<Scalar>(0.2));
                const Scalar h3 = min(sum * n3, static_cast<Scalar>(0.2));
                level(y, x)(i) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }

            // Texture features
            level(y, x)(27) = t0 * static_cast<Scalar>(0.2357);
            level(y, x)(28) = t1 * static_cast<Scalar>(0.2357);
            level(y, x)(29) = t2 * static_cast<Scalar>(0.2357);
            level(y, x)(30) = t3 * static_cast<Scalar>(0.2357);
        }
    }

    // Truncation features
    if(!zeroPadx || !zeroPady) {
        for(int y = 0; y < level.rows(); ++y) {
            for(int x = 0; x < level.cols(); ++x) {
                if((y < pady) || (y >= level.rows() - pady) ||
                        (x < padx) || (x >= level.cols() - padx)) {
                    level(y, x) = bgmu;
                } else {
                    level(y, x)(Dimension - 1) = 0;
                }
            }
        }
    }

    if(zeroPadx || zeroPady) {
        int x = zeroPadx ? padx + 1 : 0;
        int y = zeroPady ? pady + 1 : 0;

        Level tmp = level.block(y, x, level.rows() - 2 * y,
                                level.cols() - 2 * x);
        level.swap(tmp);
    }

    if(zeroPadx && zeroPady) {
        for(int y = 0; y < level.rows(); ++y) {
            for(int x = 0; x < level.cols(); ++x) {
                level(y, x)(Dimension - 1) = 0;
            }
        }
    }
}

template<int Dimension>
void FeaturePyr::computeTrackingFeat(const cv::Mat & image,
                                     const FeatureParam & param,
                                     const Cell & bgmu, Level & level) {

    int cellSize = param.cellSize_;
    int padx = param.padx_;
    int pady = param.pady_;

    // Get the size of image
    const int width  = image.cols;
    const int height = image.rows;
    const int depth  = image.channels();

    // Make sure the image is big enough
    if((width < cellSize) || (height < cellSize) || (depth != 3) ||
            !param.isValid()) {
        std::cerr << "Attempting to compute an empty pyramid level" ; fflush(stdout);
        return;
    }

    bool zeroPadx = (padx == 0);
    bool zeroPady = (pady == 0);
    if(zeroPadx) padx = 1;
    if(zeroPady) pady = 1;

    // Resize the feature matrix
    int blockHt = (height + cellSize / 2) / cellSize;
    int blockWd = (width + cellSize / 2)  / cellSize;

    level = Level::Constant(blockHt + 2 * pady, blockWd + 2 * padx,
                            Cell::Zero());

    const Scalar invCellSize = static_cast<Scalar>(1) / cellSize;
    const Scalar tmp = 0.5F;

    for(int y = 0; y < height; ++y) {
        const int yabove = max(y - 1, 0);
        const int ybelow = min(y + 1, height - 1);

        for(int x = 0; x < width; ++x) {
            const int xright = min(x + 1, width - 1);
            const int xleft = max(x - 1, 0);

            const Pixel & pixelyp = image.at<Pixel>(ybelow, x);
            const Pixel & pixelym = image.at<Pixel>(yabove, x);
            const Pixel & pixelxp = image.at<Pixel>(y, xright);
            const Pixel & pixelxm = image.at<Pixel>(y, xleft);

            // Use the channel with the largest gradient magnitude
            int maxMagnitude = 0;
            int argDx = 255;
            int argDy = 255;

            for(int i = 0; i < depth; ++i) {
                const int dx = static_cast<int>(pixelxp[i] - pixelxm[i]);
                const int dy = static_cast<int>(pixelyp[i] - pixelym[i]);

                if(dx * dx + dy * dy > maxMagnitude) {
                    maxMagnitude = dx * dx + dy * dy;
                    argDx = dx + 255;
                    argDy = dy + 255;
                }
            }

            const char bin0 = detail::HOGTable::Singleton().bins[argDy][argDx][0];
            const char bin1 = detail::HOGTable::Singleton().bins[argDy][argDx][1];
            const Scalar magnitude0 =
                detail::HOGTable::Singleton().magnitudes[argDy][argDx][0];
            const Scalar magnitude1 =
                detail::HOGTable::Singleton().magnitudes[argDy][argDx][1];

            // Bilinear interpolation
            const Scalar xp = (x + tmp) * invCellSize + padx - 0.5f;
            const Scalar yp = (y + tmp) * invCellSize + pady - 0.5f;
            const int ixp = xp;
            const int iyp = yp;
            const Scalar xp0 = xp - ixp;
            const Scalar yp0 = yp - iyp;
            const Scalar xp1 = 1 - xp0;
            const Scalar yp1 = 1 - yp0;

            level(iyp    , ixp)(bin0) += xp1 * yp1 * magnitude0;
            level(iyp    , ixp)(bin1) += xp1 * yp1 * magnitude1;
            level(iyp    , ixp + 1)(bin0) += xp0 * yp1 * magnitude0;
            level(iyp    , ixp + 1)(bin1) += xp0 * yp1 * magnitude1;
            level(iyp + 1, ixp)(bin0) += xp1 * yp0 * magnitude0;
            level(iyp + 1, ixp)(bin1) += xp1 * yp0 * magnitude1;
            level(iyp + 1, ixp + 1)(bin0) += xp0 * yp0 * magnitude0;
            level(iyp + 1, ixp + 1)(bin1) += xp0 * yp0 * magnitude1;

        }
    }

    // Normalize by the number of pixels
    const Scalar normalization = 2.0 / (cellSize * cellSize);

    if(param.type_ == HOG_SIMPLE_LBP || param.type_ == HOG_SIMPLE_LBP_COLOR) {

        for(int y = 0; y < height; ++y) {
            const int yabove = max(y - 1, 0);
            const int ybelow = min(y + 1, height - 1);

            for(int x = 0; x < width; ++x) {
                const int xright = min(x + 1, width - 1);
                const int xleft = max(x - 1, 0);

                // Bilinear interpolation
                const Scalar xp = (x + tmp) * invCellSize + padx - 0.5f;
                const Scalar yp = (y + tmp) * invCellSize + pady - 0.5f;
                const int ixp = xp;
                const int iyp = yp;
                const Scalar xp0 = xp - ixp;
                const Scalar yp0 = yp - iyp;
                const Scalar xp1 = 1 - xp0;
                const Scalar yp1 = 1 - yp0;

                // Texture (Uniform LBP) features
                const int LBP_TABLE[256] = {
                    0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9,
                    4, 9, 9, 9, 5, 9, 6, 7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
                    2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7,
                    4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8
                };

                // Use the green channel
                const Pixel & pixel = image.at<Pixel>(y, x);
                const Scalar p = pixel[1];

                // clock-wise pixels in 8 negihborhood of (x,y)
                const int lbp = (static_cast<int>(image.at<Pixel>(yabove, xleft)[1] >= p)) |
                                (static_cast<int>(image.at<Pixel>(yabove, x)[1] >= p) << 1) |
                                (static_cast<int>(image.at<Pixel>(yabove, xright)[1] >= p) << 2) |
                                (static_cast<int>(image.at<Pixel>(y,      xright)[1] >= p) << 3) |
                                (static_cast<int>(image.at<Pixel>(ybelow, xright)[1] >= p) << 4) |
                                (static_cast<int>(image.at<Pixel>(ybelow, x)[1] >= p) << 5) |
                                (static_cast<int>(image.at<Pixel>(ybelow, xleft)[1] >= p) << 6) |
                                (static_cast<int>(image.at<Pixel>(y,      xleft)[1] >= p) << 7);


                // Bilinear interpolation
                level(iyp    , ixp)(LBP_TABLE[lbp] + 22) += xp1 * yp1 * normalization;
                level(iyp    , ixp + 1)(LBP_TABLE[lbp] + 22) += xp0 * yp1 * normalization;
                level(iyp + 1, ixp)(LBP_TABLE[lbp] + 22) += xp1 * yp0 * normalization;
                level(iyp + 1, ixp + 1)(LBP_TABLE[lbp] + 22) += xp0 * yp0 * normalization;
            }
        }
    }

    if(param.type_ == HOG_SIMPLE_COLOR || param.type_ == HOG_SIMPLE_LBP_COLOR) {

        const int startBin = (param.type_ == HOG_SIMPLE_COLOR ? 22 : 32);

        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                // Bilinear interpolation
                const Scalar xp = (x + tmp) * invCellSize + padx - 0.5f;
                const Scalar yp = (y + tmp) * invCellSize + pady - 0.5f;
                const int ixp = xp;
                const int iyp = yp;
                const Scalar xp0 = xp - ixp;
                const Scalar yp0 = yp - iyp;
                const Scalar xp1 = 1 - xp0;
                const Scalar yp1 = 1 - yp0;

                // Color features
                const Pixel & pixel = image.at<Pixel>(y, x);
                const Scalar r = pixel[2] * static_cast<Scalar>(1.0F / 255.0F);
                const Scalar g = pixel[1] * static_cast<Scalar>(1.0F / 255.0F);
                const Scalar b = pixel[0] * static_cast<Scalar>(1.0F / 255.0F);

                const Scalar minRGB = min(r, min(g, b));
                const Scalar maxRGB = max(r, max(g, b));
                const Scalar chroma = maxRGB - minRGB;

                if(chroma > 0.05F) {
                    Scalar hue = 0.0F;

                    if(r == maxRGB) {
                        hue = (g - b) / chroma;
                    } else if(g == maxRGB) {
                        hue = (b - r) / chroma + 2;
                    } else {
                        hue = (r - g) / chroma + 4;
                    }

                    if(hue < 0.0F) {
                        hue += 6.0F;
                    }

                    if(hue >= 6.0F) {
                        hue = 0.0F;
                    }

                    const Scalar saturation = chroma / maxRGB;

                    // Bilinear interpolation
                    const int bin0 = hue;
                    const int bin1 = (bin0 < 5) ? (bin0 + 1) : 0; // (hue0 < 5) ? (hue0 + 1) : 0;
                    const Scalar alpha = hue - bin0;
                    const Scalar magnitude0 = saturation * normalization * (1 - alpha);
                    const Scalar magnitude1 = saturation * normalization * alpha;

                    level(iyp    , ixp)(bin0 + startBin) += xp1 * yp1 * magnitude0;
                    level(iyp    , ixp)(bin1 + startBin) += xp1 * yp1 * magnitude1;
                    level(iyp    , ixp + 1)(bin0 + startBin) += xp0 * yp1 * magnitude0;
                    level(iyp    , ixp + 1)(bin1 + startBin) += xp0 * yp1 * magnitude1;
                    level(iyp + 1, ixp)(bin0 + startBin) += xp1 * yp0 * magnitude0;
                    level(iyp + 1, ixp)(bin1 + startBin) += xp1 * yp0 * magnitude1;
                    level(iyp + 1, ixp + 1)(bin0 + startBin) += xp0 * yp0 * magnitude0;
                    level(iyp + 1, ixp + 1)(bin1 + startBin) += xp0 * yp0 * magnitude1;
                }
            }
        }
    }


    // Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
    Matrix energy(level.rows(), level.cols());
    for(int y = 0; y < level.rows(); ++y) {
        for(int x = 0; x < level.cols(); ++x) {
            Scalar sumSq = 0;

            for(int i = 0; i < 9; ++i)
                sumSq += (level(y, x)(i) + level(y, x)(i + 9)) *
                         (level(y, x)(i) + level(y, x)(i + 9));

            energy(y, x) = sumSq;
        }
    }

    // Compute the four normalization factors then normalize and clamp everything
    for(int y = pady; y < level.rows() - pady; ++y) {
        for(int x = padx; x < level.cols() - padx; ++x) {
            const Scalar n0 = 1 / sqrt(energy(y - 1, x - 1) +
                                       energy(y - 1, x) +
                                       energy(y    , x - 1) +
                                       energy(y    , x) +
                                       EPS);
            const Scalar n1 = 1 / sqrt(energy(y - 1, x) +
                                       energy(y - 1, x + 1) +
                                       energy(y    , x) +
                                       energy(y    , x + 1) +
                                       EPS);
            const Scalar n2 = 1 / sqrt(energy(y    , x - 1) +
                                       energy(y    , x) +
                                       energy(y + 1, x - 1) +
                                       energy(y + 1, x) +
                                       EPS);
            const Scalar n3 = 1 / sqrt(energy(y    , x) +
                                       energy(y    , x + 1) +
                                       energy(y + 1, x) +
                                       energy(y + 1, x + 1) +
                                       EPS);            

            // Contrast-sensitive features
            Scalar t0 = 0;
            Scalar t1 = 0;
            Scalar t2 = 0;
            Scalar t3 = 0;

            for(int i = 0; i < 18; ++i) {
                const Scalar sum = level(y, x)(i);
                const Scalar h0 = min(sum * n0, static_cast<Scalar>(0.2));
                const Scalar h1 = min(sum * n1, static_cast<Scalar>(0.2));
                const Scalar h2 = min(sum * n2, static_cast<Scalar>(0.2));
                const Scalar h3 = min(sum * n3, static_cast<Scalar>(0.2));
                level(y, x)(i) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }

            // Texture features
            level(y, x)(18) = t0 * static_cast<Scalar>(0.2357);
            level(y, x)(19) = t1 * static_cast<Scalar>(0.2357);
            level(y, x)(20) = t2 * static_cast<Scalar>(0.2357);
            level(y, x)(21) = t3 * static_cast<Scalar>(0.2357);
        }
    }

    // Truncation features
    if(!zeroPadx || !zeroPady) {
        for(int y = 0; y < level.rows(); ++y) {
            for(int x = 0; x < level.cols(); ++x) {
                if((y < pady) || (y >= level.rows() - pady) ||
                        (x < padx) || (x >= level.cols() - padx)) {
                    level(y, x) = bgmu;
                }
            }
        }
    }

    if(zeroPadx || zeroPady) {
        int x = zeroPadx ? padx + 1 : 0;
        int y = zeroPady ? pady + 1 : 0;

        Level tmp = level.block(y, x, level.rows() - 2 * y,
                                level.cols() - 2 * x);
        level.swap(tmp);
    }

}

template<int Dimension>
bool FeaturePyr::compare(const Level &a, const Level &b) {

    if ( a.rows() != b.rows() || a.cols() != b.cols() ) return false;

    for ( int i = 0; i < a.rows(); ++i )
        for ( int j = 0; j < a.cols(); ++j )
            for ( int k = 0; k < Dimension; ++k )
                if ( a(i, j)(k) != b(i, j)(k) ) return false;

    return true;
}

/// Instantiation
INSTANTIATE_CLASS_(FeaturePyr_);
INSTANTIATE_BOOST_SERIALIZATION_(FeaturePyr_);


int VirtualPadding(int padding, int ds) {
    // subtract one because each level already
    // has a one padding wide border around it
    return padding * (pow(2, ds) - 1);
}

} // namespace RGM

