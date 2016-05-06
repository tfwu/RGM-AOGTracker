#include <cstdio>
#include <numeric>
#include <set>

#include "patch_work.hpp"

namespace RGM {

using namespace Eigen;
using namespace std;

#ifndef RGM_CPU_L1_CACHE_SIZE
#define RGM_CPU_L1_CACHE_SIZE 1 // means 32k
#endif

template<int Dimension> int Patchwork::MaxRows_(0);
template<int Dimension> int Patchwork::MaxCols_(0);
template<int Dimension> int Patchwork::HalfCols_(0);

#ifndef RGM_USE_DOUBLE
template<int Dimension> fftwf_plan Patchwork::Forwards_(0);
template<int Dimension> fftwf_plan Patchwork::Inverse_(0);
#else
template<int Dimension> fftw_plan Patchwork::Forwards_(0);
template<int Dimension> fftw_plan Patchwork::Inverse_(0);
#endif

template<int Dimension>
Patchwork::Patchwork_(const FeaturePyr & pyramid) :
    padx_(pyramid.padx()), pady_(pyramid.pady()), interval_(pyramid.interval()) {

    // Remove the zero padding from the bottom/right sides
    // since convolutions with Fourier wrap around
    const typename FeaturePyr::Cell & bgmu(pyramid.bgmu());
    isZeroPadding_ = true;
    //    for(int i = 0; i < Dimension - 1; ++i) {
    //        if(bgmu(i) != 0) {
    //            isZeroPadding_ = false;
    //            break;
    //        }
    //    }

    const int nbLevels = pyramid.nbLevels();
    const int nbValidLevels = pyramid.nbValidLevels();

    rectangles_.resize(nbValidLevels);
    std::vector<int> idxValidLevels(nbValidLevels);

    int wd = (isZeroPadding_ ? padx_ : 0);
    int ht = (isZeroPadding_ ? pady_ : 0);

    for(int i = 0, j = 0; i < nbLevels; ++i) {
        if(pyramid.validLevels()[i]) {
            rectangles_[j].first.setWidth(pyramid.levels()[i].cols() - wd);
            rectangles_[j].first.setHeight(pyramid.levels()[i].rows() - ht);
            idxValidLevels[j] = i;
            j++;
        }
    }

    // Build the patchwork planes
    const int nbPlanes = blf(rectangles_, MaxCols_, MaxRows_);

    // Constructs an empty patchwork in case of error
    if(nbPlanes <= 0) {
        return;
    }

    planes_.resize(nbPlanes);

    for(int i = 0; i < nbPlanes; ++i) {
        planes_[i] = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());

        Map<Level, Aligned>
                plane(reinterpret_cast<typename FeaturePyr::Cell *>(
                          planes_[i].data()), MaxRows_, HalfCols_ * 2);

        // Set the last feature to 1
        for(int y = 0; y < MaxRows_; ++y)
            for(int x = 0; x < MaxCols_; ++x) {
                plane(y, x) = bgmu; //(Dimension - 1) = 1.0f;
            }
    }

    // Recopy the pyramid levels into the planes
    for(int i = 0; i < nbValidLevels; ++i) {
        Map<Level, Aligned>
                plane(reinterpret_cast<typename FeaturePyr::Cell *>(
                          planes_[rectangles_[i].second].data()),
                MaxRows_, HalfCols_ * 2);

        int j = idxValidLevels[i];
        plane.block(rectangles_[i].first.y(), rectangles_[i].first.x(),
                    rectangles_[i].first.height(),
                    rectangles_[i].first.width()) =
                pyramid.levels()[j].topLeftCorner(
                    rectangles_[i].first.height(),
                    rectangles_[i].first.width());
    }

    // Transform the planes
    int i;
#pragma omp parallel for private(i)
    for(i = 0; i < nbPlanes; ++i)
#ifndef RGM_USE_DOUBLE
        fftwf_execute_dft_r2c(Forwards_,
                              reinterpret_cast<float *>(
                                  planes_[i].data()->data()),
                              reinterpret_cast<fftwf_complex *>(
                                  planes_[i].data()->data()));
#else
        fftw_execute_dft_r2c(Forwards_,
                             reinterpret_cast<double *>(
                                 planes_[i].data()->data()),
                             reinterpret_cast<fftw_complex *>(
                                 planes_[i].data()->data()));
#endif
}

template<int Dimension>
void Patchwork::convolve(const vector<Filter *> & filters,
                         vector<vector<Matrix> > & convolutions) const {
    const int nbFilters = filters.size();
    const int nbPlanes = planes_.size();
    const int nbLevels = rectangles_.size(); // # of valid levels

    // Early return if the patchwork or the filters are empty
    if(empty() || !nbFilters) {
        convolutions.clear();
        return;
    }

    // Pointwise multiply the transformed filters with the patchwork's planes
    vector<vector<CMatrix> > sums(nbFilters);

    for(int i = 0; i < nbFilters; ++i) {
        sums[i].resize(nbPlanes);

        for(int j = 0; j < nbPlanes; ++j) {
            sums[i][j].resize(MaxRows_, HalfCols_);
        }
    }

    // The following assumptions are not dangerous in the sense that
    // the program will only work slower if they do not hold
    // Assume L1 cache of x*32K
    const int cacheSize = RGM_CPU_L1_CACHE_SIZE * 32768;
    // Assume nbPlanes < nbFilters
    const int fragmentsSize = (nbPlanes + 1) * sizeof(Cell);
    const int step = min(cacheSize / fragmentsSize,
                     #ifdef _OPENMP
                         MaxRows_ * HalfCols_ / omp_get_max_threads());
#else
                         MaxRows_ * HalfCols_);
#endif

    int i;
#pragma omp parallel for private(i)
    for(i = 0; i <= MaxRows_ * HalfCols_ - step; i += step)
        for(int j = 0; j < nbFilters; ++j)
            for(int k = 0; k < nbPlanes; ++k)
                for(int l = 0; l < step; ++l)
                    sums[j][k](i + l) =
                            (*filters[j]).first(i + l).cwiseProduct(
                                planes_[k](i + l)).sum();

    for(i = MaxRows_ * HalfCols_ - ((MaxRows_ * HalfCols_) % step);
        i < MaxRows_ * HalfCols_; ++i)
        for(int j = 0; j < nbFilters; ++j)
            for(int k = 0; k < nbPlanes; ++k) {
                sums[j][k](i) = (*filters[j]).first(i).cwiseProduct(
                            planes_[k](i)).sum();
            }

    // Transform back the results and store them in convolutions
    convolutions.resize(nbFilters);

    for(int i = 0; i < nbFilters; ++i) {
        convolutions[i].resize(nbLevels);
    }

    int wd = (isZeroPadding_ ? padx_ : 0);
    int ht = (isZeroPadding_ ? pady_ : 0);

#pragma omp parallel for private(i)
    for(i = 0; i < nbFilters * nbPlanes; ++i) {
        const int k = i / nbPlanes; // Filter index
        const int l = i % nbPlanes; // Plane index

        Map<Matrix, Aligned>
                output(reinterpret_cast<Scalar *>(sums[k][l].data()), MaxRows_,
                       HalfCols_ * 2);

#ifndef RGM_USE_DOUBLE
        fftwf_execute_dft_c2r(Inverse_,
                              reinterpret_cast<fftwf_complex *>(sums[k][l].data()),
                              output.data());
#else
        fftw_execute_dft_c2r(Inverse_,
                             reinterpret_cast<fftw_complex *>(sums[k][l].data()),
                             output.data());
#endif

        for(int j = 0; j < nbLevels; ++j) {
            const int rows = rectangles_[j].first.height() + ht -
                    (*filters[k]).second.first + 1;
            const int cols = rectangles_[j].first.width() + wd -
                    (*filters[k]).second.second + 1;

            if((rows > 0) && (cols > 0) && (rectangles_[j].second == l)) {
                const int x = rectangles_[j].first.x();
                const int y = rectangles_[j].first.y();
                const int width = rectangles_[j].first.width();
                const int height = rectangles_[j].first.height();

                if((rows <= height) && (cols <= width)) {
                    convolutions[k][j] = output.block(y, x, rows, cols);
                } else {
                    convolutions[k][j].resize(rows, cols);
                    convolutions[k][j].topLeftCorner
                            (min(rows, height), min(cols, width)) =
                            output.block(y, x, min(rows, height), min(cols, width));

                    if(rows > height) {
                        convolutions[k][j].bottomRows(
                                    rows - height).fill(output(y, x));
                    }

                    if(cols > width) {
                        convolutions[k][j].rightCols(
                                    cols - width).fill(output(y, x));
                    }
                }
            }
        }
    }
}

template<int Dimension>
bool Patchwork::InitFFTW(int maxRows, int maxCols) {
    // It is an error if maxRows or maxCols are too small
    if((maxRows < 2) || (maxCols < 2)) {
        return false;
    }

    int myrank = 0;
#ifdef RGM_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

    // Temporary matrices
    Matrix tmp(maxRows * Dimension, maxCols + 2);

    int dims[2] = {maxRows, maxCols};

#ifndef RGM_USE_DOUBLE
    if ( Forwards_ ) {
        fftwf_destroy_plan(Forwards_);
    }
    if ( Inverse_ ) {
        fftwf_destroy_plan(Inverse_);
    }

    // Use fftwf_import_wisdom_from_file and not fftwf_import_wisdom_from_filename as old versions
    // of fftw seem to not include it
    string planFilename = string("wisdom_") + boost::lexical_cast<string>(myrank) + string(".fftw");

    FILE * file = fopen(planFilename.c_str(), "r");

    if(file) {
        fftwf_import_wisdom_from_file(file);
        fclose(file);
    }

    const fftwf_plan forwards =
            fftwf_plan_many_dft_r2c(2, dims, Dimension, tmp.data(), 0,
                                    Dimension, 1,
                                    reinterpret_cast<fftwf_complex *>(tmp.data()), 0,
                                    Dimension, 1, FFTW_PATIENT);

    const fftwf_plan inverse =
            fftwf_plan_dft_c2r_2d(dims[0], dims[1],
            reinterpret_cast<fftwf_complex *>(tmp.data()),
            tmp.data(), FFTW_PATIENT);

    file = fopen(planFilename.c_str(), "w");

    if(file) {
        fftwf_export_wisdom_to_file(file);
        fclose(file);
    }


    //        std::string filename("wisdom.fftw");
    //        fftwf_import_wisdom_from_filename(filename.c_str());

    //        const fftwf_plan forwards =
    //            fftwf_plan_many_dft_r2c(2, dims, Dimension, tmp.data(), 0,
    //                                    Dimension, 1,
    //                                    reinterpret_cast<fftwf_complex *>(tmp.data()), 0,
    //                                    Dimension, 1, FFTW_PATIENT);

    //        const fftwf_plan inverse =
    //            fftwf_plan_dft_c2r_2d(dims[0], dims[1], reinterpret_cast<fftwf_complex *>(tmp.data()),
    //                                  tmp.data(), FFTW_PATIENT);

    //        fftwf_export_wisdom_to_filename(filename.c_str());
#else
    if ( Forwards_ ) {
        fftw_destroy_plan(Forwards_);
    }
    if ( Inverse_ ) {
        fftw_destroy_plan(Inverse_);
    }

    FILE * file = fopen("wisdom.fftw", "r");

    if(file) {
        fftw_import_wisdom_from_file(file);
        fclose(file);
    }

    const fftw_plan forwards =
            fftw_plan_many_dft_r2c(2, dims, Dimension, tmp.data(), 0,
                                   Dimension, 1,
                                   reinterpret_cast<fftw_complex *>(tmp.data()), 0,
                                   Dimension, 1, FFTW_PATIENT);

    const fftw_plan inverse =
            fftw_plan_dft_c2r_2d(dims[0], dims[1],
            reinterpret_cast<fftw_complex *>(tmp.data()),
            tmp.data(), FFTW_PATIENT);

    file = fopen("wisdom.fftw", "w");

    if(file) {
        fftw_export_wisdom_to_file(file);
        fclose(file);
    }
#endif

    // If successful, set MaxRows_, MaxCols_, HalfCols_, Forwards_ and Inverse_
    if(forwards && inverse) {
        MaxRows_ = maxRows;
        MaxCols_ = maxCols;
        HalfCols_ = maxCols / 2 + 1;
        Forwards_ = forwards;
        Inverse_ = inverse;
        return true;
    }

    return false;
}

template<int Dimension>
void Patchwork::TransformFilter( const Level & filter, Filter & result) {
    // Early return if no filter given or if Init was not called or if the filter is too large
    if(!filter.size() || !MaxRows_ || (filter.rows() > MaxRows_) ||
            (filter.cols() > MaxCols_)) {
        result = Filter();
        return;
    }

    // Recopy the filter into a plane
    result.first = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());
    result.second = pair<int, int>(filter.rows(), filter.cols());

    Map<Level, Aligned> plane(
                reinterpret_cast<typename FeaturePyr::Cell *>(result.first.data()),
                MaxRows_, HalfCols_ * 2);

    for(int y = 0; y < filter.rows(); ++y)
        for(int x = 0; x < filter.cols(); ++x)
            plane((MaxRows_ - y) % MaxRows_, (MaxCols_ - x) % MaxCols_) = filter(y, x) /
                    (MaxRows_ * MaxCols_);

    // Transform that plane
#ifndef RGM_USE_DOUBLE
    fftwf_execute_dft_r2c(Forwards_,
                          reinterpret_cast<float *>(plane.data()->data()),
                          reinterpret_cast<fftwf_complex *>(result.first.data()->data()));
#else
    fftw_execute_dft_r2c(Forwards_,
                         reinterpret_cast<double *>(plane.data()->data()),
                         reinterpret_cast<fftw_complex *>(result.first.data()->data()));
#endif
}

/// Instantiation
INSTANTIATE_CLASS_(Patchwork_);

} // namespace RGM
