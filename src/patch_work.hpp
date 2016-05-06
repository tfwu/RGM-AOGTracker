// This file is adapted from FFLDv2 (the Fast Fourier Linear Detector version 2)
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
#ifndef RGM_PATCH_WORK_HPP_
#define RGM_PATCH_WORK_HPP_

#include <utility>

extern "C" {
#include <fftw3.h>
}

#include "feature_pyr.hpp"
#include "rectangle.hpp"

namespace RGM {

/// The Patchwork class computes full convolutions much faster using FFT
template<int Dimension>
class Patchwork_ {
  public:
    /// Type of a patchwork plane cell
    /// (fixed-size complex vector of size Dimension).
    typedef Eigen::Array<CScalar, Dimension, 1> Cell;
    /// Type of a patchwork plane (matrix of cells).
    typedef Eigen::Matrix<Cell, Dynamic, Dynamic, RowMajor> Plane;
    /// Type of a patchwork filter (plane + original filter size).
    typedef std::pair<Plane, std::pair<int, int> > Filter;

    typedef typename FeaturePyr::Level Level;

    /// Constructs an empty patchwork. An empty patchwork has no plane.
    Patchwork_() : padx_(0), pady_(0), interval_(0), isZeroPadding_(true) {}

    /// Constructs a patchwork from a pyramid.
    Patchwork_(const FeaturePyr & pyramid);

    /// Returns the amount of horizontal zero padding (in cells).
    int padx() const { return padx_; }

    /// Returns the amount of vertical zero padding (in cells).
    int pady() const { return pady_; }

    /// Returns the number of levels per octave in the pyramid.
    int interval() const { return interval_; }

    /// Returns whether the patchwork is empty. An empty patchwork has no plane.
    bool empty() const { return planes_.empty(); }

    /// Returns the convolutions of the patchwork with filters
    /// @param[in] filters The filters.
    /// @param[out] convolutions The convolutions (filters x levels).
    void convolve(const vector<Filter *> & filters,
                  vector<vector<Matrix> > & convolutions) const;

    /// Initializes the FFTW library.
    /// @param[in] maxRows Maximum number of rows of a pyramid level
    /// (including padding).
    /// @param[in] maxCols Maximum number of columns of a pyramid level
    /// (including padding).    
    /// @returns Whether the initialization was successful.
    /// @note Must be called before any other method (including constructors).
    static bool InitFFTW(int maxRows, int maxCols);

    /// Returns the current maximum number of rows of a pyramid level
    /// (including padding).
    static int MaxRows() { return MaxRows_; }

    /// Returns the current maximum number of columns of a pyramid level
    /// (including padding).
    static int MaxCols() { return MaxCols_; }

    /// Returns a transformed version of a filter to be used by the
    /// @c convolve method.
    /// @param[in] filter Filter to transform.
    /// @param[out] result Transformed filter.
    /// @note If Init was not already called or if the filter is larger than
    /// the last maxRows and maxCols passed to the Init method
    /// the result will be empty.
    static void TransformFilter(const Level & filter, Filter & result);

  private:
    int padx_;
    int pady_;
    bool isZeroPadding_;
    int interval_;
    vector<pair<Rectangle, int> > rectangles_;
    vector<Plane> planes_;

    static int MaxRows_;
    static int MaxCols_;
    static int HalfCols_;

#ifndef RGM_USE_DOUBLE
    static fftwf_plan Forwards_;
    static fftwf_plan Inverse_;   
#else
    static fftw_plan Forwards_;
    static fftw_plan Inverse_;    
#endif

}; // class Patchwork

} // namespace RGM

#endif // RGM_PATCH_WORK_HPP_
