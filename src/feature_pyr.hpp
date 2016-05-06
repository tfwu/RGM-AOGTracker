#ifndef RGM_FEATURE_PYR_HPP_
#define RGM_FEATURE_PYR_HPP_

#include "rgm_struct.hpp"

namespace RGM {

/// Feature Pyramid
template<int Dimension>
class FeaturePyr_ {

public:
    /// Type of a cell and a level
    typedef Eigen::Array<Scalar, Dimension, 1> Cell;
    typedef Eigen::Matrix<Cell, Dynamic, Dynamic, RowMajor> Level;

#ifndef RGM_USE_DOUBLE
    typedef Eigen::Array<double, Dimension, 1> dCell;
    typedef Eigen::Matrix<dCell, Dynamic, Dynamic, RowMajor> dLevel;
#else
    typedef Cell   dCell;
    typedef Level  dLevel;
#endif

    /// Constructs an empty pyramid
    FeaturePyr_() : imgWd_(0), imgHt_(0) {}

    /// Destructor
    ~FeaturePyr_();

    /// Copy contructor
    FeaturePyr_(const FeaturePyr &pyr);

    /// Constructs a pyramid from the image of a Scene.
    FeaturePyr_(const Mat & image, const FeatureParam & param);
    FeaturePyr_(const Mat & image, const FeatureParam & param,
                const Cell & bgmu);

    /// Compute the feature pyramid
    void computePyramid(const Mat & image);

    /// Compute a feature level
    static void computeFeature(const Mat & image, const FeatureParam & param,
                               const Cell & bgmu, Level & level);

    /// Returns whether the pyramid is empty. An empty pyramid has no level.
    bool empty() const { return levels_.empty(); }

    /// Returns the param
    const FeatureParam & param() const { return param_; }
    FeatureParam & getParam() { return param_; }
    /// Returns the type of features
    featureType type() const { return param_.type_; }
    /// Returns the cell size
    int cellSize() const { return param_.cellSize_; }
    /// Returns the amount of horizontal zero padding (in cells).
    int padx() const { return param_.padx_; }
    /// Returns the amount of vertical zero padding (in cells).
    int pady() const { return param_.pady_; }
    /// Returns the number of octaves in the pyramid
    int octave() const { return param_.octave_; }
    /// Returns the number of levels per octave in the pyramid.
    int interval() const { return param_.interval_; }
    /// Returns if an extra octave is used
    bool extraOctave() const { return param_.extraOctave_; }
    /// Returns if a part octave is used
    bool partOctave() const { return param_.partOctave_; }

    /// Returns the bg mean
    const Cell & bgmu() const { return bgmu_; }
    Cell & getBgmu() {return bgmu_; }

    /// Returns the pyramid levels.
    const vector<Level> & levels() const { return levels_; }

    /// Returns the scales
    /// @note Scales are given by the following formula:
    /// 2^(1 - @c index / @c interval).
    const vector<Scalar> & scales() const { return scales_; }

    /// Adjusts the scales
    void adjustScales(Scalar multi);

    /// Returns the status of levels
    const vector<bool> & validLevels() const { return validLevels_; }
    vector<bool> & getValidLevels() { return validLevels_; }

    /// Returns the indice of valid levels
    const vector<int> & idxValidLevels();

    /// Returns the number of levels
    int nbLevels() const { return levels_.size(); }

    /// Returns the number of valid levels
    int nbValidLevels() const;

    /// Returns the original image size
    int imgWd() const { return imgWd_; }
    int imgHt() const { return imgHt_; }

    /// Returns the convolutions of the pyramid with a filter.
    /// @param[in] filter Filter.
    /// @param[out] convolutions Convolution of each level.
    void convolve(const Level & filter, vector<Matrix> & convolutions) const;

    /// Returns the flipped version (horizontally) of a level.
    static Level Flip(const Level & level, featureType t);

    /// Maps a pyramid level to a simple matrix
    /// (useful to apply standard matrix operations to it).
    /// @note The size of the matrix will be rows x (cols * Dimension).
    static Eigen::Map<Matrix, Aligned> Map(Level & level);
    /// for double type
    static Eigen::Map<dMatrix, Aligned> dMap(dLevel & level);

    /// Maps a const pyramid level to a simple const matrix
    /// (useful to apply standard matrix operations to it).
    /// @note The size of the matrix will be rows x (cols * Dimension).
    static const Eigen::Map<const Matrix, Aligned> Map(const Level & level);

    /// Converts a pyramid level ta a Mat_
    /// @param[in] startDim Index of starting dimension (out of Dimension)
    /// @param[in] endDim Index of ending dimension
    static cv::Mat_<Scalar> convertToMat(const Level & level,
                                         int startDim, int endDim);

    /// Returns the contrast insensitive orientations
    static cv::Mat_<Scalar> fold(const Level & level, int featType);

    /// Resizes a level
    static void resize(const Level & in, Scalar factor,  Level & out);

    /// Visualizes a pyramid level (HOG features)
    static Mat visualize(const Level & level, int bs=20);

    /// Compares two levels
    static bool compare(const Level & a, const Level & b);

private:
    /// Compute HOG features as described in "Object Detection with
    /// Discriminatively Trained Part Based Models" by Felzenszwalb, Girshick,
    /// McAllester and Ramanan, PAMI 2010
    static void computeHOGDPM(const Mat & image, const FeatureParam & param,
                              const Cell & bgmu, Level & level);

    /// Compute HOG features as implemented in FFLDv2
    /// (the Fast Fourier Linear Detector version 2) by Charles Dubout
    static void computeHOGFFLD(const Mat & image, const FeatureParam & param,
                               const Cell & bgmu, Level & level);

    /// Compute features for object tracking
    static void computeTrackingFeat(const Mat & image, const FeatureParam & param,
                               const Cell & bgmu, Level & level);

    /// Computes the 2D convolution of a pyramid level with a filter
    static void Convolve(const Level & x, const Level & y, Matrix & z);

private:
    FeatureParam param_;
    Cell bgmu_;

    vector<Level> levels_;    
    vector<Scalar> scales_;
    vector<bool> validLevels_;
    vector<int> idxValidLevels_;

    int imgWd_;
    int imgHt_;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(param_);
        ar & BOOST_SERIALIZATION_NVP(bgmu_);
        ar & BOOST_SERIALIZATION_NVP(levels_);        
        ar & BOOST_SERIALIZATION_NVP(scales_);
        ar & BOOST_SERIALIZATION_NVP(validLevels_);
        ar & BOOST_SERIALIZATION_NVP(idxValidLevels_);
        ar & BOOST_SERIALIZATION_NVP(imgWd_);
        ar & BOOST_SERIALIZATION_NVP(imgHt_);
    }

    DEFINE_RGM_LOGGER;
};

/// Computes the virtual padding
int VirtualPadding(int padding, int ds);

} // namespace RGM



/// Some compilers complain about the lack of a NumTraits for
/// Eigen::Array<Scalar, Dimension, 1>
namespace Eigen
{
#define NUM_TRAITS(n) \
    template <> \
    struct NumTraits<Array<RGM::Scalar, n, 1> > : \
    GenericNumTraits<Array<RGM::Scalar, n, 1> > { \
    static inline RGM::Scalar dummy_precision() \
{ \
    return 0; \
} \
}

NUM_TRAITS(32);
NUM_TRAITS(42);
NUM_TRAITS(48);

#ifndef RGM_USE_DOUBLE

#define NUM_TRAITS_DOUBLE(n) \
    template <> \
    struct NumTraits<Array<double, n, 1> > :\
    GenericNumTraits<Array<double, n, 1> > {\
    static inline double dummy_precision()\
{\
    return 0;\
}\
}

NUM_TRAITS_DOUBLE(32);
NUM_TRAITS_DOUBLE(42);
NUM_TRAITS_DOUBLE(48);

#endif
} // namespace Eigen

/// Serialize FeaturePyr_::Level
namespace boost
{
namespace serialization
{
#define SAVE_LEVEL(n) \
    template <class Archive> \
    void save(Archive & ar, const RGM::FeaturePyr_<n>::Level & m, \
    const unsigned int version) \
{ \
    int rows=m.rows(),cols=m.cols(); \
    ar & BOOST_SERIALIZATION_NVP(rows); \
    ar & BOOST_SERIALIZATION_NVP(cols); \
    int dim = n; \
    ar & BOOST_SERIALIZATION_NVP(dim); \
    for ( int i = 0; i < rows; ++i ) { \
    for ( int j = 0; j < cols; ++j ) { \
    ar & make_array(m(i,j).data(), dim); \
} \
} \
}

#define LOAD_LEVEL(n) \
    template <class Archive> \
    void load(Archive & ar, RGM::FeaturePyr_<n>::Level & m,\
    const unsigned int version) \
{ \
    int rows,cols, dim; \
    ar & BOOST_SERIALIZATION_NVP(rows); \
    ar & BOOST_SERIALIZATION_NVP(cols); \
    ar & BOOST_SERIALIZATION_NVP(dim); \
    assert(dim == n); \
    m = RGM::FeaturePyr_<n>::Level::Constant(rows, cols,\
    RGM::FeaturePyr_<n>::Cell::Zero());\
    for ( int i = 0; i < rows; ++i ) { \
    for ( int j = 0; j < cols; ++j ) { \
    ar & make_array(m(i,j).data(), dim); \
} \
}\
}

#define SERIALIZE_LEVEL(n) \
    template <class Archive> \
    void serialize(Archive & ar, RGM::FeaturePyr_<n>::Level & m,\
    const unsigned int version) \
{ \
    split_free(ar,m,version); \
}

SAVE_LEVEL(32)
SAVE_LEVEL(42)
SAVE_LEVEL(48)

LOAD_LEVEL(32)
LOAD_LEVEL(42)
LOAD_LEVEL(48)

SERIALIZE_LEVEL(32)
SERIALIZE_LEVEL(42)
SERIALIZE_LEVEL(48)

} // namespace serialization
} // namespace boost

#endif // RGM_FEATURE_PYR_HPP_
