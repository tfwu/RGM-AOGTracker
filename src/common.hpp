#ifndef RGM_COMMON_HPP_
#define RGM_COMMON_HPP_

#ifdef RGM_USE_MPI
#include "mpi.h"
#endif

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <cmath>
#include <fstream>
#include <ostream>
#include <istream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <assert.h>
#include <algorithm>
#include <numeric>
//#include <random> // need c++11 support

#include <Eigen/Core>

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "util/UtilLog.hpp"

namespace RGM {
/// Type of a Scalar
#ifndef RGM_USE_DOUBLE
typedef float Scalar;
#else
typedef double Scalar;
#endif

/// Instantiate serialization
#define INSTANTIATE_BOOST_SERIALIZATION(classname) \
    template  void classname::serialize<boost::archive::binary_iarchive>(\
    boost::archive::binary_iarchive& ar, const unsigned int version); \
    template  void classname::serialize<boost::archive::binary_oarchive>(\
    boost::archive::binary_oarchive& ar, const unsigned int version); \
    template  void classname::serialize<boost::archive::text_iarchive>(\
    boost::archive::text_iarchive& ar, const unsigned int version); \
    template  void classname::serialize<boost::archive::text_oarchive>(\
    boost::archive::text_oarchive& ar, const unsigned int version)

/// Define serialization
#define DEFINE_SERIALIZATION \
    private:\
    friend class boost::serialization::access;\
    template<class Archive>\
    void serialize(Archive & ar, const unsigned int version)

/// Instantiate class w.r.t. the feature dimension
#define INSTANTIATE_CLASS_(classname) \
    template class classname<22>; \
    template class classname<28>; \
    template class classname<32>; \
    template class classname<42>; \
    template class classname<38>; \
    template class classname<48>

#define INSTANTIATE_BOOST_SERIALIZATION_(classname) \
    INSTANTIATE_BOOST_SERIALIZATION(classname<22>); \
    INSTANTIATE_BOOST_SERIALIZATION(classname<28>); \
    INSTANTIATE_BOOST_SERIALIZATION(classname<32>); \
    INSTANTIATE_BOOST_SERIALIZATION(classname<42>); \
    INSTANTIATE_BOOST_SERIALIZATION(classname<38>); \
    INSTANTIATE_BOOST_SERIALIZATION(classname<48>)

/// round
#define ROUND(x) boost::math::round(x)

/// Common functions and classes from std
using std::fstream;
using std::ostream;
using std::istream;
using std::ios;
using std::isnan;
using std::isinf;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using std::max;
using std::min;
using std::numeric_limits;

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Aligned;

using cv::Mat;
#define RGM_IMG_RESIZE cv::INTER_AREA

/// Type of a complex value.
typedef std::complex<Scalar> CScalar;

/// Type of a int matrix
typedef Eigen::Matrix<int, Dynamic, Dynamic, RowMajor> MatrixXi;

/// Type of a matrix.
typedef Eigen::Matrix<Scalar, Dynamic, Dynamic, RowMajor> Matrix;

/// Type of a complex matrix.
typedef Eigen::Matrix<CScalar, Dynamic, Dynamic, RowMajor> CMatrix;

/// Double precision for gradient used in optimization
#ifndef RGM_USE_DOUBLE
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> dMatrix;
#else
typedef Matrix  dMatrix;
#endif

/// Type of a color image pixel
typedef cv::Vec<Scalar, 3> Pixel;

/// Type of an Anchor point in the feature pyramid  (x, y, s)
typedef Eigen::RowVector3i  Anchor;

/// Const.
static const Scalar Inf = std::numeric_limits<Scalar>::infinity();
static const Scalar EPS = std::numeric_limits<Scalar>::epsilon();

///
#ifndef FeaturePyr
#define FeaturePyr FeaturePyr_<Dimension>
#endif

#ifndef FeatureBg
#define FeatureBg FeatureBg_<Dimension>
#endif

#ifndef Appearance
#define Appearance Appearance_<Dimension>
#endif

#ifndef Node
#define Node Node_<Dimension>
#endif

#ifndef Edge
#define Edge Edge_<Dimension>
#endif

#ifndef AOGrammar
#define AOGrammar AOGrammar_<Dimension>
#endif

#ifndef ParseTree
#define ParseTree ParseTree_<Dimension>
#endif

#ifndef PtIntersector
#define PtIntersector PtIntersector_<Dimension>
#endif

#ifndef TrainSample
#define TrainSample TrainSample_<Dimension>
#endif

#ifndef TrainSampleSet
#define TrainSampleSet TrainSampleSet_<Dimension>
#endif

#ifndef Patchwork
#define Patchwork Patchwork_<Dimension>
#endif

#ifndef DPInference
#define DPInference DPInference_<Dimension>
#endif

#ifndef ParameterLearner
#define ParameterLearner ParameterLearner_<Dimension>
#endif

#ifndef Subcategory
#define Subcategory Subcategory_<Dimension>
#endif

#ifndef AdditiveFilter
#define AdditiveFilter AdditiveFilter_<Dimension>
#endif

#ifndef AdditiveModel
#define AdditiveModel AdditiveModel_<Dimension>
#endif

#ifndef DecisionPolicy
#define DecisionPolicy DecisionPolicy_<Dimension>
#endif

#ifndef PolicyInference
#define PolicyInference PolicyInference_<Dimension>
#endif

/// File separator
#if (defined(WIN32)  || defined(_WIN32) || defined(WIN64) || defined(_WIN64))
static const char FILESEP = '\\';
#else
static const char FILESEP = '/';
#endif

/// Read
#define RGM_READ(filename, t) \
    std::ifstream in; \
    in.open(filename.c_str(), std::ios::in); \
    if ( !in.is_open() ) return false; \
    switch (t) {\
    case 1: {\
        boost::archive::text_iarchive ia(in);\
        ia >> *this;\
        break;\
    }\
    case 0:\
    default: {\
        boost::archive::binary_iarchive ia(in);\
        ia >> *this;\
        break;\
    }\
    }\
    in.close()

/// Save
#define RGM_SAVE(filename, t) \
    std::ofstream out;\
    out.open(filename.c_str(), std::ios::out);\
    if ( !out.is_open() ) {\
        RGM_LOG(error, "Failed to write to file " + filename);\
        return;\
    }\
    switch (t) {\
    case 1: {\
        boost::archive::text_oarchive oa(out);\
        oa << *this;\
        break;\
    }\
    case 0:\
    default: {\
        boost::archive::binary_oarchive oa(out);\
        oa << *this;\
        break;\
    }\
    }\
    out.close()

#define RGM_FUNC_CONFIG(funcName) \
    switch(Dimension) { \        
        case 32: \
            funcName<32>(config); \
            break; \
        case 42: \
            funcName<42>(config); \
            break; \
        case 38: \
            funcName<38>(config); \
            break; \
        case 48: \
            funcName<48>(config); \
            break; \
        default: \
            RGM_LOG(error, "wrong feature type"); \
            break; \
    }

#define RGM_FUNC(funcName) \
    switch(Dimension) { \
        case 32: \
            funcName<32>(); \
            break; \
        case 42: \
            funcName<42>(); \
            break; \
        case 38: \
            funcName<38>(); \
            break; \
        case 48: \
            funcName<48>(); \
            break; \
        default: \
            RGM_LOG(error, "wrong feature type"); \
            break; \
    }


}  /// namespace RGM

#endif  /// RGM_COMMON_HPP_
