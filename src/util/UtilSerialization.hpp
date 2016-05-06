#ifndef RGM_UTILSERIALIZATION_HPP_
#define RGM_UTILSERIALIZATION_HPP_

#include "common.hpp"

namespace boost
{
namespace serialization
{
/// Eigen::Array
template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & ar, const Eigen::Array<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    int rows=m.rows(),cols=m.cols();
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(cols);
    ar & make_array(m.data(), rows*cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & ar, Eigen::Array<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    int rows,cols;
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(cols);
    m.resize(rows,cols);
    ar & make_array(m.data(), rows*cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void serialize(Archive & ar, Eigen::Array<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    split_free(ar,m,version);
}

///  Eigen::Matrix
template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & ar, const Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    int rows=m.rows(),cols=m.cols();
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(cols);
    ar & make_array(m.data(), rows*cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & ar, Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    int rows,cols;
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(cols);
    m.resize(rows,cols);
    ar & make_array(m.data(), rows*cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void serialize(Archive & ar, Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version)
{
    split_free(ar,m,version);
}


/// cv::Mat
template<class Archive>
void save(Archive & ar, const cv::Mat& m, const unsigned int version) {
    std::size_t elem_size = m.elemSize();
    std::size_t elem_type = m.type();

    ar & BOOST_SERIALIZATION_NVP(m.cols);
    ar & BOOST_SERIALIZATION_NVP(m.rows);
    ar & BOOST_SERIALIZATION_NVP(elem_size);
    ar & BOOST_SERIALIZATION_NVP(elem_type);

    const std::size_t data_size = m.cols * m.rows * elem_size;
    ar & make_array(m.ptr(), data_size);
}

template<class Archive>
void load(Archive & ar, cv::Mat& m, const unsigned int version)
{
    int cols, rows;
    std::size_t elem_size, elem_type;

    ar & BOOST_SERIALIZATION_NVP(cols);
    ar & BOOST_SERIALIZATION_NVP(rows);
    ar & BOOST_SERIALIZATION_NVP(elem_size);
    ar & BOOST_SERIALIZATION_NVP(elem_type);

    m.create(rows, cols, elem_type);

    std::size_t data_size = m.cols * m.rows * elem_size;
    ar & boost::serialization::make_array(m.ptr(), data_size);

}

template<class Archive>
void serialize(Archive & ar, cv::Mat & m, const unsigned int version) {

   split_free(ar,m,version);
}


/// cv::PCA
template<class Archive>
void serialize(Archive & ar, cv::PCA & m, const unsigned int version) {

    ar & BOOST_SERIALIZATION_NVP(m.eigenvalues);
    ar & BOOST_SERIALIZATION_NVP(m.eigenvectors);
    ar & BOOST_SERIALIZATION_NVP(m.mean);
}

/// cv::Rect
template<class Archive>
void serialize(Archive & ar, cv::Rect & m, const unsigned int version) {

    ar & BOOST_SERIALIZATION_NVP(m.x);
    ar & BOOST_SERIALIZATION_NVP(m.y);
    ar & BOOST_SERIALIZATION_NVP(m.width);
    ar & BOOST_SERIALIZATION_NVP(m.height);
}


} // namespace serialziation
} // namespace boost

#endif // RGM_UTILSERIALIZATION_HPP_
