// This file is adapted from FFLDv2 (the Fast Fourier Linear Detector version 2)
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
#ifndef RGM_RECTANGLE_HPP_
#define RGM_RECTANGLE_HPP_

#include "common.hpp"

namespace RGM {
/// The Rectangle_ class defines a rectangle in the plane using T precision.
/// If the coordinates of the top left corner of the rectangle are (x, y),
/// the coordinates of the bottom right corner are
/// (x + width - 1, y + height - 1), where width and height are the dimensions
/// of the rectangle.
template <typename T>
class Rectangle_ {
  public:
    /// Constructs an empty rectangle. An empty rectangle has no area.
    Rectangle_() : x_(0), y_(0), width_(0), height_(0) {}

    /// Constructs a rectangle with the given @p width and @p height.
    Rectangle_(T width, T height) : x_(0), y_(0), width_(width),
        height_(height) {}

    /// Constructs a rectangle with coordinates (@p x, @p y) and the given
    /// @p width and @p height.
    Rectangle_(T x, T y, T width, T height) : x_(x), y_(y), width_(width),
        height_(height) {}

    /// Constructs a rectangle by copying from @p rect
    Rectangle_(const Rectangle_<T> & rect) : x_(rect.x()), y_(rect.y()),
        width_(rect.width()), height_(rect.height()) {}

    /// Constructs a rectangle with cv::Rect_ @p rect
    Rectangle_(const cv::Rect_<T> & rect) : x_(rect.x), y_(rect.y),
        width_(rect.width), height_(rect.height) {}

    /// Returns the x-coordinate of the rectangle.
    T x() const { return x_; }

    /// Sets the x coordinate of the rectangle to @p x.
    void setX(T x) { x_ = x; }

    /// Returns the y-coordinate of the rectangle.
    T y() const { return y_; }

    /// Sets the y coordinate of the rectangle to @p y.
    void setY(T y) { y_ = y; }

    /// Returns the width of the rectangle.
    T width() const { return width_; }

    /// Sets the height of the rectangle to the given @p width.
    void setWidth(T width) { width_ = width; }

    /// Returns the height of the rectangle.
    T height() const { return height_; }

    /// Sets the height of the rectangle to the given @p height.
    void setHeight(T height) { height_ = height; }

    /// Returns the left side of the rectangle.
    /// @note Equivalent to x().
    T left() const { return x(); }

    /// Sets the left side of the rectangle to @p left.
    /// @note The right side of the rectangle is not modified.
    void setLeft(T left) {setWidth(right() - left + 1); setX(left);}

    /// Returns the top side of the rectangle.
    /// @note Equivalent to y().
    T top() const { return y(); }

    /// Sets the top side of the rectangle to @p top.
    /// @note The bottom side of the rectangle is not modified.
    void setTop(T top) {setHeight(bottom() - top + 1); setY(top);}

    /// Returns the right side of the rectangle.
    /// @note Equivalent to x() + width() - 1.
    T right() const { return x() + width() - 1; }

    /// Sets the right side of the rectangle to @p right.
    /// @note The left side of the rectangle is not modified.
    void setRight(T right) { setWidth(right - left() + 1); }

    /// Returns the bottom side of the rectangle.
    /// @note Equivalent to y() + height() - 1.
    T bottom() const { return y() + height() - 1; }

    /// Sets the bottom side of the rectangle to @p bottom.
    /// @note The top side of the rectangle is not modified.
    void setBottom(T bottom) { setHeight(bottom - top() + 1); }

    /// Returns the center
    T xcenter() const { return x() + width() / 2; }
    T ycenter() const { return y() + height() / 2; }

    /// Sets the width and height to the maximum
    void setMaxWidth(T wd) { setWidth(max(wd, width())); }
    void setMaxHeight(T ht) { setHeight(max(ht, height())); }
    void setMax(const Rectangle_<T> & rect) {
        setMaxWidth(rect.width());
        setMaxHeight(rect.height());
    }

    /// Sets the width and height to the minimum
    void setMinWidth(T wd) { setWidth(min(wd, width())); }
    void setMinHeight(T ht) { setHeight(min(ht, height())); }
    void setMin(const Rectangle_<T> & rect) {
        setMinWidth(rect.width());
        setMinHeight(rect.height());
    }

    /// Returns whether the rectangle is empty. An empty rectangle has no area.
    bool empty() const {
        return (width() <= 0) || (height() <= 0) ||
               isnan(x()) || isnan(y()) ||
               isnan(width()) || isnan(height());
    }

    /// Returns the area of the rectangle.
    /// @note Equivalent to max(width(), 0) * max(height(), 0).
    T area() const { return max<T>(width(), 0) * max<T>(height(), 0); }

    /// Return if it is the same as rect
    bool operator== (const Rectangle_<T> & rect) const {
        return (x() == rect.x() && y() == rect.y() &&
                width() == rect.width() && height() == rect.height());
    }

    /// Returns if it is the same type as rect
    /// @note The type of a rectangle is defined by @p width and @p height
    bool isSameType(const Rectangle_<T> & rect) const {
        return (width() == rect.width() && height() == rect.height());
    }

    /// Returns the partitioned rectangles with @p rect
    /// @note If it does not overlap with @p rect Returns both of them directly
    /// @note If it overlaps with @p rect Returns the partitioned rectangles
    std::vector<Rectangle_<T> > partition(Rectangle_<T> &rect);

    /// Converts to cv::Rect
    cv::Rect_<T> cvRect() const {
        return cv::Rect_<T>(x(), y(), width(), height());
    }

    /// Expands
    Rectangle_<T> expand(T num, T * maxWd = NULL, T * maxHt = NULL);

    /// clip
    bool clip(int wd, int ht);

  private:
    T x_;
    T y_;
    T width_;
    T height_;

    DEFINE_SERIALIZATION;

}; /// class Rectangle_

/// Typedef
typedef Rectangle_<int>        Rectangle;
typedef std::vector<Rectangle> Rectangles;

/// Returns all the overlapped rectangles between any pair of rectangles
/// in @p rects1 and @p rects2
Rectangles getOverlappedRectangles(Rectangles &rects1,  Rectangles &rects2);


namespace detail {
/// Order rectangles by decreasing area.
class AreaComparator {
  public:
    AreaComparator(const std::vector<std::pair<Rectangle, int> > & rectangles)
        : rectangles_(rectangles)
    { }

    /// Returns whether rectangle @p a comes before @p b.
    bool operator()(int a, int b) const {
        const int areaA = rectangles_[a].first.area();
        const int areaB = rectangles_[b].first.area();

        return (areaA > areaB) ||
               ((areaA == areaB) && (rectangles_[a].first.height() >
                                     rectangles_[b].first.height()));
    }

  private:
    const std::vector<std::pair<Rectangle, int> > & rectangles_;
}; /// class AreaComparator

/// Order free gaps (rectangles) by position and then by size
struct PositionComparator {
    /// Returns whether rectangle @p a comes before @p b
    bool operator()(const Rectangle & a, const Rectangle & b) const {
        return (a.y() < b.y()) ||
               ((a.y() == b.y()) &&
                ((a.x() < b.x()) ||
                 ((a.x() == b.x()) &&
                  ((a.height() > b.height()) ||
                   ((a.height() == b.height()) && (a.width() > b.width()))))));
    }
}; /// struct PositionComparator

} /// namespace detail

/// Bottom-Left Fill
int blf(std::vector<std::pair<Rectangle, int> > & rectangles,
        int maxWidth, int maxHeight);


/// The Detection class
template <typename T>
class Detection_ : public Rectangle_<T> {
  public:
    /// Constructs an empty Detection
    Detection_() : c_(-1), l_(-1), x_(-1), y_(-1), score_(-10.0F), ptIdx_(-1),
        ptNodeIdx_(-1) {}

    /// Constructs a Detection with @p score and @p bndbox
    Detection_(Scalar score, const Rectangle_<T> & bndbox) : c_(-1), l_(-1),
        x_(-1), y_(-1), score_(score), Rectangle_<T>(bndbox), ptIdx_(-1),
        ptNodeIdx_(-1) {}

    /// Constructs a Detection with @p l, @p x, @p y and @p score
    Detection_(int l, int x, int y, Scalar score) : c_(-1), l_(l), x_(x), y_(y),
        score_(score), ptIdx_(-1), ptNodeIdx_(-1) {}

    /// Constructs a Detection with @p c, @p score and @p bndbox
    Detection_(int c, Scalar score, const Rectangle_<T> & bndbox) : c_(c),
        l_(-1), x_(-1), y_(-1), score_(score), Rectangle_<T>(bndbox),
        ptIdx_(-1), ptNodeIdx_(-1) {}

    /// Constructs a Detection with full specification except for pt idx
    Detection_(int c, int l, int x, int y, Scalar score,
               const Rectangle_<T> & bndbox)  : c_(c), l_(l), x_(x), y_(y),
        score_(score), Rectangle_<T>(bndbox), ptIdx_(-1), ptNodeIdx_(-1) {}

    /// Constructs a Detection with full specification
    Detection_(int c, int l, int x, int y, Scalar score,
               const Rectangle_<T> & bndbox, int ptIdx, int ptNodeIdx)
        : c_(c), l_(l), x_(x), y_(y), score_(score), Rectangle_<T>(bndbox),
          ptIdx_(ptIdx), ptNodeIdx_(ptNodeIdx) {}

    /// Copy constructor
    Detection_(const Detection_<T> & detection) : c_(detection.c_),
        l_(detection.l_), x_(detection.x_), y_(detection.y_),
        score_(detection.score_), Rectangle_<T>(detection),
        ptIdx_(detection.ptIdx_), ptNodeIdx_(detection.ptNodeIdx_) {}

    /// Compares the scores in decreasing order
    bool operator<(const Detection_<T> & detection) const {
        return score_ > detection.score_; // for decreasing sort
    }

    /// clip the bbox
    bool clipBbox(int wd, int ht);

    /// show
    void show(Mat img, bool display);

    int    c_; //
    int    l_;
    int    x_;
    int    y_;
    Scalar score_;

    int ptIdx_;
    int ptNodeIdx_; // usually for single object And PtNode

    DEFINE_SERIALIZATION;

}; // class Detection


/// Functor used to test for the intersection of two rectangles
/// according to the Pascal criterion (area of intersection over area of union).
template <typename T>
class Intersector_ {
  public:
    /// Constructor.
    /// @param[in] reference The reference rectangle.
    /// @param[in] threshold The threshold of the criterion.
    /// @param[in] dividedByUnion Use Felzenszwalb's criterion instead
    /// (area of intersection over area of second rectangle).
    /// Useful to remove small detections inside bigger ones.
    Intersector_(const Rectangle_<T> & reference, float threshold = 0.5,
                 bool dividedByUnion = false,  bool dividedByRef = false);

    /// Tests for the intersection between a given rectangle and the reference.
    /// @param[in] rect The rectangle to intersect with the reference.
    /// @param[out] score The score of the intersection.
    bool operator()(const Rectangle_<T> & rect, float * score = 0) const;

  private:
    const Rectangle_<T> & reference_;
    float threshold_;
    bool   dividedByUnion_;
    bool dividedByRef_;

}; /// class Intersector_


/// The configuration class
template <typename T>
class Configuration_ {
  public:
    /// Constructs an empty configuration
    Configuration_();

    /// Clip root bbox
    bool clipBbox(int wd, int ht);

    /// Show
    void show(Mat img);

    /// Compares the scores in decreasing order
    bool operator<(const Configuration_<T> & config) const {
        return score_ > config.score_; // for decreasing sort
    }

    int c_;
    Scalar score_;
    int l_; /// the pyramid level in which the detection is declared
    Scalar scale_; /// the resizing factor

    /// detection windows at original image resolution
    std::vector<Rectangle_<T> > parts_;

    DEFINE_SERIALIZATION;

}; /// class Configuration_

typedef Configuration_<Scalar> Configuration;


/// Functor used to test for the intersection of two Detection
/// according to the Pascal criterion (area of intersection over area of union).
template <typename T>
class ConfigIntersector_ {
  public:
    /// Constructor.
    /// @param[in] reference The reference rectangle.
    /// @param[in] threshold The threshold of the criterion.
    /// @param[in] dividedByUnion Use Felzenszwalb's criterion instead
    /// (area of intersection over area of second rectangle).
    /// Useful to remove small detections inside bigger ones.
    ConfigIntersector_(const Configuration_<T> & reference,
                       Scalar threshold = 0.5, bool dividedByUnion = false);

    /// Tests for the intersection between a given rectangle and the reference.
    /// @param[in] rect The rectangle to intersect with the reference.
    /// @param[out] score The score of the intersection.
    bool operator()(const Configuration_<T> & config, Scalar * score = 0) const;

  private:
    const Configuration_<T> reference_;
    Scalar threshold_;
    bool   dividedByUnion_;
};

/// The ParseInfo class
class ParseInfo : public Rectangle_<Scalar> {
  public:
    /// Default constructor
    ParseInfo();

    /// Constructs a parse info with given arguments
    explicit ParseInfo(int c, int l, int x, int y, int ds, int dx, int dy,
                       Scalar score, const Rectangle_<Scalar> & bbox,
                       Scalar loss = 0, Scalar goodness = 0);

    /// Copy constructor
    explicit ParseInfo(const ParseInfo & info);

    /// clip the bbox
    bool clipBbox(int wd, int ht, bool change = true);

    int c_;
    int l_;
    int x_;
    int y_;
    int ds_;
    int dx_;
    int dy_;

    Scalar score_;
    Scalar loss_;
    Scalar goodness_;

    DEFINE_SERIALIZATION;

}; // class ParseNode

} /// namespace RGM


#endif /// RGM_RECTANGLE_HPP_



