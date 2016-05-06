#include "rectangle.hpp"

namespace RGM {

// ------- Rectangle_ ------

template <typename T>
std::vector<Rectangle_<T> > Rectangle_<T>::partition(Rectangle_<T> & rect) {
    std::vector<Rectangle_<T> > bb;

    // Not overlap with rect
    T x1 = std::max<T>(left(), rect.left());
    T x2 = std::min<T>(right(), rect.right());
    if(x1 > x2) {
        bb.push_back(*this);
        bb.push_back(rect);
        return bb;
    }

    T y1 = std::max<T>(top(), rect.top());
    T y2 = std::min<T>(bottom(), rect.bottom());
    if(y1 > y2) {
        bb.push_back(*this);
        bb.push_back(rect);
        return bb;
    }

    // Overlap with rect
    Rectangle_<T> overlap(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    bb.push_back(overlap);
    bb.push_back(overlap);

    if(left() == rect.left() && right() == rect.right()) {
        // Hor cut
        if(top() < rect.top()) {
            bb.push_back(Rectangle_<T>(x1, top(),    x2 - x1 + 1, y1 - top() + 1));
            bb.push_back(Rectangle_<T>(x1, bottom(), x2 - x1 + 1,
                                       rect.bottom() - bottom() + 1));
        } else {
            bb.push_back(Rectangle_<T>(x1, rect.top(),    x2 - x1 + 1,
                                       y1 - rect.top() + 1));
            bb.push_back(Rectangle_<T>(x1, rect.bottom(), x2 - x1 + 1,
                                       bottom() - rect.bottom() + 1));
        }
    } else {
        // Ver cut
        if(left() < rect.left()) {
            bb.push_back(Rectangle_<T>(left(),  y1, x1 - left() + 1,
                                       y2 - y1 + 1));
            bb.push_back(Rectangle_<T>(right(), y1, rect.right() - right() + 1,
                                       y2 - y1 + 1));
        } else {
            bb.push_back(Rectangle_<T>(rect.left(),  y1, x1 - rect.left() + 1,
                                       y2 - y1 + 1));
            bb.push_back(Rectangle_<T>(rect.right(), y1,
                                       right() - rect.right() + 1, y2 - y1 + 1));
        }
    }

    return bb;
}


template <typename T>
Rectangle_<T> Rectangle_<T>::expand(T num, T * maxWd, T * maxHt) {
    Rectangle_<T> displacement = *this;

    x_ = std::max<T>(0, x_ - num);
    y_ = std::max<T>(0, y_ - num);
    width_ += num * 2;
    height_ += num * 2;
    if(maxWd != NULL && right() >= *maxWd) {
        width_ -= (right() - *maxWd + 1);
    }
    if(maxHt != NULL && bottom() >= *maxHt) {
        height_ -= (bottom() - *maxHt + 1);
    }

    displacement.setX(x() - displacement.x());
    displacement.setY(y() - displacement.y());
    displacement.setWidth(displacement.width() - width());
    displacement.setHeight(displacement.height() - height());

    return displacement;
}

template<typename T>
bool Rectangle_<T>::clip(int wd, int ht) {
    T x1                = max<T>(0, x());
    T y1                = max<T>(0, y());
    T x2                = min<T>(wd - 1, right());
    T y2                = min<T>(ht - 1, bottom());
    T width             = x2 - x1 + 1;
    T height            = y2 - y1 + 1;

    if(width <= 0 || height <= 0) {
        return false;
    }

    setX(x1);
    setY(y1);
    setWidth(width);
    setHeight(height);

    return true;
}

template <typename T>
template <class Archive>
void Rectangle_<T>::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(x_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(width_);
    ar & BOOST_SERIALIZATION_NVP(height_);
}

/// Specification
template class Rectangle_<int>;
template class Rectangle_<Scalar>;

INSTANTIATE_BOOST_SERIALIZATION(Rectangle_<int>);
INSTANTIATE_BOOST_SERIALIZATION(Rectangle_<Scalar>);


Rectangles getOverlappedRectangles(Rectangles & rects1,  Rectangles & rects2) {
    Rectangles bb;

    for(int i = 0; i < rects2.size(); ++i) {
        Rectangle & rect2(rects2[i]);
        for(int j = 0; j < rects1.size(); ++j) {
            Rectangle & rect1(rects1[j]);

            int x1 = max<int>(rect1.left(), rect2.left());
            int y1 = max<int>(rect1.top(), rect2.top());
            int x2 = min<int>(rect1.right(), rect2.right());
            int y2 = min<int>(rect1.bottom(), rect2.bottom());

            if(x1 > x2 || y1 > y2) {
                continue;
            }

            bb.push_back(Rectangle(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
        }
    }

    return bb;
}

int blf(vector<pair<Rectangle, int> > & rectangles, int maxWidth,
        int maxHeight) {
    /// Order the rectangles by decreasing area.
    /// If a rectangle is bigger than MaxRows x MaxCols returns -1
    vector<int> ordering(rectangles.size());

    for(int i = 0; i < rectangles.size(); ++i) {
        if((rectangles[i].first.width() > maxWidth) ||
                (rectangles[i].first.height() > maxHeight)) {
            return -1;
        }

        ordering[i] = i;
    }

    sort(ordering.begin(), ordering.end(), detail::AreaComparator(rectangles));

    // Index of the plane containing each rectangle
    for(int i = 0; i < rectangles.size(); ++i) {
        rectangles[i].second = -1;
    }

    vector<set<Rectangle, detail::PositionComparator> > gaps;

    // Insert each rectangle in the first gap big enough
    for(int i = 0; i < rectangles.size(); ++i) {
        pair<Rectangle, int> & rect = rectangles[ordering[i]];

        // Find the first gap big enough
        set<Rectangle, detail::PositionComparator>::iterator g;

        for(int i = 0; (rect.second == -1) && (i < gaps.size()); ++i) {
            for(g = gaps[i].begin(); g != gaps[i].end(); ++g) {
                if((g->width() >= rect.first.width()) &&
                        (g->height() >= rect.first.height())) {
                    rect.second = i;
                    break;
                }
            }
        }

        // If no gap big enough was found, add a new plane
        if(rect.second == -1) {
            set<Rectangle, detail::PositionComparator> plane;
            // The whole plane is free
            plane.insert(Rectangle(maxWidth, maxHeight));
            gaps.push_back(plane);
            g = gaps.back().begin();
            rect.second = static_cast<int>(gaps.size()) - 1;
        }

        // Insert the rectangle in the gap
        rect.first.setX(g->x());
        rect.first.setY(g->y());

        // Remove all the intersecting gaps, and add newly created gaps
        for(g = gaps[rect.second].begin(); g != gaps[rect.second].end();) {
            if(!((rect.first.right() < g->left()) ||
                    (rect.first.bottom() < g->top()) ||
                    (rect.first.left() > g->right()) ||
                    (rect.first.top() > g->bottom()))) {
                // Add a gap to the left of the new rectangle if possible
                if(g->x() < rect.first.x())
                    gaps[rect.second].insert(Rectangle(g->x(), g->y(),
                                                       rect.first.x() - g->x(),
                                                       g->height()));

                // Add a gap on top of the new rectangle if possible
                if(g->y() < rect.first.y())
                    gaps[rect.second].insert(Rectangle(g->x(), g->y(),
                                                       g->width(),
                                                       rect.first.y() - g->y()));

                // Add a gap to the right of the new rectangle if possible
                if(g->right() > rect.first.right())
                    gaps[rect.second].insert(
                        Rectangle(rect.first.right() + 1, g->y(),
                                  g->right() - rect.first.right(),
                                  g->height()));

                // Add a gap below the new rectangle if possible
                if(g->bottom() > rect.first.bottom())
                    gaps[rect.second].insert(
                        Rectangle(g->x(), rect.first.bottom() + 1,
                                  g->width(),
                                  g->bottom() - rect.first.bottom()));

                // Remove the intersecting gap
                gaps[rect.second].erase(g++);
            } else {
                ++g;
            }
        }
    }

    return static_cast<int>(gaps.size());
}


// ------- Detection --------

template<typename T>
bool Detection_<T>::clipBbox(int wd, int ht) {
    T x1                = max<T>(0, this->x());
    T y1                = max<T>(0, this->y());
    T x2                = min<T>(wd - 1, this->right());
    T y2                = min<T>(ht - 1, this->bottom());
    T width             = x2 - x1 + 1;
    T height            = y2 - y1 + 1;

    if(width <= 0 || height <= 0) {
        return false;
    }

    this->setX(x1);
    this->setY(y1);
    this->setWidth(width);
    this->setHeight(height);

    return true;
}

template<typename T>
void Detection_<T>::show(Mat img, bool display) {
    cv::rectangle(img, this->cvRect(), cv::Scalar::all(255), 5);
    cv::rectangle(img, this->cvRect(), cv::Scalar(0, 0, 255), 3);

    if(display) {
        cv::String winName("Detection");
        cv::imshow(winName, img);
        cv::waitKey(0);
    }
}

template<typename T>
template<class Archive>
void Detection_<T>::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Rectangle_<T>);
    ar & BOOST_SERIALIZATION_NVP(c_);
    ar & BOOST_SERIALIZATION_NVP(l_);
    ar & BOOST_SERIALIZATION_NVP(x_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(score_);
    ar & BOOST_SERIALIZATION_NVP(ptIdx_);
    ar & BOOST_SERIALIZATION_NVP(ptNodeIdx_);
}

/// Specification
template class Detection_<int>;
template class Detection_<Scalar>;

INSTANTIATE_BOOST_SERIALIZATION(Detection_<int>);
INSTANTIATE_BOOST_SERIALIZATION(Detection_<Scalar>);


// ------- Intersector ------

template <typename T>
Intersector_<T>::Intersector_(const Rectangle_<T> & reference, float threshold,
                              bool dividedByUnion, bool dividedByRef)
    : reference_(reference), threshold_(threshold),
      dividedByUnion_(dividedByUnion), dividedByRef_(dividedByRef) {
}

template <typename T>
bool Intersector_<T>::operator()(const Rectangle_<T> & rect,
                                 float * score) const {
    if(score) {
        *score = 0.0;
    }

    const int left = max<int>(reference_.left(), rect.left());
    const int right = min<int>(reference_.right(), rect.right());

    if(right < left) {
        return false;
    }

    const int top = max<int>(reference_.top(), rect.top());
    const int bottom = min<int>(reference_.bottom(), rect.bottom());

    if(bottom < top) {
        return false;
    }

    const int intersectionArea = (right - left + 1) * (bottom - top + 1);
    const int rectArea = rect.area();

    if(dividedByUnion_) {
        const int referenceArea = reference_.area();
        const int unionArea = referenceArea + rectArea - intersectionArea;

        if(score) {
            *score = static_cast<float>(intersectionArea) / unionArea;
        }

        if(intersectionArea >= unionArea * threshold_) {
            return true;
        }

    } else {
        if(score) {
            *score = static_cast<float>(intersectionArea) /
                     (dividedByRef_ ? reference_.area() : rectArea);
        }

        if(intersectionArea >= rectArea * threshold_) {
            return true;
        }
    }

    return false;
}

/// Specification
template class Intersector_<int>;
template class Intersector_<Scalar>;


// ------ Configuration_ ------

template <typename T>
Configuration_<T>::Configuration_()
    : c_(-1), score_(-10.F), l_(-1), scale_(-1.F) {
}

template <typename T>
bool Configuration_<T>::clipBbox(int wd, int ht) {
    if(!parts_.size())
        return false;

    if(parts_[0].x() < 0)
        parts_[0].setX(0);
    if(parts_[0].y() < 0)
        parts_[0].setY(0);
    if(parts_[0].right() > wd)
        parts_[0].setWidth(wd - parts_[0].x() + 1);
    if(parts_[0].bottom() > ht)
        parts_[0].setHeight(ht - parts_[0].y() + 1);

    if(parts_[0].width() <= 0 || parts_[0].height() <= 0) {
        return false;
    }

    return true;
}

template <typename T>
void Configuration_<T>::show(Mat img) {
    for(int i = 1; i < parts_.size(); ++i) {
        cv::rectangle(img, parts_[i].cvRect(), cv::Scalar::all(255), 3);
        cv::rectangle(img, parts_[i].cvRect(), cv::Scalar(255, 0, 0), 2);
    }

    cv::rectangle(img, parts_[0].cvRect(), cv::Scalar::all(255), 5);
    cv::rectangle(img, parts_[0].cvRect(), cv::Scalar(0, 0, 255), 3);

    cv::imshow("Config", img);
    cv::waitKey();
}

template <typename T>
template<class Archive>
void Configuration_<T>::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(c_);
    ar & BOOST_SERIALIZATION_NVP(score_);
    ar & BOOST_SERIALIZATION_NVP(l_);
    ar & BOOST_SERIALIZATION_NVP(scale_);
    ar & BOOST_SERIALIZATION_NVP(parts_);
}

/// Specification
template class Configuration_<int>;
template class Configuration_<Scalar>;

INSTANTIATE_BOOST_SERIALIZATION(Configuration_<int>);
INSTANTIATE_BOOST_SERIALIZATION(Configuration_<Scalar>);


// ------ ConfigIntersector ------

template <typename T>
ConfigIntersector_<T>::ConfigIntersector_(const Configuration_<T> & reference,
                                          Scalar threshold, bool dividedByUnion)
    : reference_(reference), threshold_(threshold),
      dividedByUnion_(dividedByUnion) {
}

template <typename T>
bool ConfigIntersector_<T>::operator()(const Configuration_<T> & config,
                                       Scalar * score) const {
    if(score)
        *score = 0.0;

    if(reference_.parts_.size() == 0 || config.parts_.size() == 0)
        return false;

    const int left = max<int>(reference_.parts_[0].left(),
                              config.parts_[0].left());
    const int right = min<int>(reference_.parts_[0].right(),
                               config.parts_[0].right());

    if(right < left)
        return false;

    const int top = max<int>(reference_.parts_[0].top(),
                             config.parts_[0].top());
    const int bottom = min<int>(reference_.parts_[0].bottom(),
                                config.parts_[0].bottom());

    if(bottom < top)
        return false;

    const int intersectionArea = (right - left + 1) * (bottom - top + 1);
    const int rectArea = config.parts_[0].area();

    if(dividedByUnion_) {
        const int referenceArea = reference_.parts_[0].area();
        const int unionArea = referenceArea + rectArea - intersectionArea;

        if(score)
            *score = static_cast<Scalar>(intersectionArea) / unionArea;

        if(intersectionArea >= unionArea * threshold_) {
            return true;
        }
    } else {
        if(score)
            *score = static_cast<Scalar>(intersectionArea) / rectArea;

        if(intersectionArea >= rectArea * threshold_) {
            return true;
        }
    }

    return false;
}

/// Specification
template class ConfigIntersector_<int>;
template class ConfigIntersector_<Scalar>;



// ------ ParseInfo ------

ParseInfo::ParseInfo() :
    c_(-1), l_(-1), x_(-1), y_(-1), ds_(-1), dx_(-1), dy_(-1), score_(-10.0F),
    loss_(0.0F), goodness_(std::numeric_limits<Scalar>::infinity()) {
}

ParseInfo::ParseInfo(int c, int l, int x, int y, int ds, int dx, int dy,
                     Scalar score, const Rectangle_<Scalar> & bbox, Scalar loss,
                     Scalar goodness) :
    c_(c), l_(l), x_(x), y_(y), ds_(ds), dx_(dx), dy_(dy), score_(score),
    loss_(loss), goodness_(goodness),
    Rectangle_<Scalar>(bbox) {
}

ParseInfo::ParseInfo(const ParseInfo & info) :
    c_(info.c_), l_(info.l_), x_(info.x_), y_(info.y_),
    ds_(info.ds_), dx_(info.dx_), dy_(info.dy_), score_(info.score_),
    loss_(info.loss_), goodness_(info.goodness_),
    Rectangle_<Scalar>(info.x(), info.y(), info.width(), info.height()) {
}

bool ParseInfo::clipBbox(int wd, int ht, bool change) {
    float x1                = std::max<float>(0, x());
    float y1                = std::max<float>(0, y());
    float x2                = std::min<float>(wd - 1, right());
    float y2                = std::min<float>(ht - 1, bottom());
    float width             = x2 - x1 + 1;
    float height            = y2 - y1 + 1;

    if(width <= 0 || height <= 0) {
        return false;
    }

    if(change) {
        setX(x1);
        setY(y1);
        setWidth(width);
        setHeight(height);
    }

    return true;
}

template<class Archive>
void ParseInfo::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Rectangle_<Scalar>);
    ar & BOOST_SERIALIZATION_NVP(c_);
    ar & BOOST_SERIALIZATION_NVP(l_);
    ar & BOOST_SERIALIZATION_NVP(x_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(ds_);
    ar & BOOST_SERIALIZATION_NVP(dx_);
    ar & BOOST_SERIALIZATION_NVP(dy_);
    ar & BOOST_SERIALIZATION_NVP(score_);
    ar & BOOST_SERIALIZATION_NVP(loss_);
    ar & BOOST_SERIALIZATION_NVP(goodness_);
}

INSTANTIATE_BOOST_SERIALIZATION(ParseInfo);


} /// namespace RGM
