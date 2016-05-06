#ifndef RGM_UTILMATH_HPP_
#define RGM_UTILMATH_HPP_

#include <math.h>

#include "common.hpp"

namespace RGM {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template<typename T>
class MathUtil_
{

public:
    static T median(vector<T> & vData, int num=numeric_limits<int>::max(),
                    bool sortData=false);
    static vector<float> pdist(const vector<cv::Point_<T> > & pts,
                               int num=numeric_limits<int>::max());
    static vector<T>     linspace(T s, T e, T interval);
    static vector<T>     hist(vector<T> & y, vector<T> & x);
    static vector<T>     convnSame(vector<T> & x, vector<T> & filter);

    static T calcErr(vector<T> & pscores, vector<T> & nscores, T & thr) ;
    static T calcVar(vector<T> & pscores, T & thr) ;
    static void calcMeanStd(vector<T> & pscores, T & m, T & s);
}; // MathUtil_

} // namespace RGM

#endif // RGM_UTILMATH_HPP_
