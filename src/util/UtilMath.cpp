#include <limits>
#include <numeric>

#include <Eigen/Core>

#include "util/UtilMath.hpp"

namespace RGM
{

using namespace std;

template<typename T>
T MathUtil_<T>::median(vector<T> & vData, int num, bool sortData)
{
    if (vData.size()==0 || num==0) {
        return numeric_limits<T>::quiet_NaN();
    }

    num = max<int>(min<int>(vData.size(), num), 1);

    if ( vData.size()==1 || num==1 ) {
        return vData[0];
    }

    if ( sortData ) {
        sort(vData.begin(), vData.begin()+num); // Ascending

        if ( num % 2 == 0) {
            int idx = num / 2;
            return (vData[idx-1] + vData[idx]) / (T)2;
        } else {
            int idx = (num-1) / 2;
            return vData[idx];
        }
    } else {
        vector<T> tmp(num);
        copy(vData.begin(), vData.begin()+num, tmp.begin());
        sort(tmp.begin(), tmp.end());

        if ( num % 2 == 0) {
            int idx = num / 2;
            return (tmp[idx-1] + tmp[idx]) / (T)2;
        } else {
            int idx = (num-1) / 2;
            return tmp[idx];
        }
    }
}


template<typename T>
vector<float> MathUtil_<T>::pdist(const vector<cv::Point_<T> > & pts, int num)
{

    vector<float> dist;

    num = min<int>(pts.size(), num);

    for ( int i=0; i<num-1; ++i ) {
        const cv::Point_<T> & pt1( pts[i] );
        for ( int j=i+1; j<num; ++j ) {
            const cv::Point_<T> & pt2( pts[j] );
            dist.push_back( sqrt((float)(pt1.x-pt2.x) * (pt1.x-pt2.x) +
                                 (pt1.y-pt2.y) * (pt1.y-pt2.y)) );
        }
    }

    return dist;
}

template<typename T>
vector<T> MathUtil_<T>::linspace(T s, T e, T interval)
{

    vector<T> x;

    for ( T i=s; i<=e; i+=interval ) {
        x.push_back(i);
    }

    return x;
}


template<typename T>
vector<T> MathUtil_<T>::hist(vector<T> & y, vector<T> & x)
{

    vector<T> n(x.size(), 0);

    vector<T> xx(x.size()+1);
    xx[0]           = -numeric_limits<T>::infinity();
    copy(x.begin(), x.end(), xx.begin()+1);

    for ( int i=0; i<y.size(); ++i ) {

        int j=0;
        for ( ; j<xx.size()-1; ++j ) {
            if (y[i]>xx[j] && y[i]<=xx[j+1]) {
                break;
            }
        }

        n[j]++;
    }

    return n;
}

template<typename T>
vector<T> MathUtil_<T>::convnSame(vector<T> & x, vector<T> & filter)
{

    vector<T> xx(x.size()+2*filter.size(), 0);
    copy(x.begin(), x.end(), xx.begin()+filter.size());

    reverse(filter.begin(), filter.end());

    vector<T> result(xx.size()-filter.size());
    for ( int i=0; i<result.size(); ++i) {
        result[i] = inner_product(filter.begin(), filter.end(), xx.begin()+i,
                                  0.0F);
    }

    vector<T> r(x.size());

    int istart = floor(filter.size()/2.0F);
    copy(result.begin()+istart, result.begin()+istart+x.size(), r.begin());

    return r;
}

template<typename T>
T MathUtil_<T>::calcErr(vector<T> & pscores, vector<T> & nscores, T & thr)
{
    T minerr = 1.0;

    int numpos = pscores.size();
    int numneg = nscores.size();
    int num = numpos+numneg;

    vector<pair<T, int> > scores(num);

    for ( int i=0; i<numpos; ++i ) {
        scores[i] = pair<T, int>(-pscores[i], i);
    }
    for ( int i=numpos, j=0; j<numneg; ++i, ++j ) {
        scores[i] = pair<T, int>(-nscores[j], i);
    }

    sort(scores.begin(), scores.end());

    vector<int> tp(num, 0), fp(num, 0);

    for (int i=0; i<num; ++i ) {
        if ( scores[i].second < numpos ) {
            tp[i] = 1;
            fp[i] = 0;
        } else {
            fp[i] = 1;
            tp[i] = 0;
        }
    }

    Eigen::VectorXd tpr(num), fpr(num), err(num);

    tpr(0) = tp[0];
    fpr(0) = fp[0];
    for ( int i=1; i<num; ++i ) {
        tpr(i) = tpr(i-1) + tp[i];
        fpr(i) = fpr(i-1) + fp[i];
    }

    tpr /= numpos;
    // to fnr
    tpr.array() = 1 - tpr.array();
    fpr /= numneg;

    err = tpr+fpr;

    int r, c;
    minerr = err.minCoeff(&r, &c) / 2.0f;

    thr = max<T>(-scores[r].first, *min_element(pscores.begin(), pscores.end()));

    return minerr;
}

template<typename T>
T MathUtil_<T>::calcVar(vector<T> & pscores, T & thr)
{
    int num = pscores.size();

    //thr = numeric_limits<T>::max();

    T  m = 0;
    for ( int i=0; i<num; ++i) {
        m += pscores[i];
        //thr = min<T>(thr, pscores[i]);
    }

    m /= num;

    T v = 0;
    for ( int i=0; i<num; ++i) {
        v += pow(pscores[i]-m, 2.0F);
    }

    if ( num > 1) {
        v /= (num-1);
        v = sqrt(v);
    }

    thr = m - v;

    return v;
}

template<typename T>
void MathUtil_<T>::calcMeanStd(vector<T> & pscores, T & m, T & s) {
    int num = pscores.size();    

    m = 0;
    s = 0;
    if ( num == 0 ) return;

    for ( int i=0; i<num; ++i) {
        m += pscores[i];        
    }
    m /= num;

    for ( int i=0; i<num; ++i) {
        s += pow(pscores[i]-m, 2.0F);
    }
    if ( num > 1) {
        s /= (num-1);        
        s = sqrt(s);
    }    
}

/// Specification
template class MathUtil_<int>;
template class MathUtil_<float>;
template class MathUtil_<double>;

} // namespace RGM
