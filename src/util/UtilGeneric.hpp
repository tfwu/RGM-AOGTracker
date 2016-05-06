#ifndef RGM_UTILGENERIC_HPP_
#define RGM_UTILGENERIC_HPP_

#include "common.hpp"

namespace RGM {

namespace detail {

template<typename T>
struct pair_less_than_ {
    template<typename T1>
    bool operator()(const pair<T, T1> &a, const pair<T, T1> &b)
    {
        return a.first < b.first;
    }
};

template<typename T>
struct pair_larger_than_ {
    template<typename T1>
    bool operator()(const pair<T, T1> &a, const pair<T, T1> &b)
    {
        return a.first > b.first;
    }
};

template<typename T>
struct equal_to_ {
    bool operator()(T i, T j)
    {
        return (i == j);
    }
};

} // namespace detail

// Fci - forward const iterator
template<typename Fci>
vector<vector<Fci> > enumerateCombinations_(Fci begin, Fci end,
                                            unsigned int combination_size);

template<typename T>
vector<T> & operator+=(vector<T> & a, const vector<T> & b);

template <typename T>
void uniqueVector_(vector<T> & vec);

template <typename T>
vector<int> sortVector_(vector<T> & vData, bool ascending = false,
                        bool bSorted =false );

template <typename T>
string vec2string_(const vector<T> & a);

} // namespace RGM

#endif // RGM_UTILGENERIC_HPP_
