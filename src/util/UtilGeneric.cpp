#include "util/UtilGeneric.hpp"

namespace RGM {

using namespace  std;

template<typename Fci>
vector<vector<Fci> > enumerateCombinations_(Fci begin, Fci end,
                                            unsigned int combination_size)
{
    // empty set of combinations
    vector<vector<Fci> > result;

    // there is exactly one combination of size 0 - empty set
    if (combination_size == 0u) {
        return result;
    }

    vector<Fci> current_combination;
    current_combination.reserve(combination_size + 1u); // one additional slot

    // in my vector to store the end sentinel there.
    // Construction of the first combination
    for (unsigned int i = 0u; i < combination_size && begin != end;
         ++i, ++begin) {
        current_combination.push_back(begin);
    }

    // Since I assume the iterators support only incrementing, I have to iterate over
    // the set to get its size, which is expensive. Here I had to iterate anyway to
    // produce the first combination, so I use the loop to also check the size.
    assert(current_combination.size() >= combination_size);

    // Store the first combination in the results set
    result.push_back(current_combination);
    // Here I add mentioned earlier sentinel to
    current_combination.push_back(end);
    // simplify rest of the code. If I did it
    // earlier, previous statement would get ugly.
    while (true) {
        unsigned int i = combination_size;
        Fci tmp;                                        // Thanks to the sentinel I can find first
        do {                                            // iterator to change, simply by scaning
            // from right to left and looking for the
            tmp = current_combination[--i];             // first "bubble". The fact, that it's
            ++tmp;                                      // a forward iterator makes it ugly but I
        }                                               // can't help it.
        while (i > 0u && tmp == current_combination[i + 1u]);

        // Here is probably my most obfuscated expression.
        // Loop above looks for a "bubble". If there is no "bubble", that means, that
        // current_combination is the last combination, Expression in the if statement
        // below evaluates to true and the function exits returning result.
        // If the "bubble" is found however, the statement below has a side effect of
        // incrementing the first iterator to the left of the "bubble".
        if (++current_combination[i] == current_combination[i + 1u]) {
            return result;
        }
        // Rest of the code sets positions of the rest of the iterators
        // (if there are any), that are to the right of the incremented one,
        // to form next combination

        while (++i < combination_size) {
            current_combination[i] = current_combination[i - 1u];
            ++current_combination[i];
        }
        // Below is the ugly side of using the sentinel. Well it had to have some
        // disadvantage. Try without it.
        result.push_back(
                    vector<Fci>(current_combination.begin(),
                                current_combination.end() - 1));
    }
}

/// Specification
template vector<vector<vector<int>::const_iterator > > enumerateCombinations_(
        vector<int>::const_iterator begin, vector<int>::const_iterator end,
        unsigned int combination_size);

template<typename T>
vector<T> & operator+=(vector<T> & a, const vector<T> & b)
{
    assert(a.size() ==  b.size());
    if (a.size() == 0) {
        return a;
    }

    vector<T> result;
    result.reserve(a.size());

    transform(a.begin(), a.end(), b.begin(), back_inserter(result), plus<T>());

    a.swap(result);

    return a;
}

template vector<int> & operator+=(vector<int>& a, const vector<int>& b);
template vector<float> & operator+=(vector<float>& a, const vector<float>& b);
template vector<double> & operator+=(vector<double>& a, const vector<double>& b);

template <typename T>
void uniqueVector_(vector<T> & vec)
{
    sort(vec.begin(), vec.end());
    typename vector<T>::iterator it = unique(vec.begin(), vec.end(),
                                             detail::equal_to_<T>());
    vec.resize(distance(vec.begin(), it));
}

template void uniqueVector_<int>(vector<int> & vec);
template void uniqueVector_<unsigned int>(vector<unsigned int> & vec);
template void uniqueVector_<float>(vector<float> & vec);
template void uniqueVector_<double>(vector<double> & vec);

template <typename T>
vector<int> sortVector_(vector<T> & vData, bool ascending, bool bSorted)
{
    int num = (int)vData.size();
    assert(num>0);

    vector<pair<T, int> > vTemp(num);
    for ( int i=0; i<num; ++i )	{
        vTemp[i] = pair<T, int>(vData[i], i);
    }

    if ( ascending )
        sort(vTemp.begin(), vTemp.end(), detail::pair_less_than_<T>());
    else
        sort(vTemp.begin(), vTemp.end(), detail::pair_larger_than_<T>());

    vector<int> vIndex(num);
    for ( int i=0; i<num; ++i ) {
        vIndex[i] = vTemp[i].second;

        if ( bSorted )
            vData[i] = vTemp[i].first;
    }

    return vIndex;
}

template vector<int> sortVector_(vector<int> & vData, bool ascending,
                                 bool bSorted);
template vector<int> sortVector_(vector<float> & vData, bool ascending,
                                 bool bSorted);
template vector<int> sortVector_(vector<double> & vData, bool ascending,
                                 bool bSorted);

template <typename T>
string vec2string_(const vector<T> & a)
{
    boost::format frmt("%1%");

    string str;

    for ( int i = 0; i < a.size(); ++i ) {
        frmt % a[i];
        str += frmt.str();
    }

    return str;
}

template string vec2string_(const vector<bool> & a);
template string vec2string_(const vector<int> & a);

} // namespace RGM
