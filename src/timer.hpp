#ifndef RGM_TIMER_HPP_
#define RGM_TIMER_HPP_

#include <ctime>
#include "common.hpp"

#if (defined(WIN32)  || defined(_WIN32) || defined(WIN64) || defined(_WIN64))
#define TIME( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#else
#define TIME( arg ) (time( arg ))
#endif


namespace RGM
{

/// Timer
class Timers
{
public:
    struct Task {
        Task(string name) :
            _name(name), _cumulative_time(0), _offset(0), _next(NULL) {}

        void Start() { _offset = TIME(0); }
        void Stop() { _cumulative_time += TIME(0) - _offset; }
        void Reset() { _cumulative_time = 0; }
        double ElapsedSeconds() { return _cumulative_time; }

        string _name;
        Task* _next;
        double _offset;
        double _cumulative_time;
    };

    Timers() :
        _head(NULL), _tail(NULL) {}

    ~Timers() { clear(); }

    Task* operator()(string name);
    void showUsage();
    void clear();
private:
    Task* _head;
    Task* _tail;    

    DEFINE_RGM_LOGGER;
};

} //namespace RGM

#endif // RGM_TIMER_HPP_
