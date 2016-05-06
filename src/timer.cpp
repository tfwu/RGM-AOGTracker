#include "timer.hpp"

namespace RGM
{

// -------- Timers -------

Timers::Task* Timers::operator()(string name)
{
    Task* t = NULL;
    if (_head == NULL) {
        t = new Task(name);
        _head = t;
        _tail = t;
    } else {
        Task* cursor = _head;
        while (cursor != NULL) {
            if (cursor->_name.compare(name) == 0) {
                t = cursor;
                break;
            }
            cursor = cursor->_next;
        }
        if (t == NULL) {
            t = new Task(name);
            _tail->_next = t;
            _tail = t;
        }
    }
    return t;
}

void Timers::showUsage()
{
    boost::format msg("    %s : %f");
    std::ostringstream oss;
    oss << "  Time usage: ";
    Task* cursor = _head;
    while (cursor != NULL) {
        double t = cursor->ElapsedSeconds();
        msg % cursor->_name % t;
        oss << msg.str() << std::endl;
        Task* next = cursor->_next;
        cursor = next;
    }

    RGM_LOG(normal, oss.str());
}


void Timers::clear()
{
    Task* cursor = _head;
    while (cursor != NULL) {
        Task* next = cursor->_next;
        delete cursor;
        cursor = next;
    }

    _head = NULL;
    _tail = NULL;

}

} // namespace RGM


