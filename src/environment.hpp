/// This file is adapted from UCLA ImageParser
/// by Brandon Rothrock (rothrock@cs.ucla.edu)

#ifndef RGM_ENVIRONMENT_HPP_
#define RGM_ENVIRONMENT_HPP_

#include "timer.hpp"

namespace RGM
{
/// Predeclaration
class XMLData;

/// Environment for the application
class Environment
{
public:

    Environment();
    ~Environment();

    bool InitFromConfigFile(string & configFile);
    bool InitFromConfigString(char * str);

    // Resources
    const string&   getConfigFileName() const { return _configFileName; }
    XMLData*		GetConfig() { return _pConfig; }
    Timers&			GetTimers() { return *_pTimers; }

private:
    string _configFileName;

    XMLData*    _pConfig;
    Timers*     _pTimers;

    DEFINE_RGM_LOGGER;

}; //class Environment

} // namespace RGM

#endif // RGM_ENVIRONMENT_HPP_


