#ifndef RGM_COMMAND_LINE_HPP_
#define RGM_COMMAND_LINE_HPP_

#include "common.hpp"
#include "util/UtilFile.hpp"

namespace RGM {

/// Predeclaration
class CommandLineInterpreter;

/// Virtual class for defining commands
class ICommand {
  public:
	virtual string GetCommand() = 0;
	virtual int GetArgumentCount() = 0;
	virtual string GetArgumentName(int index) = 0;
	virtual bool Run(CommandLineInterpreter * pInterpreter) = 0;
  protected:
	ICommand() {
	}

};
//class ICommand

/// Interpret commands
class CommandLineInterpreter {
  public:
	CommandLineInterpreter(int argc, char ** argv);

	void AddCommand(ICommand * pCommand);
	int Process();

	int GetInt(int index);
	float GetFloat(int index);
	string GetString(int index);

  private:
	void PrintHelp();

	std::vector<ICommand *> _commands;
	int _argc;
	char ** _argv;

};
//class CommandLineInterpreter


/// Command for Tracking
class AOGTrackerCommand : public ICommand {
  public:
    std::string GetCommand();

    int GetArgumentCount();

    std::string GetArgumentName(int index);

    bool Run(CommandLineInterpreter* pInterpreter);
};


} // namespace RGM

#endif // RGM_COMMAND_LINE_HPP_
