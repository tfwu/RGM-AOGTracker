#include "command_line.hpp"
#include "tracker.hpp"

#include "util/UtilOpencv.hpp"

namespace RGM {

const static char * k_date = __DATE__;
const static char * k_time = __TIME__;
const static char * k_appname = "RGM";

// ---------- CommandLineInterpreter -----------

CommandLineInterpreter::CommandLineInterpreter(int argc, char ** argv) :
    _argc(argc), _argv(argv) {
}

void CommandLineInterpreter::AddCommand(ICommand * pCommand) {
	_commands.push_back(pCommand);
}

int CommandLineInterpreter::Process() {
    if(_argc < 2) {
		PrintHelp();
		return 1;
	}

	int argCount = _argc - 2;
	string command = _argv[1];
	ICommand * pCommand = NULL;
    for(int i = 0; i < _commands.size(); i++) {
        if((_commands[i]->GetCommand().compare(command) == 0)
                && (_commands[i]->GetArgumentCount() == argCount)) {
			pCommand = _commands[i];
			break;
		}
	}

    if(pCommand == NULL) {
        std::cout << "Unknown command: " << command << std::endl;
		PrintHelp();
		return 1;
	}

    if(pCommand->GetArgumentCount() != argCount) {
        std::cout << "Wrong number of arguments for command: " << command << std::endl;
		PrintHelp();
		return 1;
	}

    if(pCommand->Run(this)) {
		return 0;
	}

	return 1;
}

void CommandLineInterpreter::PrintHelp() {
	printf("%s [built %s %s]\n\nCommands:\n", k_appname, k_date, k_time);
    for(int i = 0; i < _commands.size(); i++) {
		printf("%s ", _commands[i]->GetCommand().c_str());
        for(int j = 0; j < _commands[i]->GetArgumentCount(); j++) {
			printf("<%s> ", _commands[i]->GetArgumentName(j).c_str());
		}
		printf("\n");
	}
}

int CommandLineInterpreter::GetInt(int index) {
	return boost::lexical_cast<int>(_argv[index + 2]);
}

float CommandLineInterpreter::GetFloat(int index) {
	return boost::lexical_cast<float>(_argv[index + 2]);
}

string CommandLineInterpreter::GetString(int index) {
	return _argv[index + 2];
}


// ------ Commands: AOGTracker ------

std::string AOGTrackerCommand::GetCommand() {
    return "Tracking";
}

int AOGTrackerCommand::GetArgumentCount() {
    return 1;
}

std::string AOGTrackerCommand::GetArgumentName(int index) {
    switch(index) {
        case 0:
            return "configfile";
    }
    return "";
}

bool AOGTrackerCommand::Run(CommandLineInterpreter *pInterpreter) {
    // Get the config. xml file
    std::string configFileName = pInterpreter->GetString(0);

    RunAOGTracker(configFileName);
}

} // namespace RGM

