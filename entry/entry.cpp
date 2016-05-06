#include "command_line.hpp"
#include "util/UtilLog.hpp"

using namespace RGM;

int main(int argc, char** argv) {

#ifdef RGM_USE_MPI
    // Initialize the MPI environment
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided < MPI_THREAD_MULTIPLE) {
        std::cout << "ERROR: The MPI library does not have full thread support"
                  << std::endl ; fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get the number of processes
    int ntasks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // Get the rank of the process
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Get the name
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &namelen);

    std::cout << "MPI Task " << myrank << " of " << ntasks << " is on "
              << processor_name << std::endl ; fflush(stdout);
#endif

	log_init();

	DEFINE_RGM_LOGGER;
    RGM_LOG(normal, "Welcome to AOGTracker (v1.0)");

	CommandLineInterpreter cli(argc, argv);

    cli.AddCommand(new AOGTrackerCommand());

	int outcode = cli.Process();

#ifdef RGM_USE_MPI
//    if ( myrank != 0 ) {
//        int finished = 1;
//        MPI_Send(&finished, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//    } else {
//        for ( int i = 1; i < ntasks; ++i ) {
//            int finished = 0;
//            MPI_Recv(&finished, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//    }

    MPI_Finalize();

    std::cout << "[" << myrank << "] finished all jobs and quit." << std::endl;
    fflush(stdout);
#endif

	return outcode;
}

