cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

project( RGM ) # Reconfigurable Grammar Model

###    Build Options     ##########################################################################

option(RGM_USE_DOUBLE "Use doubles instead of floats" OFF)
option(RGM_USE_MPI "Use MPI" OFF) # to run tracker on cluster
option(RGM_RUN_VOT "Run VOT challenge" ON)
option(RGM_USE_LOG "Use log" ON)

# set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH  "Directory where all executables will be stored"
)

# set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH  "Directory where all the libraries will be stored"
)

###    Configuration    ###########################################################################

#    Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -Wall)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC) # set global flags
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}) # set debug flags
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}) # set release flags

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -g -O0)
else()
    set(CMAKE_CXX_COMPILER_FLAGS ${CMAKE_CXX_COMPILER_FLAGS} -O3)
endif()

#message("${CMAKE_CXX_COMPILER_FLAGS}")

#    OpenMP, Not required, but strongly recommended on multi-core systems
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#    Global Definitions
if (RGM_USE_DOUBLE)
    add_definitions(-DRGM_USE_DOUBLE)
endif()

if (RGM_USE_MPI)
    add_definitions(-DRGM_USE_MPI)
endif()

if (RGM_RUN_VOT)
    add_definitions(-DRGM_RUN_VOT)    
endif()

if (NOT RGM_USE_LOG)
    add_definitions(-DRGM_NO_LOG)
endif()

###    Dependencies    ##########################################################################

#    CMake module path for custom module finding
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

#    Boost
# set(BOOST_ROOT /usr/local/include/boost) # modified to your own setting if the default failed
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF) # these settings are needed for boost log
find_package(Boost 1.55 COMPONENTS system filesystem log serialization thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#    OpenCV
find_package(OpenCV REQUIRED) #core highgui imgproc imgcodecs videoio
include_directories(${OpenCV_INCLUDE_DIRS})

#    Eigen matrix
#file(GLOB EIGEN_ARCHIVE "eigen3" PATH /usr/include)
set(EIGEN_ARCHIVE /usr/include/eigen3) # change to your own setting
find_path(EIGEN_INCLUDE_DIR Eigen ${EIGEN_ARCHIVE} .)
if(NOT EIGEN_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find Eigen matrix library.")
endif()
include_directories(${EIGEN_INCLUDE_DIR})
add_definitions(-DEIGEN_USE_NEW_STDVECTOR
                -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)

#   FFTW
set(FFTW3_LIBRARIES )
find_path(FFTW3_INCLUDE_DIR fftw3.h)
if(RGM_USE_DOUBLE)
  find_library(FFTW3_LIBRARIES fftw3)
  #find_library(FFTW3_LIBRARIES_OMP fftw3_omp)
else()
  find_library(FFTW3_LIBRARIES fftw3f)
  #find_library(FFTW3_LIBRARIES_OMP fftw3f_omp)
endif()
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARIES)
  message(FATAL_ERROR "Could not find fftw3.")
endif()
#message("${FFTW3_INCLUDE_DIR}")
include_directories(${FFTW3_INCLUDE_DIR})

#  MPICH
#  not work with openmpi since it does not fully support thread-safty
set(MPI_LIBRARIES )
if (RGM_USE_MPI)
    set(MPI_INCLUDE_PATH /usr/include/mpich)
    find_library(MPI_LIBRARIES mpich)
    if(NOT MPI_LIBRARIES)
      MESSAGE(FATAL_ERROR "Could not find mpich.")
    endif()
    include_directories(${MPI_INCLUDE_PATH})
endif()

# TRAX for VOT
set(TRAX_LIB )
if (RGM_RUN_VOT)
    find_library(TRAX_LIB NAMES   trax)
    if(NOT TRAX_LIB)
        message(FATAL_ERROR "Could not find trax. Please download (https://github.com/votchallenge/trax) and install it.")
    endif()
endif()

#   put all external libs together
set(ALL_EXTERNAL_LIBS
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}           
        ${FFTW3_LIBRARIES}
        ${FFTW3_LIBRARIES_OMP}
        ${MPI_LIBRARIES}        
        ${TRAX_LIB}
    )
#message("${ALL_EXTERNAL_LIBS}")


###    RGM Lib and Executable    ##########################################################################

#    Include Directories
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/external)

# get all source file
file(GLOB_RECURSE   CXX_SRCS    ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE   HXX_SRCS    ${PROJECT_SOURCE_DIR}/src/*.hpp *.h)

file(GLOB_RECURSE   UTIL_CXX_SRCS    ${PROJECT_SOURCE_DIR}/src/util/*.cpp)
file(GLOB_RECURSE   UTIL_HXX_SRCS    ${PROJECT_SOURCE_DIR}/src/util/*.hpp)

if (NOT RGM_USE_FFMPEG)
    list(REMOVE_ITEM CXX_SRCS ${PROJECT_SOURCE_DIR}/src/frame_capture.cpp)
    list(REMOVE_ITEM HXX_SRCS ${PROJECT_SOURCE_DIR}/src/frame_capture.h)
    list(REMOVE_ITEM CXX_SRCS ${PROJECT_SOURCE_DIR}/src/frame_reader.cpp)
    list(REMOVE_ITEM HXX_SRCS ${PROJECT_SOURCE_DIR}/src/frame_reader.h)
endif()

if ( NOT RGM_RUN_VOT )
    list(REMOVE_ITEM HXX_SRCS ${PROJECT_SOURCE_DIR}/src/vot.h)
endif()

set(CXX_SRCS ${CXX_SRCS} ${UTIL_CXX_SRCS})
set(HXX_SRCS ${HXX_SRCS} ${UTIL_HXX_SRCS})

# RGM  library
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(RGMd ${CXX_SRCS} ${HXX_SRCS})
    set(RGM_LIBRARIES RGMd)
else()
    add_library(RGM ${CXX_SRCS} ${HXX_SRCS})
    set(RGM_LIBRARIES RGM)
endif()
#target_link_libraries(${RGM_LIBRARIES} ${ALL_EXTERNAL_LIBS})

# RGM excutable
file(GLOB_RECURSE   MAIN_SRCS   ${CMAKE_SOURCE_DIR}/entry/*.cpp)
if ( NOT RGM_RUN_VOT )
    list(REMOVE_ITEM MAIN_SRCS ${PROJECT_SOURCE_DIR}/entry/AOGTracker4VOT.cpp)
endif()

foreach (m ${MAIN_SRCS})
    get_filename_component(f ${m} NAME_WE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_executable(${f}d ${m})
            target_link_libraries(${f}d ${RGM_LIBRARIES} ${ALL_EXTERNAL_LIBS})
    else()
            add_executable(${f} ${m})
            target_link_libraries(${f}  ${RGM_LIBRARIES} ${ALL_EXTERNAL_LIBS})
    endif()
endforeach()

# Copy configuration file to bin folder.
MESSAGE (STATUS "COPY CONFIG FILE: ${CMAKE_SOURCE_DIR}/config/ to ${EXECUTABLE_OUTPUT_PATH}")
add_custom_target(config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config/ ${EXECUTABLE_OUTPUT_PATH})





