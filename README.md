# AOGTracker 

## Reproducing all experimental results in the paper
Tianfu Wu , Yang Lu and Song-Chun Zhu, Online Object Tracking, Learning and Parsing with And-Or Graphs, arXiv 1509.08067, TPAMI (under revision).  http://arxiv.org/abs/1509.08067

See a demo here: https://www.youtube.com/watch?v=1Ian4qzkNLA

The code is written by Matt Tianfu Wu (tfwu@stat.ucla.edu). Please feel free to report issues to him. 

Copyright (c) 2016, Matt Tianfu Wu

All rights reserved.

If you find the code is useful in your projects, please consider to cite the paper,

	@article{AOGTracker,
	  author    = {Tianfu Wu and
	               Yang Lu and
	               Song{-}Chun Zhu},
	  title     = {Online Object Tracking, Learning and Parsing with And-Or Graphs},
	  journal   = {CoRR},
	  volume    = {abs/1509.08067},
	  year      = {2015},
	  url       = {http://arxiv.org/abs/1509.08067},
	  timestamp = {Thu, 01 Oct 2015 14:28:48 +0200},
	  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/WuLZ15b},
	  bibsource = {dblp computer science bibliography, http://dblp.org}
	}


## 0. System requirements:

### OS
We tested our tracker on Ubuntu 14.04 LTS. Other OS will be supported later on.

### Third-party libraries
sudo apt-get install build-essential cmake libboost1.55-all-dev libopencv-dev libeigen3-dev  libfftw3-dev graphviz mpich2

Note: please use libboost1.55-all-dev for the best practice. 

###  The TRAX library for VOT
It is needed for integrating AOGTracker4VOT into vot-toolkit. Please follow the instrunctions at https://github.com/votchallenge/trax.git.

## 1. Compiling the code
Assume you put the code at PATH_TO_AOGTracker.
It is recommended to build the software in a separate directory. For example

	cd PATH_TO_AOGTracker
	mkdir build
	cd build

Then use CMake to generate the necessary Makefiles with different options (e.g., release version with MPI and VOT support), which you can change accordingly 
	
	cmake -DCMAKE_BUILD_TYPE=Release -DRGM_USE_MPI=OFF -DRGM_RUN_VOT=ON ..

Then build the code with

	make or make -j 8 (using multithread)

Or, use CMake-gui to do this and use your own favoriate c++ IDE (e.g., Qt creator) to build the code.

After compiling, the release/debug version executables (entry or entryd, AOGTracker4VOT or AOGTracker4VOTd) will be put in PATH_TO_AOGTracker/build/bin

## 2. Preparing the Datasets
TB100/50 is available at http://cvlab.hanyang.ac.kr/tracker_benchmark/. Please download all the data to PATH_TO_TB100 (e.g. /home/your_user_name/Data/TB100/)
Note that: TB100-occ is provided which specifies omitting frame index in TRE (provided by TB-100 authors)

VOT datasets are vailable at http://www.votchallenge.net/. vot-toolkit will download the data automatically.

## 3. Run AOGTracker in the termial
Change settings in the configuratin xml file, PATH_TO_AOGTracker/config/tracker_config.xml (e.g., specify your data directory, and TB100-occ directory for omitFrameIdxSpecDir, etc)

### Using single workstation (when -DDRGM_USE_MPI=OFF was used in compiling the code)

	cd PATH_TO_AOGTracker/build/bin
	./entry Tracking PATH_TO_AOGTracker/config/tracker_config.xml

### Using a cluster through MPI  (when -DDRGM_USE_MPI=ON was used in compiling the code)
If you have mulitple workstations available, please follow https://help.ubuntu.com/community/MpichCluster to set up the cluster.
	
	cd PATH_TO_AOGTracker/build/bin
	/usr/bin/mpiexec.mpich -f machine ./entry Tracking PATH_TO_AOGTracker/config/tracker_config.xml

	Note: "machine" is a txt file specifying the cluster machines, and the executable and data directory should be shared among all cluster machines.

## 4. Generate performance comparison plots in TB-100
We provide the matlab scripts. Run PATH_TO_AOGTracker/matlab/PerfCompTB100/GenPerfPlots_TB100.m

## 5. Run AOGTracker4VOT
a) Follow the vot-toolkit tutorial to set up the testing environment using matlab.

b) Modify the configuration.m file by adding to the end: set_global_variable('trax_timeout', 20*60); 

c) Modify the tracker_AOGTracker.m, e.g., 

	tracker_label = ['AOGTracker'];

	> for VOT2013, VOT2014 and VOT2015
		tracker_command = 'PATH_TO_AOGTracker/build/bin/AOGTracker4VOT PATH_TO_AOGTracker/config/vot_config.xml';

	> for VOT-TIR2015
		tracker_command = 'PATH_TO_AOGTracker/build/bin/AOGTracker4VOT PATH_TO_AOGTracker/config/vottir_config.xml';

## 6. Acknowledgement
In general, the code is developed with the help from voc-release 5 by Dr. Ross Girshick and Prof. Felzenszwalb. 
The codes for computing HOG features, FFT convolution and LBFGS are adapted from FFLD by Dr. Charles Dubout <http://charles.dubout.ch/en/coding.html>. We are grateful to them for making their codes publicly available.
