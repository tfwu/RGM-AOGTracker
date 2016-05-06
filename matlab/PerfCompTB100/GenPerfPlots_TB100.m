function GenPerfPlots_TB100(AOGResultIdentifier, AOGTrackerVersion, datasetDir, resultRootDir)
% AOGResultIdentifier： the "note" in tracker_config.xml, usually time stamp
% AOGTrackerVersion: four variants of the AOGTracker (AOG-st, AOG-s,
%                       ObjectOnly-st, ObjectOnly-s)
% datasetDir: directory where the TB100 dataset is stored
% resultRootDir: directory to save results

% compute performance for other trackers in TB-100
resultDir_Others = GenPerfMat_OtherTrackers_TB100(datasetDir, resultRootDir);

% compute performance for AOGTracker in TB-100
resultDir_AOGTracker = GenPerfMat_AOGTracker_TB100(AOGResultIdentifier, AOGTrackerVersion, datasetDir, resultRootDir);

% generate plots
Performance_Comp_TB100(resultDir_AOGTracker, resultDir_Others, resultRootDir);