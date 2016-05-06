%% modified from tarcker_benchmark_v1.0 (http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html)
function perfMatDir = GenPerfMat_OtherTrackers_TB100(datasetDir, resultRootDir)
% datasetDir: TB-100 dataset directory
% resultRootDir: directory where tracking results evaluated in TB-100 are
%                stored. It will download the raw results if not existed.

if nargin < 2
    resultRootDir = fileparts(datasetDir);
end

addpath('./util');

if ~exist(resultRootDir, 'dir')
    mkdir(resultRootDir);
end

% evaluated performance
perfMatDir = fullfile(resultRootDir, 'TB-OtherTrackers-Results');
if ~exist(perfMatDir, 'dir')
    mkdir(perfMatDir);
end

% collect groundtruth from benchmark
gtFileDir = fullfile(resultRootDir, 'TB100-GroundtruthFiles');
if ~exist(gtFileDir, 'dir')
    collectGroundtruthFile(datasetDir, 'groundtruth_rect', gtFileDir);
end

evalTypes = {'OPE', 'TRE', 'SRE'};
for t = 1 : length(evalTypes)
    evalType = evalTypes{t};
    if strcmp(evalType, 'OPE') || strcmp(evalType,'TRE')
        curResultFolder = fullfile(resultRootDir, 'pami15_TRE');
        filename1 = fullfile(resultRootDir, 'pami15_TRE_SeqAndTrackerNames.mat');
        filename2 = fullfile(resultRootDir, 'pami15_TRE_GtAndResults.mat');
    else
        curResultFolder = fullfile(resultRootDir, 'pami15_SRE');
        filename1 = fullfile(resultRootDir, 'pami15_SRE_SeqAndTrackerNames.mat');
        filename2 = fullfile(resultRootDir, 'pami15_SRE_GtAndResults.mat');
    end
    
    try
        load(filename);
    catch
        [nameSeqAll, nameTrkAll] = getSeqAndTrackerNames(curResultFolder);
        save(filename1, 'nameSeqAll', 'nameTrkAll');
    end
    
    try
        load(filename2);
    catch
        [allGts, allResults, allResTypes] = loadGtAndResults(nameSeqAll, nameTrkAll, gtFileDir, curResultFolder);
        save(filename2, 'allGts', 'allResults', 'allResTypes');
    end
    
    nameSeqAll = lower(nameSeqAll);
    for subsetIdx = 0 : 2  % 0: TB100, 1: TB50, 2: TB-CVPR2013
        if subsetIdx == 0
            nameDataset = 'TB-100';
            nameSeqUsed = nameSeqAll;
        elseif subsetIdx == 1
            seqNames = {'Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', ...
                'BlurOwl', 'Bolt', 'Box','Car1','Car4','CarDark','CarScale','ClifBar','Couple', ...
                'Crowds','David','Deer','Diving','DragonBaby','Dudek', 'Football',...
                'Freeman4','Girl','Human3','Human4.2','Human6','Human9','Ironman','Jump','Jumping',...
                'Liquor','Matrix','MotorRolling','Panda','RedTeam','Shaking','Singer2','Skating1',...
                'Skating2.1','Skating2.2','Skiing','Soccer','Surfer','Sylvester','Tiger2','Trellis','Walking',...
                'Walking2','Woman' };
            nameSeqUsed = lower(seqNames);
            nameDataset = 'TB-50';
        elseif subsetIdx == 2
            seqNames = {'cardark', 'car4', 'david', 'david2', 'sylvester',...
                'trellis', 'fish', 'mhyang', 'soccer', 'matrix',...
                'ironman', 'deer', 'skating1', 'shaking', 'singer1', ...
                'singer2', 'coke', 'bolt', 'boy', 'dudek', 'crossing',...
                'couple', 'football1', 'jogging.1', 'jogging.2', 'doll',...
                'girl', 'walking2', 'walking', 'fleetface', 'freeman1',...
                'freeman3', 'freeman4', 'david3', 'jumping', 'carscale',...
                'skiing', 'dog1', 'suv', 'motorrolling', 'mountainbike',...
                'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2','basketball',...
                'football', 'subway', 'tiger1', 'tiger2'};
            nameSeqUsed = lower(seqNames);
            nameDataset = 'TB-CVPR13';
        end
        
        computePerfMat(evalType, nameDataset, nameSeqUsed, ...
            nameSeqAll, nameTrkAll, allGts, allResults, perfMatDir);
    end
end



%%
function computePerfMat(evalType, nameDataset, nameSeqUsed, ...
    nameSeqAll, nameTrkAll, allGts, allResults, perfMatDir)

numSeqUsed = length(nameSeqUsed);
numTrk = length(nameTrkAll);

thresholdSetOverlap = 0:0.05:1;
thresholdSetError = 0:50;

switch evalType
    case 'SRE'
        numRound = length(allResults{1, 1});
    case 'TRE'
        numRound = length(allResults{1, 1});
    case 'OPE'
        numRound = 1;
end

filenameAUC=fullfile(perfMatDir, [nameDataset '_aveSuccessRatePlot_' num2str(numTrk) 'alg_overlap_' evalType '.mat']);
filenameErr=fullfile(perfMatDir, [nameDataset '_aveSuccessRatePlot_' num2str(numTrk) 'alg_error_' evalType '.mat']);

if ~exist(filenameAUC, 'file') || ~exist(filenameErr, 'file')
    
    successNumOverlap = zeros(numRound,length(thresholdSetOverlap));
    successNumErr = zeros(numRound,length(thresholdSetError));
    
    aveSuccessRatePlot = zeros(numTrk, numSeqUsed, length(thresholdSetOverlap));
    aveSuccessRatePlotErr = zeros(numTrk, numSeqUsed, length(thresholdSetError));
    
    for i = 1 : numTrk
        for j = 1 : numSeqUsed
            seqIdx = find(strcmp(nameSeqUsed{j}, nameSeqAll));
            res = allResults{seqIdx, i};
            gt = allGts{seqIdx, 1};
            lenALL = 0;
            for k = 1 : numRound
                curRes = res{k};
                [~, ~, errCoverage, errCenter] = calcSeqErrRobust(curRes, gt);
                for tIdx=1:length(thresholdSetOverlap)
                    successNumOverlap(k,tIdx) = sum(errCoverage >thresholdSetOverlap(tIdx));
                end
                
                for tIdx=1:length(thresholdSetError)
                    successNumErr(k,tIdx) = sum(errCenter <= thresholdSetError(tIdx));
                end
                
                lenALL = lenALL + curRes.seq_len;
            end
            if strcmp(evalType, 'OPE')
                aveSuccessRatePlot(i, j,:) = successNumOverlap/(lenALL+eps);
                aveSuccessRatePlotErr(i, j,:) = successNumErr/(lenALL+eps);
            else
                aveSuccessRatePlot(i, j,:) = sum(successNumOverlap)/(lenALL+eps);
                aveSuccessRatePlotErr(i, j,:) = sum(successNumErr)/(lenALL+eps);
            end
        end
    end
    
    save(filenameAUC,'aveSuccessRatePlot','nameTrkAll');
    
    aveSuccessRatePlot = aveSuccessRatePlotErr;
    save(filename,'aveSuccessRatePlot','nameTrkAll');
end

load(filenameAUC);
perf = zeros(1, numTrk);
for idxTrk=1:numTrk
    %each row is the sr plot of one sequence
    idxSeqSet = 1:size(aveSuccessRatePlot, 2);
    tmp=aveSuccessRatePlot(idxTrk, idxSeqSet,:);
    aa=reshape(tmp,[length(idxSeqSet),size(aveSuccessRatePlot,3)]);
    aa=aa(sum(aa,2)>eps,:);
    bb=mean(aa);
    perf(idxTrk) = mean(bb);
end
[sP, sIdx] = sort(perf, 'descend');
fprintf('****** Overlap: numSeq=%d, %s, %s\n', numSeqUsed, evalType, nameDataset);
for idx = 1 : numTrk
    fprintf('  %s:%f,', nameTrkAll{sIdx(idx)}, sP(idx));
end
fprintf('\n');


load(filenameErr);
perf(:) = 0;
for idxTrk=1:numTrk
    %each row is the sr plot of one sequence
    idxSeqSet = 1:size(aveSuccessRatePlot, 2);
    tmp=aveSuccessRatePlot(idxTrk, idxSeqSet,:);
    aa=reshape(tmp,[length(idxSeqSet),size(aveSuccessRatePlot,3)]);
    aa=aa(sum(aa,2)>eps,:);
    bb=mean(aa);
    perf(idxTrk) = mean(bb);
end
[sP, sIdx] = sort(perf, 'descend');
fprintf('****** Err: numSeq=%d, %s, %s\n', numSeqUsed, evalType, nameDataset);
for idx = 1 : numTrk
    fprintf('  %s:%f,', nameTrkAll{sIdx(idx)}, sP(idx));
end
fprintf('\n');


%%
function [allGts, allResults, allResTypes] = loadGtAndResults(nameSeqAll, nameTrkAll, gtFileDir, curResultFolder)
numSeq = length(nameSeqAll);
numTrk = length(nameTrkAll);
% load gt
allGts = cell(numSeq, 1);
for i = 1 : numSeq
    filename = fullfile(gtFileDir, [nameSeqAll{i} '_groundtruth_rect.txt']);
    try
        allGts{i, 1} = dlmread(filename);
    catch
        error(sprintf('Not found %s\n', filename));
    end
end

% load tracking results
allResults = cell(numSeq, numTrk);
allResTypes = cell(numSeq * numTrk, 1);
k = 1;
for i = 1 : numSeq
    numGt = size(allGts{i, 1}, 1);
    for j = 1 : numTrk
        filename = fullfile(curResultFolder, [nameSeqAll{i} '_' nameTrkAll{j} '.mat']);
        if exist(filename, 'file')
            load(filename);
            
            % "patch" the provided raw results which have some "issues" in
            % frame ranges
            results = patchResults(results, numGt);
            
            allResults{i, j} = results;
            allResTypes{k} = results{1,1}.type;
            k = k + 1;
        else
            error('Not found %s\n', filename);
        end
    end
end
allResTypes = unique(allResTypes);





