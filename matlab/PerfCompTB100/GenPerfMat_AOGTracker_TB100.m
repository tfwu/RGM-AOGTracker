%% modified from tarcker_benchmark_v1.0 (http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html)
function perfMatRootPath = GenPerfMat_AOGTracker_TB100(resultIdentifier, AOGTrackerVersion, datasetDir,  resultRootDir)
% resultIdentifier： the "note" in tracker_config.xml, usually time stamp
% AOGTrackerVersion: four variants of the AOGTracker (AOG-st, AOG-s,
%                       ObjectOnly-st, ObjectOnly-s)
% datasetDir: directory where the TB100 dataset is stored
% resultRootDir: directory to save results

addpath('./util');

perfMatRootPath = fullfile(resultRootDir, 'TB-AOGTracker-Results');
perfMatPath = fullfile(perfMatRootPath, AOGTrackerVersion);
if ~exist(perfMatPath, 'dir')
   mkdir(perfMatPath); 
end

evalTypes = {'OPE', 'TRE', 'SRE'};
for t = 1 : 3
    evalType = evalTypes{t};
    % "resultIdentifier" corresponds to the "note" in tracker_config.xml
    resultFile = fullfile(datasetDir, ['AOGTracker_Result_' evalType '_' resultIdentifier '.txt']); 
    for subsetIdx = 0 : 2  % 0: TB100, 1: TB50, 2: TB-CVPR2013
        if subsetIdx == 0
            note = '_TB-100';
        elseif subsetIdx == 1
            seqNames = {'Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', ...
                'BlurOwl', 'Bolt', 'Box','Car1','Car4','CarDark','CarScale','ClifBar','Couple', ...
                'Crowds','David','Deer','Diving','DragonBaby','Dudek', 'Football',...
                'Freeman4','Girl','Human3','Human4.2','Human6','Human9','Ironman','Jump','Jumping',...
                'Liquor','Matrix','MotorRolling','Panda','RedTeam','Shaking','Singer2','Skating1',...
                'Skating2.1','Skating2.2','Skiing','Soccer','Surfer','Sylvester','Tiger2','Trellis','Walking',...
                'Walking2','Woman' };
            seqNames = lower(seqNames);
            note = '_TB-50';
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
            seqNames = lower(seqNames);
            note = '_TB-CVPR13';
        end
        
        % each row in the resultFile is
        % seqName \t seqResultFile \t gtFile \t startFrameIdx0 \t startFrameIdx \t endFrameIdx \t timeConsumed \t speed \t shiftType(TRE)
        if ~exist(resultFile, 'file')
            warning('Not found %s\n', resultFile);
            return;
        end
        fid = fopen(resultFile);
        allRes = textscan(fid, '%s %s %s %d %d %d %f %f %s', 'Delimiter', '\t');
        fclose(fid);
        numResult = length(allRes{1});          
        
        [~, sIdx] = sort(allRes{3});        
        
        thresholdSetOverlap = 0:0.05:1;
        thresholdSetError = 0:50;
        
        idxSeq = 1;
        idxRes = 1;        
        aveSuccessRatePlot = [];
        aveSuccessRatePlotErr = [];
        usedSeqNames = [];
        while idxRes <= numResult
            seqName = allRes{1}{sIdx(idxRes)};
            gtFile = allRes{3}{sIdx(idxRes)};
            idxResNext = idxRes + 1;
            while idxResNext <= numResult && strcmp(seqName, allRes{1}{sIdx(idxResNext)}) && strcmp(gtFile, allRes{3}{sIdx(idxResNext)})
                idxResNext = idxResNext + 1;
            end
            idxNum = idxResNext - idxRes;
            s = struct;
            for i = 1 : idxNum
                s(i).name = allRes{1}{sIdx(idxRes)};
                s(i).resultFile = allRes{2}{sIdx(idxRes)};
                s(i).gtFile = allRes{3}{sIdx(idxRes)};
                s(i).startFrame0 = allRes{4}(sIdx(idxRes));
                s(i).startFrame = allRes{5}(sIdx(idxRes));
                s(i).endFrame = allRes{6}(sIdx(idxRes)) - 1;
                s(i).timeConsumed = allRes{7}(sIdx(idxRes));
                s(i).speed = allRes{8}(sIdx(idxRes));
                s(i).shift = allRes{9}{sIdx(idxRes)};
                idxRes = idxRes + 1;
            end
            
            [~, gtBasename, ~] = fileparts(gtFile);
            idxTmp = strfind(gtBasename, '.');
            if ~isempty(idxTmp)
                seqName1 = [seqName '.' gtBasename(idxTmp(1)+1:end)];
            else
                seqName1 = seqName;
            end
            
            if subsetIdx > 0
                if isempty(strmatch(lower(seqName1), seqNames, 'exact'))
                    continue;
                end
            end
            
%             disp([num2str(idxSeq) ' ' seqName1 ' ' num2str(idxNum)]);            
            usedSeqNames{idxSeq} = seqName1;
            
            rect_anno = dlmread(s(1).gtFile);
            
            lenALL = 0;
            
            successNumOverlap = zeros(idxNum,length(thresholdSetOverlap));
            successNumErr = zeros(idxNum,length(thresholdSetError));
            
            for idx = 1:idxNum
                res = dlmread(s(idx).resultFile);
                res(:, 1) = res(:, 1) + 1;
                res(:, 2) = res(:, 2) + 1;
                istart = s(idx).startFrame - s(idx).startFrame0 + 1;
                iend = min(size(rect_anno, 1), s(idx).endFrame - s(idx).startFrame0 + 1);
                anno = rect_anno(istart : iend, :);
                len = size(anno,1);
                
                res = res(1:iend-istart+1, :);
                [aveCoverage, aveErrCenter, errCoverage, errCenter] = calcSeqErr(res, anno);
                
                for tIdx=1:length(thresholdSetOverlap)
                    successNumOverlap(idx,tIdx) = sum(errCoverage >thresholdSetOverlap(tIdx));
                end
                
                for tIdx=1:length(thresholdSetError)
                    successNumErr(idx,tIdx) = sum(errCenter <= thresholdSetError(tIdx));
                end
                
                lenALL = lenALL + len;
            end
            
            if strcmp(evalType, 'OPE')
                aveSuccessRatePlot(idxSeq,:) = successNumOverlap/(lenALL+eps);
                aveSuccessRatePlotErr(idxSeq,:) = successNumErr/(lenALL+eps);
            else
                aveSuccessRatePlot(idxSeq,:) = sum(successNumOverlap)/(lenALL+eps);
                aveSuccessRatePlotErr(idxSeq,:) = sum(successNumErr)/(lenALL+eps);
            end
            idxSeq = idxSeq + 1;
        end
        
        %
        dataName1=fullfile(perfMatPath, ['aveSuccessRatePlot_AOGTracker_overlap_' evalType  note '.mat']);
        save(dataName1,'aveSuccessRatePlot', 'usedSeqNames');
        
        aa=aveSuccessRatePlot;
        aa=aa(sum(aa,2)>eps,:);
        bb=mean(aa);
        disp(['****** AOGTracker AUC: numSeq ' num2str(idxSeq-1) ' ' evalType note ': ' num2str(mean(bb))]);
               
        
        dataName2=fullfile(perfMatPath, ['aveSuccessRatePlot_AOGTracker_error_' evalType note '.mat']);
        aveSuccessRatePlot = aveSuccessRatePlotErr;
        save(dataName2,'aveSuccessRatePlot', 'usedSeqNames');
                
        aa=aveSuccessRatePlot;
        aa=aa(sum(aa,2)>eps,:);
        bb=mean(aa);
        disp(['       AOGTracker Precision: numSeq ' num2str(idxSeq-1) ' ' evalType note ': ' num2str(mean(bb))]);      
        
    end
end

