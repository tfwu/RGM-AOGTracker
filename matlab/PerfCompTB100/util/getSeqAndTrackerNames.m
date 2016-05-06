function [seqNames, trackerNames] = getSeqAndTrackerNames(resultDir)

if ~exist(resultDir, 'dir')
    [p,filename,~] = fileparts(resultDir);
    zipfile = fullfile(p, [filename '.zip']);
    if ~exist(zipfile, 'file')
        fprintf(sprintf('Downloading %s.zip ...\n', filename));
        urlwrite(sprintf('http://cvlab.hanyang.ac.kr/tracker_benchmark/v1.1/%s.zip', filename), zipfile);
    end
    fprintf(sprintf('Unzipping %s.zip ...\n', filename));
    unzip(zipfile, p);
    fprintf(sprintf('Removing %s.zip ...\n', filename));
    delete(zipfile);
end

allResultFiles = dir(fullfile(resultDir, '*.mat'));
numResultFiles = length(allResultFiles);
seqNames = cell(0, 0);
trackerNames = cell(0, 0);
trackerResultNum = [];
for i = 1 : numResultFiles
    filename = allResultFiles(i).name(1:end-4); % format: seqName_trackerName
    c = strsplit(filename, '_');
    if length(c) ~= 2
        continue;
    end
    if ~any(strcmp(c{1}, seqNames))
        seqNames{end+1} = c{1};
    end
    idx = find(strcmp(c{2}, trackerNames));
    if isempty(idx)
        trackerNames{end+1} = c{2};
        trackerResultNum(end+1) = 1;
    else
        trackerResultNum(idx(1)) = trackerResultNum(idx(1)) + 1;
    end
end
trackerFlag = trackerResultNum == length(seqNames);
trackerNames = trackerNames(trackerFlag);