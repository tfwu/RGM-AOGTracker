function collectGroundtruthFile(datasetDir, gtFilename, resultDir)
% datasetDir:  dataset root directory
% gtFilename:  e.g., 'groundtruth_rect' for TB-100
% resultDir: the destination directory 

if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

tmp = dir(datasetDir);
dirFlags = [tmp.isdir];
allFolders = tmp(dirFlags);

for i = 3 : length(allFolders)    
    seqName = allFolders(i).name;  
    seqFolder = seqName;    
    found = copyOne(datasetDir, gtFilename, seqFolder, seqName, resultDir);
    if ~found
        seqFolder = [seqName filesep seqName];
        found = copyOne(datasetDir, gtFilename, seqFolder, seqName, resultDir);
        if ~found
           warning(sprintf('Not found gt for %s\n', allFolders(i).name)); 
        end
    end
end

%%
function found = copyOne(datasetDir, gtFilename, seqFolder, seqName, resultDir)
found = false;
gtFile = fullfile(datasetDir, seqFolder, [gtFilename '.txt']);
if exist(gtFile, 'file')
    copyfile(gtFile, fullfile(resultDir, [seqName '_' gtFilename '.txt']), 'f');
    found = true;
end
for j = 1 : 10
    gtFile = fullfile(datasetDir, seqFolder, [gtFilename '.' num2str(j) '.txt']);
    if exist(gtFile, 'file')
        copyfile(gtFile, fullfile(resultDir, [seqName '.' num2str(j) '_' gtFilename '.txt']), 'f');
        found = true;
    end
end

