function results = patchResults(results, numGt)
% "patch" the provided raw results which have some "issues" in
% frame ranges

res = results{1};
tmp = strsplit(res.seq_range_str, ':');
startIdx = str2num(tmp{1});
endIdx = str2num(tmp{2});
seq_length = res.seq_len;
assert(seq_length == (endIdx - startIdx + 1));

% a special setting used in TB-100
if all(strcmpi(res.seq_name, 'David'))
    startIdx = startIdx - 300 + 1;
    endIdx = endIdx - 300 + 1;
    results{1}.seq_range_str=sprintf('%d:%d', startIdx,endIdx);
    for r = 2 : length(results)
        res = results{r};
        tmp = strsplit(res.seq_range_str, ':');
        startIdx = str2num(tmp{1});
        endIdx = str2num(tmp{2});
        startIdx = startIdx - 300 + 1;
        endIdx = endIdx - 300 + 1;
        results{r}.seq_range_str=sprintf('%d:%d', startIdx,endIdx);
    end
end

% the provided results have some "strange" range values
if startIdx <= 0
    guessRealStart = startIdx;
    if seq_length == numGt - 1 % assuming the tracker use 0-based index
        guessRealStart = startIdx - 1;
        results{1}.res = [res.res(1,:); res.res];
        results{1}.seq_len = results{1}.seq_len + 1;
        results{1}.seq_range_str=sprintf('1:%d', numGt);
    elseif seq_length == numGt
        results{1}.seq_range_str=sprintf('1:%d', numGt);
    else
        error('Errors in tracking results:%s', filename);
    end
    for r = 2 : length(results)
        res = results{r};
        tmp = strsplit(res.seq_range_str, ':'); % the provided results have some strange range values
        startIdx = str2num(tmp{1});
        endIdx = str2num(tmp{2});
        seq_length = res.seq_len;
        assert(seq_length == (endIdx - startIdx + 1));
        results{r}.seq_range_str=sprintf('%d:%d', startIdx-guessRealStart+1, endIdx-guessRealStart+1);
    end
else
    if startIdx == 2 && endIdx == numGt + 1
        results{1}.seq_range_str=sprintf('1:%d', numGt);
        for r = 2 : length(results)
            res = results{r};
            tmp = strsplit(res.seq_range_str, ':'); % the provided results have some strange range values
            startIdx = str2num(tmp{1});
            endIdx = str2num(tmp{2});
            results{r}.seq_range_str=sprintf('%d:%d', startIdx-1, endIdx-1);
        end
    end
end