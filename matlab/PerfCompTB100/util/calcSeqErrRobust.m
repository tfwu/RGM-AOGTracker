function [aveErrCoverage, aveErrCenter,errCoverage, errCenter, rectMat] = calcSeqErrRobust(results, rect_anno)

tmp = strsplit(results.seq_range_str, ':'); % the provided results have some strange range values
startIdx = str2num(tmp{1});
endIdx = str2num(tmp{2});

seq_length = results.seq_len;
if seq_length ~= (endIdx - startIdx + 1)  || startIdx <= 0 || endIdx <= 0 || startIdx > endIdx || endIdx > length(rect_anno) 
   error('error: sequence range:%s_%s', results.seq_name, results.tracker);
end

rect_anno = rect_anno(startIdx:endIdx, :);

if strcmp(results.type,'rect')
    for i = 2:seq_length
        r = results.res(i,:);
        r_anno = rect_anno(i,:);
        try
            if (isnan(r) | r(3)<=0 | r(4)<=0) & (~isnan(r_anno))
                results.res(i,:)=results.res(i-1,:);
            end
        catch
            error('groundtruth error');
        end
    end
end

centerGT = [rect_anno(:,1)+(rect_anno(:,3)-1)/2 rect_anno(:,2)+(rect_anno(:,4)-1)/2];

rectMat = zeros(seq_length, 4);
if seq_length > size(results.res,1) || seq_length > size(centerGT, 1)
    debug = 1;
end
switch results.type
    case 'rect'
        rectMat = results.res;
    case 'affine_ivt'
        for i = 1:seq_length
            [rect c] = calcRectCenter(results.tmplsize, results.res(i,:));
            rectMat(i,:) = rect;
            %                     center(i,:) = c;
        end
    case 'affine_L1'
        for i = 1:seq_length
            [rect c] = calcCenter_L1(results.res(i,:), results.tmplsize);
            rectMat(i,:) = rect;
        end
    case 'LK_Aff'
        for i = 1:seq_length
            [corner, c] = getLKcorner(results.res(2*i-1:2*i,:), results.tmplsize);
            rectMat(i,:) = corner2rect(corner);
        end
    case '4corner'
        for i = 1:seq_length
            rectMat(i,:) = corner2rect(results.res(2*i-1:2*i,:));
        end
    case 'affine'
        for i = 1:seq_length
            rectMat(i,:) = corner2rect(results.res(2*i-1:2*i,:));
        end
    case 'similarity'
        for i = 1:seq_length
            warp_p = parameters_to_projective_matrix(results.type,results.res(i,:));
            [corner, c] = getLKcorner(warp_p, results.tmplsize);
            rectMat(i,:) = corner2rect(corner);
        end
end

if isfield(results,'shift_type')
    center = [rectMat(:,1)+(rectMat(:,3)-1)/2 rectMat(:,2)+(rectMat(:,4)-1)/2];
    ratio=1;
    
    shiftType=results.shift_type;
    switch shiftType
        case 'scale_8'
            ratio=0.8;
        case 'scale_9'
            ratio=0.9;
        case 'scale_11'
            ratio=1.1;
        case 'scale_12'
            ratio=1.2;
    end
    
    w = rectMat(:,3)/ratio;
    h = rectMat(:,4)/ratio;
    rectMat = round([center(:,1)-w/2,center(:,2)-h/2,w,h]);
end



rectMat(1,:) = rect_anno(1,:);

center = [rectMat(:,1)+(rectMat(:,3)-1)/2 rectMat(:,2)+(rectMat(:,4)-1)/2];

errCenter = sqrt(sum(((center(1:seq_length,:) - centerGT(1:seq_length,:)).^2),2));

index = rect_anno>0;
idx=(sum(index,2)==4);
tmp = calcRectInt(rectMat(idx,:),rect_anno(idx,:));

errCoverage=-ones(length(idx),1);
errCoverage(idx) = tmp;
errCenter(~idx)=-1;

aveErrCoverage = sum(errCoverage(idx))/length(idx);

aveErrCenter = sum(errCenter(idx))/length(idx);