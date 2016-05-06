function [aveErrCoverage, aveErrCenter,errCoverage, errCenter] = calcSeqErr(results, rect_anno)

seq_length = size(results, 1);
try
for i = 2:seq_length
    r = results(i,:);
    r_anno = rect_anno(i,:);
    if (isnan(r) | r(3)<=0 | r(4)<=0) & (~isnan(r_anno))
        results.res(i,:)=results.res(i-1,:);
    end
end
catch
   debug = 1; 
end
centerGT = [rect_anno(:,1)+(rect_anno(:,3)-1)/2 rect_anno(:,2)+(rect_anno(:,4)-1)/2];

rectMat = results;

rectMat(1,:) = rect_anno(1,:);

center = [rectMat(:,1)+(rectMat(:,3)-1)/2 rectMat(:,2)+(rectMat(:,4)-1)/2];

errCenter = sqrt(sum(((center(1:seq_length,:) - centerGT(1:seq_length,:)).^2),2));

index = rect_anno > 0;
idx=(sum(index,2)==4);
tmp = calcRectInt(rectMat(idx,:),rect_anno(idx,:));

errCoverage= -ones(length(idx),1);
errCoverage(idx) = tmp;
errCenter(~idx)=-1;

aveErrCoverage = sum(errCoverage(idx))/length(idx);

aveErrCenter = sum(errCenter(idx))/length(idx);