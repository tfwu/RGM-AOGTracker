%% get colors for vis
function colors = distinctColors(num)
colors_candidate = colormap('hsv');
colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/num)):end, :);
%colors_candidate = round(colors_candidate .* 255);
colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
colors = colors_candidate;
idx = randperm(length(colors));
colors = colors(idx);
