function fold = make_group_folds(y, groups, K, seed)
%MAKE_GROUP_FOLDS Patient-grouped, class-stratified fold assignment.
%   fold = make_group_folds(y, groups, K, seed) returns a fold index (1..K)
%   per image. All images of one patient share a fold. Patients are
%   shuffled within each class and dealt to the folds in turn. When groups
%   is empty, every image is its own group (image-level folds).
y = y(:);
if isempty(groups), groups = (1:numel(y)).'; end
groups = groups(:);
rng(seed, 'twister');
[ug, ~, gi] = unique(groups);
gLabel = accumarray(gi, double(y), [], @(v) mode(v));
fold_g = zeros(numel(ug), 1);
classes = unique(gLabel);
offset = 0;
for c = classes.'
    idx = find(gLabel == c);
    idx = idx(randperm(numel(idx)));
    fold_g(idx) = mod((0:numel(idx)-1).' + offset, K) + 1;
    offset = offset + numel(idx);
end
fold = fold_g(gi);
end
