function [sel, norm, info] = cwinca_select(X, y, fold, cfg)
%CWINCA_SELECT Cumulative weighted iterative NCA (Section 2.3, Steps 6-10).
%   [sel, norm, info] = cwinca_select(X, y, fold, cfg)
%   X     training features (rows = images), y labels, fold = internal folds
%   sel   indices of the selected features (columns of X)
%   norm  struct with min/max of X (fitted on X only)
%   Steps: min-max normalization -> NCA weights (fscnca, default lbfgs) ->
%   descending sort -> start/stop from the cumulative weight at tr1/tr2 ->
%   kNN (k=1, L1, equal weights) accuracy of every top-n subset with the
%   given internal folds -> greedy choice of the most accurate subset.
norm.mn = min(X, [], 1);
norm.mx = max(X, [], 1);
Xn = (X - norm.mn) ./ max(norm.mx - norm.mn, eps('single'));
mdl = fscnca(double(Xn), y, 'Solver', 'lbfgs');
w = mdl.FeatureWeights;
[ws, ind] = sort(w, 'descend');
cw = cumsum(ws) / sum(ws);
start = find(cw >= cfg.cwinca.tr1, 1, 'first');
stop  = find(cw >= cfg.cwinca.tr2, 1, 'first');
sizes = start:cfg.cwinca.step:stop;
acc = zeros(numel(sizes), 1);
for j = 1:numel(sizes)
    p = knn_cv_predict(Xn(:, ind(1:sizes(j))), y, fold, cfg.cwinca.knnK, cfg.cwinca.knnDistance, 'equal');
    acc(j) = mean(p == y);
end
[bestAcc, jb] = max(acc);
sel = ind(1:sizes(jb));
info = struct('start', start, 'stop', stop, 'nSelected', numel(sel), ...
    'bestInnerAcc', bestAcc, 'sizes', sizes, 'acc', acc, 'weights', w);
end
