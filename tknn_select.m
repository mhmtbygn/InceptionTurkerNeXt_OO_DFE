function model = tknn_select(X, y, fold, cfg)
%TKNN_SELECT tkNN with iterative majority voting (Section 2.3, Steps 11-13).
%   120 kNN outcomes (4 distances x 10 k values x 3 weightings) are
%   computed with the given internal folds. The outcomes are sorted by
%   accuracy, and IMV builds 118 voted outcomes from the top 3..120. The
%   most accurate of the 238 outcomes is selected (greedy).
[D, K, Wt] = ndgrid(1:numel(cfg.tknn.distances), cfg.tknn.k, 1:numel(cfg.tknn.weights));
bag = [D(:), K(:), Wt(:)];                      % 120 x 3
nb = size(bag, 1);
P = zeros(numel(y), nb); S = zeros(numel(y), nb); acc = zeros(nb, 1);
for l = 1:nb
    [P(:, l), S(:, l)] = knn_cv_predict(X, y, fold, bag(l, 2), ...
        cfg.tknn.distances{bag(l, 1)}, cfg.tknn.weights{bag(l, 3)});
    acc(l) = mean(P(:, l) == y);
end
[~, idx] = sort(acc, 'descend');
pos = max(y);
nv = nb - 2;                                     % 118 voted outcomes
V = zeros(numel(y), nv); VS = zeros(numel(y), nv); vacc = zeros(nv, 1);
for p = 1:nv
    members = idx(1:p + 2);
    V(:, p)  = mode(P(:, members), 2);
    VS(:, p) = mean(P(:, members) == pos, 2);    % vote proportion of the positive class
    vacc(p)  = mean(V(:, p) == y);
end
allAcc = [acc; vacc];
[~, xd] = max(allAcc);
if xd <= nb
    model.members = xd; model.voted = false;
    model.cvPred = P(:, xd); model.cvScore = S(:, xd);
else
    p = xd - nb;
    model.members = idx(1:p + 2); model.voted = true;
    model.cvPred = V(:, p); model.cvScore = VS(:, p);
end
model.bag = bag; model.acc = acc; model.votedAcc = vacc; model.selected = xd;
model.distances = cfg.tknn.distances; model.weights = cfg.tknn.weights;
end
