function [pred, score] = tknn_apply(model, Xtr, ytr, Xte)
%TKNN_APPLY Fit the selected tkNN outcome(s) on Xtr and predict Xte.
cls = unique(ytr); pos = max(ytr);
m = numel(model.members);
P = zeros(size(Xte, 1), m); S = zeros(size(Xte, 1), m);
for j = 1:m
    b = model.bag(model.members(j), :);
    mdl = fitcknn(Xtr, ytr, 'NumNeighbors', b(2), 'Distance', model.distances{b(1)}, ...
        'DistanceWeight', model.weights{b(3)}, 'ClassNames', cls);
    [P(:, j), s] = predict(mdl, Xte);
    S(:, j) = s(:, end);
end
if model.voted
    pred = mode(P, 2); score = mean(P == pos, 2);
else
    pred = P(:, 1); score = S(:, 1);
end
end
