function [pred, score] = knn_cv_predict(X, y, fold, k, dist, wt)
%KNN_CV_PREDICT Cross-validated kNN predictions for a given fold vector.
%   score is the posterior of the second (positive) class.
pred = zeros(size(y)); score = zeros(size(y));
cls = unique(y);
for f = unique(fold).'
    te = fold == f; tr = ~te;
    mdl = fitcknn(X(tr, :), y(tr), 'NumNeighbors', k, 'Distance', dist, ...
        'DistanceWeight', wt, 'ClassNames', cls);
    [p, s] = predict(mdl, X(te, :));
    pred(te) = p; score(te) = s(:, end);
end
end
