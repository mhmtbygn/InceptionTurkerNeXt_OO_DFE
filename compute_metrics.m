function M = compute_metrics(ytrue, ypred, score, pos)
%COMPUTE_METRICS Binary diagnostic measures; pos = label of the positive class (OO).
t = ytrue(:) == pos; p = ypred(:) == pos;
TP = sum(t & p); FN = sum(t & ~p); FP = sum(~t & p); TN = sum(~t & ~p);
n = TP + FN + FP + TN;
M.TP = TP; M.FN = FN; M.FP = FP; M.TN = TN; M.n = n;
M.accuracy    = 100 * (TP + TN) / n;
M.sensitivity = 100 * TP / (TP + FN);
M.specificity = 100 * TN / (TN + FP);
M.precision   = 100 * TP / (TP + FP);
M.npv         = 100 * TN / (TN + FN);
M.f1          = 2 * M.precision * M.sensitivity / (M.precision + M.sensitivity);
M.balancedAcc = (M.sensitivity + M.specificity) / 2;
M.gmean       = sqrt(M.sensitivity * M.specificity);
M.mcc = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / n^2;
M.kappa = ((TP + TN) / n - pe) / (1 - pe);
M.auc = NaN;
if nargin >= 3 && ~isempty(score) && all(isfinite(score)) && numel(unique(t)) == 2
    [~, ~, ~, M.auc] = perfcurve(t, score(:), true);
end
end
