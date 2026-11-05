function A = run_ablations(X, y, pid, cfg, pos)
%RUN_ABLATIONS DFE ablation cases D1-D5 under protocols O and B.
%   D1 full model; D2 global features only; D3 patch features only;
%   D4 no CWINCA (all features); D5 single kNN instead of tkNN/IMV.
%   McNemar tests compare each case with D1 on the paired image predictions.
cases = {'D1', struct(); ...
         'D2', struct('cols', 1:768); ...
         'D3', struct('cols', 769:size(X, 2)); ...
         'D4', struct('useCWINCA', false); ...
         'D5', struct('useIMV', false)};
rows = {};
for proto = {'O', 'B'}
    preds = cell(size(cases, 1), 1);
    for i = 1:size(cases, 1)
        R = run_protocol(X, y, pid, proto{1}, cfg, cases{i, 2});
        preds{i} = R.pred;
        M = compute_metrics(y, R.pred, R.score, pos);
        ci = wilson_ci(M.TP + M.TN, M.n, cfg.alpha);
        if i == 1, p = NaN; else, p = mcnemar_exact(y, preds{1}, R.pred); end
        rows(end+1, :) = {proto{1}, cases{i, 1}, M.accuracy, ci(1), ci(2), p}; %#ok<AGROW>
    end
end
A = cell2table(rows, 'VariableNames', {'protocol', 'case', 'accuracy', 'wilson_low', 'wilson_high', 'mcnemar_p_vs_D1'});
end
