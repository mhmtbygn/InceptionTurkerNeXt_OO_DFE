function out = run_dfe_pipeline(Xtr, ytr, gtr, Xte, cfg, opts)
%RUN_DFE_PIPELINE CWINCA + tkNN fitted on a training set, applied to a test set.
%   Every data-dependent step (min-max limits, NCA weights, CWINCA subset
%   size, tkNN/IMV output) is fitted on (Xtr, ytr) with patient-grouped
%   internal folds (gtr = patient IDs). Xte is used only for prediction.
%   opts.cols       feature columns to use (ablations D2/D3), [] = all
%   opts.useCWINCA  false = ablation D4 (all features, normalized)
%   opts.useIMV     false = ablation D5 (best single kNN outcome)
if nargin < 6, opts = struct(); end
if ~isfield(opts, 'cols') || isempty(opts.cols), opts.cols = 1:size(Xtr, 2); end
if ~isfield(opts, 'useCWINCA'), opts.useCWINCA = true; end
if ~isfield(opts, 'useIMV'), opts.useIMV = true; end
Xtr = Xtr(:, opts.cols); Xte = Xte(:, opts.cols);
innerFold = make_group_folds(ytr, gtr, cfg.cwinca.innerFolds, cfg.seed + 1);
if opts.useCWINCA
    [sel, nrm, cinfo] = cwinca_select(Xtr, ytr, innerFold, cfg);
else
    nrm.mn = min(Xtr, [], 1); nrm.mx = max(Xtr, [], 1); sel = 1:size(Xtr, 2);
    cinfo = struct('nSelected', numel(sel));
end
f = @(X) (X - nrm.mn) ./ max(nrm.mx - nrm.mn, eps('single'));
Ztr = f(Xtr); Zte = f(Xte);
Ztr = Ztr(:, sel); Zte = Zte(:, sel);
model = tknn_select(Ztr, ytr, innerFold, cfg);
if ~opts.useIMV
    [~, best] = max(model.acc);
    model.members = best; model.voted = false;
end
[out.pred, out.score] = tknn_apply(model, Ztr, ytr, Zte);
out.sel = opts.cols(sel); out.cwinca = cinfo; out.model = model;
end
