function R = run_protocol(X, y, pid, protocol, cfg, opts)
%RUN_PROTOCOL Evaluate the DFE framework under protocol O, A, B or C.
%   O : image-level 10-fold CV; CWINCA and tkNN/IMV selected on all images
%       (original analysis, optimistic reference).
%   A : patient-grouped 10-fold CV; selection still on all images.
%   B : patient-grouped 10-fold CV; all selection steps inside the nine
%       training folds (primary protocol).
%   C : independent patient-level hold-out (cfg.holdoutFrac of patients).
%   Returns R.pred, R.score, R.fold (test fold of each image; 0 = not tested).
if nargin < 6, opts = struct(); end
y = y(:); pid = pid(:);
R.protocol = protocol; R.pred = nan(size(y)); R.score = nan(size(y)); R.fold = zeros(size(y));
switch upper(protocol)
    case {'O', 'A'}
        if strcmpi(protocol, 'O')
            fold = make_group_folds(y, [], cfg.outerFolds, cfg.seed);
        else
            fold = make_group_folds(y, pid, cfg.outerFolds, cfg.seed);
        end
        cols = 1:size(X, 2);
        if isfield(opts, 'cols') && ~isempty(opts.cols), cols = opts.cols; end
        Xc = X(:, cols);
        if ~isfield(opts, 'useCWINCA') || opts.useCWINCA
            [sel, nrm, R.cwinca] = cwinca_select(Xc, y, fold, cfg);   % selection on all images
        else
            nrm.mn = min(Xc, [], 1); nrm.mx = max(Xc, [], 1); sel = 1:size(Xc, 2);
        end
        Z = (Xc - nrm.mn) ./ max(nrm.mx - nrm.mn, eps('single'));
        model = tknn_select(Z(:, sel), y, fold, cfg);
        if isfield(opts, 'useIMV') && ~opts.useIMV
            [~, b] = max(model.acc);
            [R.pred, R.score] = knn_cv_predict(Z(:, sel), y, fold, model.bag(b, 2), ...
                model.distances{model.bag(b, 1)}, model.weights{model.bag(b, 3)});
        else
            R.pred = model.cvPred; R.score = model.cvScore;
        end
        R.fold = fold; R.model = model;
    case 'B'
        fold = make_group_folds(y, pid, cfg.outerFolds, cfg.seed);
        R.nSelected = zeros(cfg.outerFolds, 1);
        for k = 1:cfg.outerFolds
            te = fold == k; tr = ~te;
            o = run_dfe_pipeline(X(tr, :), y(tr), pid(tr), X(te, :), cfg, opts);
            R.pred(te) = o.pred; R.score(te) = o.score;
            R.nSelected(k) = numel(o.sel);
            fprintf('Protocol B fold %d/%d: %d features, fold accuracy %.2f%%\n', k, cfg.outerFolds, numel(o.sel), 100 * mean(o.pred == y(te)));
        end
        R.fold = fold;
    case 'C'
        rng(cfg.seed + 7, 'twister');
        [ug, ~, gi] = unique(pid);
        gLabel = accumarray(gi, y, [], @(v) mode(v));
        isTest = false(numel(ug), 1);
        for c = unique(gLabel).'
            idx = find(gLabel == c); idx = idx(randperm(numel(idx)));
            isTest(idx(1:round(cfg.holdoutFrac * numel(idx)))) = true;
        end
        te = isTest(gi); tr = ~te;
        o = run_dfe_pipeline(X(tr, :), y(tr), pid(tr), X(te, :), cfg, opts);
        R.pred(te) = o.pred; R.score(te) = o.score; R.fold(te) = 1;
        R.testPatients = ug(isTest); R.nSelected = numel(o.sel);
    otherwise
        error('Unknown protocol %s', protocol);
end
end
