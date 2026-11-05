%MAIN InceptionTurkerNeXt-based DFE for image-level osteoid osteoma (OO)
% classification on CT, with the validation protocols of the revised
% manuscript (Diagnostics 4667895).
%
% Required input: data/oo_manifest.csv with columns
%   path        image file (axial CT slice)
%   label       OO or Control
%   patient_id  anonymised patient code (all slices of a patient share it)
% The clinical images are not distributed (see README).
clear; clc;
cfg = config_itn();
if ~exist(cfg.resultsDir, 'dir'), mkdir(cfg.resultsDir); end

% ---------------------------------------------------------------- data
T = readtable(cfg.ooManifest, 'TextType', 'string');
y   = 1 + double(strcmpi(T.label, "OO"));      % 1 = Control, 2 = OO (positive)
pid = T.patient_id;
pos = 2;
fprintf('%d images, %d patients (%d OO images, %d control images)\n', ...
    height(T), numel(unique(pid)), sum(y == pos), sum(y ~= pos));

% ---------------------------------------------------------------- features
if exist(cfg.featureFile, 'file')
    S = load(cfg.featureFile, 'X'); X = S.X;
else
    net = load_pretrained_itn(cfg.weightsFile);
    X = extract_dfe_features(net, cellstr(T.path), cfg);
    if ~exist(fileparts(cfg.featureFile), 'dir'), mkdir(fileparts(cfg.featureFile)); end
    save(cfg.featureFile, 'X', '-v7.3');
end
fprintf('Feature matrix: %d x %d\n', size(X, 1), size(X, 2));

% ---------------------------------------------------------------- protocols
protocols = {'O', 'A', 'B', 'C'};
rows = {};
for i = 1:numel(protocols)
    R = run_protocol(X, y, pid, protocols{i}, cfg);
    tested = R.fold > 0;
    M = compute_metrics(y(tested), R.pred(tested), R.score(tested), pos);
    rows(end+1, :) = metric_row(protocols{i}, 'image', M); %#ok<SAGROW>
    export_predictions(fullfile(cfg.resultsDir, ['predictions_' protocols{i} '.csv']), T, y, R, pos);
    if strcmp(protocols{i}, 'B')
        RB = R;
        [yP, pP, sP] = patient_majority_vote(y, R.pred, R.score, pid, pos);
        MP = compute_metrics(yP, pP, sP, pos);
        rows(end+1, :) = metric_row('B', 'patient', MP); %#ok<SAGROW>
        CI = cluster_bootstrap_ci(y, R.pred, R.score, pid, pos, cfg.bootstrapB, cfg.alpha, cfg.seed);
        save(fullfile(cfg.resultsDir, 'protocolB_bootstrap_ci.mat'), 'CI');
        fprintf('Protocol B accuracy %.2f%% (95%% CI %.2f-%.2f, patient-cluster bootstrap, B = %d)\n', ...
            M.accuracy, CI.accuracy(1), CI.accuracy(2), CI.B);
    end
    if strcmp(protocols{i}, 'O')
        RO = R;
        w = wilson_ci(M.TP + M.TN, M.n, cfg.alpha);
        fprintf('Protocol O accuracy %.2f%% (Wilson 95%% CI %.2f-%.2f)\n', M.accuracy, w(1), w(2));
    end
end
summary = cell2table(rows, 'VariableNames', {'protocol', 'level', 'n', 'TP', 'FN', 'FP', 'TN', ...
    'accuracy', 'sensitivity', 'specificity', 'precision', 'npv', 'f1', 'balanced_accuracy', 'gmean', 'mcc', 'kappa', 'auc'});
writetable(summary, fullfile(cfg.resultsDir, 'summary_metrics.csv'));
disp(summary);

% ---------------------------------------------------------------- ablations D1-D5
abl = run_ablations(X, y, pid, cfg, pos);
writetable(abl, fullfile(cfg.resultsDir, 'ablation_DFE.csv'));
disp(abl);

function r = metric_row(protocol, level, M)
r = {protocol, level, M.n, M.TP, M.FN, M.FP, M.TN, M.accuracy, M.sensitivity, M.specificity, ...
     M.precision, M.npv, M.f1, M.balancedAcc, M.gmean, M.mcc, M.kappa, M.auc};
end

function export_predictions(file, T, y, R, pos)
tested = R.fold > 0;
P = table(T.path(tested), T.patient_id(tested), y(tested) == pos, R.pred(tested) == pos, R.score(tested), R.fold(tested), ...
    'VariableNames', {'path', 'patient_id', 'true_OO', 'pred_OO', 'score_OO', 'test_fold'});
writetable(P, file);
end
