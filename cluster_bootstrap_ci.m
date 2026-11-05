function CI = cluster_bootstrap_ci(ytrue, ypred, score, pid, pos, B, alpha, seed)
%CLUSTER_BOOTSTRAP_CI Patient-cluster bootstrap CIs (percentile) of image-level measures.
%   Patients are resampled with replacement; all images of a drawn patient
%   enter the replicate. Returns CI.(measure) = [lower upper].
if nargin < 8, seed = 1; end
rng(seed, 'twister');
[ug, ~, gi] = unique(pid(:));
members = accumarray(gi, (1:numel(gi)).', [], @(v) {v});
names = {'accuracy', 'sensitivity', 'specificity', 'auc'};
vals = nan(B, numel(names));
for b = 1:B
    draw = randi(numel(ug), numel(ug), 1);
    idx = vertcat(members{draw});
    M = compute_metrics(ytrue(idx), ypred(idx), score(idx), pos);
    for j = 1:numel(names), vals(b, j) = M.(names{j}); end
end
for j = 1:numel(names)
    CI.(names{j}) = prctile(vals(:, j), 100 * [alpha / 2, 1 - alpha / 2]);
end
CI.B = B;
end
