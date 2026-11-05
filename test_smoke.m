%TEST_SMOKE Quick checks of the architecture, the weights file and the
% CWINCA/tkNN/protocol code on synthetic features (no clinical data needed).
clear; clc;
cfg = config_itn();

% 1) architecture: 9,714,338 learnables for 26 classes
lg = itn_architecture(26);
dl = dlnetwork(removeLayers(lg, 'classoutput'));
nLearn = sum(cellfun(@numel, dl.Learnables.Value));
fprintf('Learnable parameters: %d\n', nLearn);
assert(nLearn == 9714338, 'unexpected parameter count');

% 2) released weights (skipped when the file is absent)
if exist(cfg.weightsFile, 'file')
    net = load_pretrained_itn(cfg.weightsFile);
    s = predict(net, uint8(randi(255, [224 224 3])));
    assert(numel(s) == 26);
    fprintf('Pretrained network loaded: %d classes, first class %s\n', numel(s), string(net.Layers(end).Classes(1)));
end

% 3) synthetic two-class problem with patient structure
rng(0);
nPat = 30; perPat = 3; d = 400;
pid = repelem((1:nPat).', perPat);
y = 1 + (pid > nPat / 2);
X = single(randn(numel(y), d));
X(:, 1:15) = X(:, 1:15) + 1.5 * (y == 2);
small = cfg;
small.tknn.distances = {'cityblock', 'euclidean'}; small.tknn.k = 1:3;
small.cwinca.step = 5; small.cwinca.innerFolds = 5; small.tknn.innerFolds = 5; small.outerFolds = 5;
for p = {'O', 'A', 'B', 'C'}
    R = run_protocol(X, y, pid, p{1}, small);
    t = R.fold > 0;
    M = compute_metrics(y(t), R.pred(t), R.score(t), 2);
    fprintf('Protocol %s: accuracy %.2f%% (%d images)\n', p{1}, M.accuracy, sum(t));
    assert(M.accuracy > 60, 'synthetic accuracy too low');
end
CI = cluster_bootstrap_ci(y(t), R.pred(t), R.score(t), pid(t), 2, 200, 0.05, 1);
fprintf('Bootstrap CI computed (B = %d)\n', CI.B);
disp('Smoke test passed.');
