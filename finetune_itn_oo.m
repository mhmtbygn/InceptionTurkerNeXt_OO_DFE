%FINETUNE_ITN_OO End-to-end fine-tuning of InceptionTurkerNeXt on the OO
% images (no DFE) under the Protocol B patient partitions, with Grad-CAM
% maps for the correctly classified OO test images.
% Outputs: results/predictions_ITN_finetune.csv, results/ITN_finetune_metrics.csv,
%          results/gradcam/*.png and results/gradcam/gradcam_rating_sheet.csv
% The rating sheet has an empty column 'focus_on_nidus' (1/0) for the
% expert review; the proportion reported in the manuscript is the mean of
% that column over the rated images.
clear; clc;
cfg = config_itn();
T = readtable(cfg.ooManifest, 'TextType', 'string');
y = categorical(T.label); pid = T.patient_id;
yNum = 1 + double(T.label == "OO");
fold = make_group_folds(yNum, pid, cfg.outerFolds, cfg.seed);   % same partitions as Protocol B
gdir = fullfile(cfg.resultsDir, 'gradcam'); if ~exist(gdir, 'dir'), mkdir(gdir); end
pred = strings(height(T), 1); score = nan(height(T), 1);
sheet = {};
for k = 1:cfg.outerFolds
    tr = fold ~= k; te = fold == k;
    base = load_pretrained_itn(cfg.weightsFile);
    lg = layerGraph(base);
    lg = replaceLayer(lg, 'fc', fullyConnectedLayer(2, 'Name', 'fc'));
    lg = replaceLayer(lg, 'classoutput', classificationLayer('Name', 'classoutput'));
    imdsTr = imageDatastore(T.path(tr), 'Labels', y(tr));
    imdsTe = imageDatastore(T.path(te), 'Labels', y(te));
    aug = @(ds) augmentedImageDatastore([cfg.inputSize 3], ds, 'ColorPreprocessing', 'gray2rgb');
    opts = trainingOptions(cfg.itnft.solver, 'MiniBatchSize', cfg.itnft.miniBatch, ...
        'MaxEpochs', cfg.itnft.maxEpochs, 'InitialLearnRate', cfg.itnft.learnRate, ...
        'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
    net = trainNetwork(aug(imdsTr), lg, opts);
    [p, s] = classify(net, aug(imdsTe));
    cls = net.Layers(end).Classes;
    pred(te) = string(p); score(te) = s(:, cls == "OO");
    % Grad-CAM from the last ITN stage for correctly classified OO images
    idx = find(te & y == "OO");
    for j = idx.'
        if pred(j) ~= "OO", continue; end
        im = imread(T.path(j));
        if size(im, 3) == 1, im = repmat(im, 1, 1, 3); end
        im = imresize(im, cfg.inputSize);
        map = gradCAM(net, im, categorical("OO"), 'FeatureLayer', 'addition_23', 'ReductionLayer', 'softmax');
        f = figure('Visible', 'off'); imshow(im); hold on;
        imagesc(map, 'AlphaData', 0.45); colormap jet; axis off;
        [~, nm] = fileparts(T.path(j));
        out = fullfile(gdir, sprintf('fold%02d_%s_%s.png', k, T.patient_id(j), nm));
        exportgraphics(f, out, 'Resolution', 300); close(f);
        sheet(end+1, :) = {T.path(j), T.patient_id(j), k, out, NaN}; %#ok<SAGROW>
    end
    fprintf('Fold %d/%d: accuracy %.2f%%\n', k, cfg.outerFolds, 100 * mean(pred(te) == string(y(te))));
end
M = compute_metrics(yNum, 1 + double(pred == "OO"), score, 2);
writetable(struct2table(M), fullfile(cfg.resultsDir, 'ITN_finetune_metrics.csv'));
writetable(table(T.path, T.patient_id, string(y), pred, score, fold, ...
    'VariableNames', {'path', 'patient_id', 'true', 'pred', 'score_OO', 'test_fold'}), ...
    fullfile(cfg.resultsDir, 'predictions_ITN_finetune.csv'));
writetable(cell2table(sheet, 'VariableNames', {'path', 'patient_id', 'test_fold', 'gradcam_png', 'focus_on_nidus'}), ...
    fullfile(gdir, 'gradcam_rating_sheet.csv'));
fprintf('Fine-tuned InceptionTurkerNeXt: accuracy %.2f%%, F1 %.2f%%\n', M.accuracy, M.f1);
