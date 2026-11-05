%FINETUNE_BACKBONES Standard backbones on the Protocol B patient partitions.
% ResNet50 and EfficientNet-B0 are loaded with imagePretrainedNetwork
% (ImageNet weights, R2024a or later). ViT-B/16 and ConvNeXt-Tiny are read
% from MAT files that hold a dlnetwork named 'net' with an ImageNet backbone
% (for example created with visionTransformer or importNetworkFromPyTorch);
% their last fully connected layer is replaced by a 2-class layer here.
% Hyperparameters are cfg.base.* (set them to the values used in the paper).
clear; clc;
cfg = config_itn();
T = readtable(cfg.ooManifest, 'TextType', 'string');
y = categorical(T.label); pid = T.patient_id;
yNum = 1 + double(T.label == "OO");
fold = make_group_folds(yNum, pid, cfg.outerFolds, cfg.seed);
rows = {};
for m = 1:numel(cfg.base.models)
    name = cfg.base.models{m};
    pred = strings(height(T), 1); score = nan(height(T), 1);
    for k = 1:cfg.outerFolds
        tr = fold ~= k; te = fold == k;
        net = get_baseline(name, cfg);
        inSz = net.Layers(1).InputSize(1:2);
        aug = @(files, labels) augmentedImageDatastore([inSz 3], imageDatastore(files, 'Labels', labels), 'ColorPreprocessing', 'gray2rgb');
        opts = trainingOptions(cfg.base.solver, 'MiniBatchSize', cfg.base.miniBatch, ...
            'MaxEpochs', cfg.base.maxEpochs, 'InitialLearnRate', cfg.base.learnRate, ...
            'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
        net = trainnet(aug(T.path(tr), y(tr)), net, 'crossentropy', opts);
        s = minibatchpredict(net, aug(T.path(te), y(te)));
        cls = categories(y);
        p = scores2label(s, cls);
        pred(te) = string(p); score(te) = s(:, strcmp(cls, 'OO'));
    end
    M = compute_metrics(yNum, 1 + double(pred == "OO"), score, 2);
    rows(end+1, :) = {name, sum(pred == string(y)), M.accuracy, M.f1}; %#ok<SAGROW>
    fprintf('%s: accuracy %.2f%%, F1 %.2f%%\n', name, M.accuracy, M.f1);
end
writetable(cell2table(rows, 'VariableNames', {'model', 'correct', 'accuracy', 'f1_OO'}), ...
    fullfile(cfg.resultsDir, 'baselines_protocolB.csv'));

function net = get_baseline(name, cfg)
switch name
    case {'resnet50', 'efficientnetb0'}
        net = imagePretrainedNetwork(name, 'NumClasses', 2);
    case 'convnext_tiny'
        S = load(cfg.base.convnextFile, 'net'); net = replace_head(S.net);
    case 'vit_b16'
        S = load(strrep(cfg.base.convnextFile, 'convnext_tiny', 'vit_b16'), 'net'); net = replace_head(S.net);
    otherwise
        error('Unknown baseline %s', name);
end
end

function net = replace_head(net)
fcIdx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers), 1, 'last');
fc = net.Layers(fcIdx);
net = replaceLayer(net, fc.Name, fullyConnectedLayer(2, 'Name', fc.Name));
net = initialize(net);
end
