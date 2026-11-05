function cfg = config_itn()
%CONFIG_ITN Settings of the InceptionTurkerNeXt-based DFE pipeline.
%   Values marked [paper] follow the manuscript. Values marked [default]
%   are not reported in the manuscript; replace them with the settings
%   that were used before reproducing a reported number.

% ---- paths
cfg.ooManifest   = fullfile('data', 'oo_manifest.csv');   % columns: path,label,patient_id
cfg.mcTrainDir   = fullfile('data', 'multicancer', 'train');
cfg.mcTestDir    = fullfile('data', 'multicancer', 'test');
cfg.weightsFile  = fullfile('weights', 'InceptionTurkerNeXt_multicancer_weights.mat');
cfg.featureFile  = fullfile('features', 'oo_dfe_features.mat');
cfg.resultsDir   = 'results';

% ---- backbone training on the multi-cancer dataset [paper, Section 3.2]
cfg.train.miniBatch      = 128;
cfg.train.solver         = 'sgdm';
cfg.train.initLearnRate  = 0.01;
cfg.train.dropFactor     = 0.1;
cfg.train.dropPeriod     = 10;     % [default] not reported
cfg.train.l2             = 1e-4;
cfg.train.maxEpochs      = 20;
cfg.train.momentum       = 0.9;
cfg.train.valFraction    = 0.20;   % 80:20 split of the training part

% ---- DFE feature extraction [paper, Section 2.3]
cfg.inputSize  = [224 224];
cfg.patchSize  = 16;               % 196 non-overlapping patches
cfg.featLayer  = 'gapool';         % 768 features per input

% ---- CWINCA [paper]
cfg.cwinca.tr1   = 0.50;
cfg.cwinca.tr2   = 0.99;
cfg.cwinca.step  = 1;              % evaluate every subset size between start and stop
cfg.cwinca.knnK  = 1;              % internal loss classifier: kNN, k = 1, L1, equal weights
cfg.cwinca.knnDistance = 'cityblock';
cfg.cwinca.innerFolds  = 10;

% ---- tkNN parameter bag [paper, Eq. (24)]: 4 x 10 x 3 = 120 outcomes
cfg.tknn.distances = {'cityblock', 'euclidean', 'cosine', 'spearman'};
cfg.tknn.k         = 1:10;
cfg.tknn.weights   = {'squaredinverse', 'equal', 'inverse'};
cfg.tknn.innerFolds = 10;

% ---- validation protocols [paper, Section 2.4]
cfg.outerFolds   = 10;
cfg.holdoutFrac  = 0.20;           % Protocol C: fraction of patients held out
cfg.seed         = 1;              % [default]

% ---- uncertainty
cfg.bootstrapB   = 2000;           % [default] patient-cluster bootstrap resamples
cfg.alpha        = 0.05;

% ---- baselines on the Protocol B partitions [default settings]
cfg.base.models     = {'resnet50', 'efficientnetb0', 'convnext_tiny', 'vit_b16'};
cfg.base.miniBatch  = 32;
cfg.base.maxEpochs  = 10;
cfg.base.learnRate  = 1e-4;
cfg.base.solver     = 'adam';
cfg.base.convnextFile = fullfile('weights', 'convnext_tiny_imagenet.mat');  % imported network (see README)
end
