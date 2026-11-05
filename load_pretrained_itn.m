function net = load_pretrained_itn(weightsFile)
%LOAD_PRETRAINED_ITN Rebuild the pretrained multi-cancer InceptionTurkerNeXt.
%   net = load_pretrained_itn(weightsFile) builds the layer graph with
%   itn_architecture(26), copies the released weights (portable .mat with
%   plain arrays) into every learnable layer and assembles a DAGNetwork.
%
%   Weight layout in the file follows the internal MATLAB storage:
%     convolution  [h w c f]          bias [1 1 f]
%     grouped conv [h w cpg fpg*G]    bias [1 1 fpg*G]   (reshaped here to 5-D)
%     batch norm   [1 1 C] for Scale, Offset, TrainedMean, TrainedVariance
%     fully conn.  [1 1 768 26]       bias [1 1 26]      (reshaped to [26 768])
S = load(weightsFile, 'W', 'meanImage', 'classNames');
W = S.W;
lgraph = itn_architecture(26);
names = {lgraph.Layers.Name};
for i = 1:numel(names)
    n  = names{i};
    ly = lgraph.Layers(strcmp({lgraph.Layers.Name}, n));
    if isa(ly, 'nnet.cnn.layer.ImageInputLayer')
        ly = imageInputLayer(ly.InputSize, 'Name', n, 'Normalization', 'zerocenter', ...
            'Mean', reshape(single(S.meanImage), 1, 1, []));
    elseif ~isfield(W, n)
        continue
    elseif isa(ly, 'nnet.cnn.layer.GroupedConvolution2DLayer')
        w = single(W.(n).Weights); b = single(W.(n).Bias);
        fpg = ly.NumFiltersPerGroup; G = ly.NumGroups;
        ly.Weights = reshape(w, size(w,1), size(w,2), size(w,3), fpg, G);
        ly.Bias    = reshape(b, 1, 1, fpg, G);
    elseif isa(ly, 'nnet.cnn.layer.Convolution2DLayer')
        ly.Weights = single(W.(n).Weights);
        ly.Bias    = reshape(single(W.(n).Bias), 1, 1, []);
    elseif isa(ly, 'nnet.cnn.layer.BatchNormalizationLayer')
        ly.Scale           = reshape(single(W.(n).Scale), 1, 1, []);
        ly.Offset          = reshape(single(W.(n).Offset), 1, 1, []);
        ly.TrainedMean     = reshape(single(W.(n).TrainedMean), 1, 1, []);
        ly.TrainedVariance = reshape(single(W.(n).TrainedVariance), 1, 1, []);
    elseif isa(ly, 'nnet.cnn.layer.FullyConnectedLayer')
        w = squeeze(single(W.(n).Weights));          % 768 x 26
        ly.Weights = w.';                            % 26 x 768
        ly.Bias    = reshape(single(W.(n).Bias), [], 1);
    end
    lgraph = replaceLayer(lgraph, n, ly);
end
classNames = cellstr(string(S.classNames(:)));
lgraph = replaceLayer(lgraph, 'classoutput', classificationLayer('Name', 'classoutput', 'Classes', classNames));
net = assembleNetwork(lgraph);
end
