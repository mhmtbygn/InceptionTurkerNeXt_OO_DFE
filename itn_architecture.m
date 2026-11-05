function lgraph = itn_architecture(numClasses)
%ITN_ARCHITECTURE Layer graph of InceptionTurkerNeXt (125 layers, 160 connections).
%   lgraph = itn_architecture(numClasses) returns the untrained layer graph.
%   The layer list and the connections were exported automatically from the
%   trained multi-cancer network (Net_MC), so layer names match the released
%   weights file. Stages: stem (4x4/4, 96) -> ITN(96) -> down(2x2/2) -> ITN(192)
%   -> down -> ITN(384) -> down -> ITN(768) -> BN -> GAP -> FC -> softmax.
%   Learnable parameters: 9,714,338 for 26 classes (768*N + N in the FC layer).
if nargin < 1, numClasses = 26; end
L = {};
L{end+1} = imageInputLayer([224 224 3],'Name','imageinput','Normalization','zerocenter');
L{end+1} = convolution2dLayer([4 4],96,'Stride',[4 4],'Padding','same','Name','conv');
L{end+1} = batchNormalizationLayer('Name','batchnorm','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu');
L{end+1} = averagePooling2dLayer([5 5],'Stride',[1 1],'Padding','same','Name','avgpool2d');
L{end+1} = additionLayer(2,'Name','addition');
L{end+1} = groupedConvolution2dLayer([3 3],1,96,'Stride',[1 1],'Padding','same','Name','groupedconv');
L{end+1} = additionLayer(2,'Name','addition_1');
L{end+1} = groupedConvolution2dLayer([3 3],1,96,'Stride',[1 1],'Padding','same','Name','groupedconv_1');
L{end+1} = batchNormalizationLayer('Name','batchnorm_1','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,96,'Stride',[1 1],'Padding','same','Name','groupedconv_2');
L{end+1} = geluLayer('Name','gelu_1');
L{end+1} = convolution2dLayer([1 1],96,'Stride',[1 1],'Padding','same','Name','conv_1');
L{end+1} = additionLayer(2,'Name','addition_2');
L{end+1} = convolution2dLayer([1 1],24,'Stride',[1 1],'Padding','same','Name','conv_2');
L{end+1} = batchNormalizationLayer('Name','batchnorm_2','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([3 3],1,24,'Stride',[1 1],'Padding','same','Name','groupedconv_3');
L{end+1} = geluLayer('Name','gelu_2');
L{end+1} = groupedConvolution2dLayer([1 1],4,24,'Stride',[1 1],'Padding','same','Name','groupedconv_4');
L{end+1} = additionLayer(2,'Name','addition_3');
L{end+1} = depthConcatenationLayer(4,'Name','depthcat');
L{end+1} = batchNormalizationLayer('Name','batchnorm_3','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_3');
L{end+1} = convolution2dLayer([1 1],96,'Stride',[1 1],'Padding','same','Name','conv_3');
L{end+1} = additionLayer(2,'Name','addition_4');
L{end+1} = batchNormalizationLayer('Name','batchnorm_4','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,96,'Stride',[1 1],'Padding','same','Name','groupedconv_5');
L{end+1} = geluLayer('Name','gelu_4');
L{end+1} = convolution2dLayer([1 1],96,'Stride',[1 1],'Padding','same','Name','conv_4');
L{end+1} = additionLayer(2,'Name','addition_5');
L{end+1} = batchNormalizationLayer('Name','batchnorm_5','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([2 2],2,96,'Stride',[2 2],'Padding','same','Name','groupedconv_12');
L{end+1} = batchNormalizationLayer('Name','batchnorm_6','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_5');
L{end+1} = averagePooling2dLayer([5 5],'Stride',[1 1],'Padding','same','Name','avgpool2d_1');
L{end+1} = additionLayer(2,'Name','addition_6');
L{end+1} = groupedConvolution2dLayer([3 3],1,192,'Stride',[1 1],'Padding','same','Name','groupedconv_6');
L{end+1} = additionLayer(2,'Name','addition_7');
L{end+1} = groupedConvolution2dLayer([3 3],1,192,'Stride',[1 1],'Padding','same','Name','groupedconv_7');
L{end+1} = batchNormalizationLayer('Name','batchnorm_7','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,192,'Stride',[1 1],'Padding','same','Name','groupedconv_8');
L{end+1} = geluLayer('Name','gelu_6');
L{end+1} = convolution2dLayer([1 1],192,'Stride',[1 1],'Padding','same','Name','conv_5');
L{end+1} = additionLayer(2,'Name','addition_8');
L{end+1} = convolution2dLayer([1 1],48,'Stride',[1 1],'Padding','same','Name','conv_6');
L{end+1} = batchNormalizationLayer('Name','batchnorm_8','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([3 3],1,48,'Stride',[1 1],'Padding','same','Name','groupedconv_9');
L{end+1} = geluLayer('Name','gelu_7');
L{end+1} = groupedConvolution2dLayer([1 1],4,48,'Stride',[1 1],'Padding','same','Name','groupedconv_10');
L{end+1} = additionLayer(2,'Name','addition_9');
L{end+1} = depthConcatenationLayer(4,'Name','depthcat_1');
L{end+1} = batchNormalizationLayer('Name','batchnorm_9','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_8');
L{end+1} = convolution2dLayer([1 1],192,'Stride',[1 1],'Padding','same','Name','conv_7');
L{end+1} = additionLayer(2,'Name','addition_10');
L{end+1} = batchNormalizationLayer('Name','batchnorm_10','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,192,'Stride',[1 1],'Padding','same','Name','groupedconv_11');
L{end+1} = geluLayer('Name','gelu_9');
L{end+1} = convolution2dLayer([1 1],192,'Stride',[1 1],'Padding','same','Name','conv_8');
L{end+1} = additionLayer(2,'Name','addition_11');
L{end+1} = batchNormalizationLayer('Name','batchnorm_11','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([2 2],2,192,'Stride',[2 2],'Padding','same','Name','groupedconv_19');
L{end+1} = batchNormalizationLayer('Name','batchnorm_12','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_10');
L{end+1} = averagePooling2dLayer([5 5],'Stride',[1 1],'Padding','same','Name','avgpool2d_2');
L{end+1} = additionLayer(2,'Name','addition_12');
L{end+1} = groupedConvolution2dLayer([3 3],1,384,'Stride',[1 1],'Padding','same','Name','groupedconv_13');
L{end+1} = additionLayer(2,'Name','addition_13');
L{end+1} = groupedConvolution2dLayer([3 3],1,384,'Stride',[1 1],'Padding','same','Name','groupedconv_14');
L{end+1} = batchNormalizationLayer('Name','batchnorm_13','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,384,'Stride',[1 1],'Padding','same','Name','groupedconv_15');
L{end+1} = geluLayer('Name','gelu_11');
L{end+1} = convolution2dLayer([1 1],384,'Stride',[1 1],'Padding','same','Name','conv_9');
L{end+1} = additionLayer(2,'Name','addition_14');
L{end+1} = convolution2dLayer([1 1],96,'Stride',[1 1],'Padding','same','Name','conv_10');
L{end+1} = batchNormalizationLayer('Name','batchnorm_14','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([3 3],1,96,'Stride',[1 1],'Padding','same','Name','groupedconv_16');
L{end+1} = geluLayer('Name','gelu_12');
L{end+1} = groupedConvolution2dLayer([1 1],4,96,'Stride',[1 1],'Padding','same','Name','groupedconv_17');
L{end+1} = additionLayer(2,'Name','addition_15');
L{end+1} = depthConcatenationLayer(4,'Name','depthcat_2');
L{end+1} = batchNormalizationLayer('Name','batchnorm_15','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_13');
L{end+1} = convolution2dLayer([1 1],384,'Stride',[1 1],'Padding','same','Name','conv_11');
L{end+1} = additionLayer(2,'Name','addition_16');
L{end+1} = batchNormalizationLayer('Name','batchnorm_16','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,384,'Stride',[1 1],'Padding','same','Name','groupedconv_18');
L{end+1} = geluLayer('Name','gelu_14');
L{end+1} = convolution2dLayer([1 1],384,'Stride',[1 1],'Padding','same','Name','conv_12');
L{end+1} = additionLayer(2,'Name','addition_17');
L{end+1} = batchNormalizationLayer('Name','batchnorm_17','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([2 2],2,384,'Stride',[2 2],'Padding','same','Name','groupedconv_26');
L{end+1} = batchNormalizationLayer('Name','batchnorm_18','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_15');
L{end+1} = averagePooling2dLayer([5 5],'Stride',[1 1],'Padding','same','Name','avgpool2d_3');
L{end+1} = additionLayer(2,'Name','addition_18');
L{end+1} = groupedConvolution2dLayer([3 3],1,768,'Stride',[1 1],'Padding','same','Name','groupedconv_20');
L{end+1} = additionLayer(2,'Name','addition_19');
L{end+1} = groupedConvolution2dLayer([3 3],1,768,'Stride',[1 1],'Padding','same','Name','groupedconv_21');
L{end+1} = batchNormalizationLayer('Name','batchnorm_19','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,768,'Stride',[1 1],'Padding','same','Name','groupedconv_22');
L{end+1} = geluLayer('Name','gelu_16');
L{end+1} = convolution2dLayer([1 1],768,'Stride',[1 1],'Padding','same','Name','conv_13');
L{end+1} = additionLayer(2,'Name','addition_20');
L{end+1} = convolution2dLayer([1 1],192,'Stride',[1 1],'Padding','same','Name','conv_14');
L{end+1} = batchNormalizationLayer('Name','batchnorm_20','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([3 3],1,192,'Stride',[1 1],'Padding','same','Name','groupedconv_23');
L{end+1} = geluLayer('Name','gelu_17');
L{end+1} = groupedConvolution2dLayer([1 1],4,192,'Stride',[1 1],'Padding','same','Name','groupedconv_24');
L{end+1} = additionLayer(2,'Name','addition_21');
L{end+1} = depthConcatenationLayer(4,'Name','depthcat_3');
L{end+1} = batchNormalizationLayer('Name','batchnorm_21','Epsilon',1e-05);
L{end+1} = geluLayer('Name','gelu_18');
L{end+1} = convolution2dLayer([1 1],768,'Stride',[1 1],'Padding','same','Name','conv_15');
L{end+1} = additionLayer(2,'Name','addition_22');
L{end+1} = batchNormalizationLayer('Name','batchnorm_22','Epsilon',1e-05);
L{end+1} = groupedConvolution2dLayer([1 1],4,768,'Stride',[1 1],'Padding','same','Name','groupedconv_25');
L{end+1} = geluLayer('Name','gelu_19');
L{end+1} = convolution2dLayer([1 1],768,'Stride',[1 1],'Padding','same','Name','conv_16');
L{end+1} = additionLayer(2,'Name','addition_23');
L{end+1} = batchNormalizationLayer('Name','batchnorm_23','Epsilon',1e-05);
L{end+1} = globalAveragePooling2dLayer('Name','gapool');
L{end+1} = fullyConnectedLayer(numClasses,'Name','fc');
L{end+1} = softmaxLayer('Name','softmax');
L{end+1} = classificationLayer('Name','classoutput');
lgraph = layerGraph();
for i = 1:numel(L)
    lgraph = addLayers(lgraph, L{i});
end
C = {
    'imageinput', 'conv'
    'conv', 'batchnorm'
    'batchnorm', 'gelu'
    'gelu', 'avgpool2d'
    'gelu', 'addition/in2'
    'gelu', 'groupedconv'
    'gelu', 'addition_1/in2'
    'gelu', 'groupedconv_1'
    'gelu', 'addition_2/in1'
    'gelu', 'conv_2'
    'gelu', 'addition_3/in2'
    'gelu', 'addition_4/in2'
    'avgpool2d', 'addition/in1'
    'addition', 'depthcat/in4'
    'groupedconv', 'addition_1/in1'
    'addition_1', 'depthcat/in3'
    'groupedconv_1', 'batchnorm_1'
    'batchnorm_1', 'groupedconv_2'
    'groupedconv_2', 'gelu_1'
    'gelu_1', 'conv_1'
    'conv_1', 'addition_2/in2'
    'addition_2', 'depthcat/in2'
    'conv_2', 'batchnorm_2'
    'batchnorm_2', 'groupedconv_3'
    'groupedconv_3', 'gelu_2'
    'gelu_2', 'groupedconv_4'
    'groupedconv_4', 'addition_3/in1'
    'addition_3', 'depthcat/in1'
    'depthcat', 'batchnorm_3'
    'batchnorm_3', 'gelu_3'
    'gelu_3', 'conv_3'
    'conv_3', 'addition_4/in1'
    'addition_4', 'batchnorm_4'
    'addition_4', 'addition_5/in2'
    'batchnorm_4', 'groupedconv_5'
    'groupedconv_5', 'gelu_4'
    'gelu_4', 'conv_4'
    'conv_4', 'addition_5/in1'
    'addition_5', 'batchnorm_5'
    'batchnorm_5', 'groupedconv_12'
    'groupedconv_12', 'batchnorm_6'
    'batchnorm_6', 'gelu_5'
    'gelu_5', 'avgpool2d_1'
    'gelu_5', 'addition_6/in2'
    'gelu_5', 'groupedconv_6'
    'gelu_5', 'addition_7/in2'
    'gelu_5', 'groupedconv_7'
    'gelu_5', 'addition_8/in1'
    'gelu_5', 'conv_6'
    'gelu_5', 'addition_9/in2'
    'gelu_5', 'addition_10/in2'
    'avgpool2d_1', 'addition_6/in1'
    'addition_6', 'depthcat_1/in4'
    'groupedconv_6', 'addition_7/in1'
    'addition_7', 'depthcat_1/in3'
    'groupedconv_7', 'batchnorm_7'
    'batchnorm_7', 'groupedconv_8'
    'groupedconv_8', 'gelu_6'
    'gelu_6', 'conv_5'
    'conv_5', 'addition_8/in2'
    'addition_8', 'depthcat_1/in2'
    'conv_6', 'batchnorm_8'
    'batchnorm_8', 'groupedconv_9'
    'groupedconv_9', 'gelu_7'
    'gelu_7', 'groupedconv_10'
    'groupedconv_10', 'addition_9/in1'
    'addition_9', 'depthcat_1/in1'
    'depthcat_1', 'batchnorm_9'
    'batchnorm_9', 'gelu_8'
    'gelu_8', 'conv_7'
    'conv_7', 'addition_10/in1'
    'addition_10', 'batchnorm_10'
    'addition_10', 'addition_11/in2'
    'batchnorm_10', 'groupedconv_11'
    'groupedconv_11', 'gelu_9'
    'gelu_9', 'conv_8'
    'conv_8', 'addition_11/in1'
    'addition_11', 'batchnorm_11'
    'batchnorm_11', 'groupedconv_19'
    'groupedconv_19', 'batchnorm_12'
    'batchnorm_12', 'gelu_10'
    'gelu_10', 'avgpool2d_2'
    'gelu_10', 'addition_12/in2'
    'gelu_10', 'groupedconv_13'
    'gelu_10', 'addition_13/in2'
    'gelu_10', 'groupedconv_14'
    'gelu_10', 'addition_14/in1'
    'gelu_10', 'conv_10'
    'gelu_10', 'addition_15/in2'
    'gelu_10', 'addition_16/in2'
    'avgpool2d_2', 'addition_12/in1'
    'addition_12', 'depthcat_2/in4'
    'groupedconv_13', 'addition_13/in1'
    'addition_13', 'depthcat_2/in3'
    'groupedconv_14', 'batchnorm_13'
    'batchnorm_13', 'groupedconv_15'
    'groupedconv_15', 'gelu_11'
    'gelu_11', 'conv_9'
    'conv_9', 'addition_14/in2'
    'addition_14', 'depthcat_2/in2'
    'conv_10', 'batchnorm_14'
    'batchnorm_14', 'groupedconv_16'
    'groupedconv_16', 'gelu_12'
    'gelu_12', 'groupedconv_17'
    'groupedconv_17', 'addition_15/in1'
    'addition_15', 'depthcat_2/in1'
    'depthcat_2', 'batchnorm_15'
    'batchnorm_15', 'gelu_13'
    'gelu_13', 'conv_11'
    'conv_11', 'addition_16/in1'
    'addition_16', 'batchnorm_16'
    'addition_16', 'addition_17/in2'
    'batchnorm_16', 'groupedconv_18'
    'groupedconv_18', 'gelu_14'
    'gelu_14', 'conv_12'
    'conv_12', 'addition_17/in1'
    'addition_17', 'batchnorm_17'
    'batchnorm_17', 'groupedconv_26'
    'groupedconv_26', 'batchnorm_18'
    'batchnorm_18', 'gelu_15'
    'gelu_15', 'avgpool2d_3'
    'gelu_15', 'addition_18/in2'
    'gelu_15', 'groupedconv_20'
    'gelu_15', 'addition_19/in2'
    'gelu_15', 'groupedconv_21'
    'gelu_15', 'addition_20/in1'
    'gelu_15', 'conv_14'
    'gelu_15', 'addition_21/in2'
    'gelu_15', 'addition_22/in2'
    'avgpool2d_3', 'addition_18/in1'
    'addition_18', 'depthcat_3/in4'
    'groupedconv_20', 'addition_19/in1'
    'addition_19', 'depthcat_3/in3'
    'groupedconv_21', 'batchnorm_19'
    'batchnorm_19', 'groupedconv_22'
    'groupedconv_22', 'gelu_16'
    'gelu_16', 'conv_13'
    'conv_13', 'addition_20/in2'
    'addition_20', 'depthcat_3/in2'
    'conv_14', 'batchnorm_20'
    'batchnorm_20', 'groupedconv_23'
    'groupedconv_23', 'gelu_17'
    'gelu_17', 'groupedconv_24'
    'groupedconv_24', 'addition_21/in1'
    'addition_21', 'depthcat_3/in1'
    'depthcat_3', 'batchnorm_21'
    'batchnorm_21', 'gelu_18'
    'gelu_18', 'conv_15'
    'conv_15', 'addition_22/in1'
    'addition_22', 'batchnorm_22'
    'addition_22', 'addition_23/in2'
    'batchnorm_22', 'groupedconv_25'
    'groupedconv_25', 'gelu_19'
    'gelu_19', 'conv_16'
    'conv_16', 'addition_23/in1'
    'addition_23', 'batchnorm_23'
    'batchnorm_23', 'gapool'
    'gapool', 'fc'
    'fc', 'softmax'
    'softmax', 'classoutput'
    };
for i = 1:size(C,1)
    lgraph = connectLayers(lgraph, C{i,1}, C{i,2});
end
end
