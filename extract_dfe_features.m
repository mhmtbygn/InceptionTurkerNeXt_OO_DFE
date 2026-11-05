function X = extract_dfe_features(net, files, cfg)
%EXTRACT_DFE_FEATURES Global + patch GAP features (Section 2.3, Steps 2-5).
%   Each image is resized to 224x224, grey images are copied to three
%   channels, the image is divided into 196 non-overlapping 16x16 patches and
%   every patch is resized to 224x224 (bilinear) before it enters the frozen
%   backbone. The output row is [global(768) patch1(768) ... patch196(768)],
%   that is 197 x 768 = 151,296 features per image.
n  = numel(files);
ps = cfg.patchSize;
np = (cfg.inputSize(1) / ps) * (cfg.inputSize(2) / ps);
X  = zeros(n, 768 * (np + 1), 'single');
for i = 1:n
    im = imread(files{i});
    if size(im, 3) == 1, im = repmat(im, 1, 1, 3); end
    im = imresize(im, cfg.inputSize, 'bilinear');
    batch = zeros([cfg.inputSize 3 np + 1], 'like', im);
    batch(:, :, :, 1) = im;
    u = 1;
    for r = 1:ps:cfg.inputSize(1)
        for c = 1:ps:cfg.inputSize(2)
            u = u + 1;
            batch(:, :, :, u) = imresize(im(r:r+ps-1, c:c+ps-1, :), cfg.inputSize, 'bilinear');
        end
    end
    f = activations(net, batch, cfg.featLayer, 'OutputAs', 'rows', 'MiniBatchSize', np + 1);
    X(i, :) = reshape(f.', 1, []);   % 768 features of input 1, then input 2, ...
end
end
