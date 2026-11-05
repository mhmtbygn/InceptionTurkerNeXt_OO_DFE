# InceptionTurkerNeXt and InceptionTurkerNeXt-based DFE for osteoid osteoma CT classification

MATLAB code and pretrained weights for the manuscript

> Key S., Agar A., Goktas O.F., Baygin M., Dogan S., Tuncer T. *InceptionTurkerNeXt: A Compact CNN and Deep Feature Engineering Framework for Image-Level Osteoid Osteoma Classification on Computed Tomography* (Diagnostics, under revision).

The repository contains:

- the InceptionTurkerNeXt architecture (`itn_architecture.m`), exported layer by layer from the trained network;
- the pretrained multi-cancer weights (`weights/InceptionTurkerNeXt_multicancer_weights.mat`) and a loader (`load_pretrained_itn.m`);
- the deep feature engineering (DFE) pipeline: global + patch feature extraction, CWINCA feature selection and the tkNN/IMV classifier;
- the validation protocols O, A, B and C, diagnostic measures, Wilson and patient-cluster bootstrap confidence intervals, McNemar tests and the D1–D5 ablations;
- end-to-end fine-tuning of InceptionTurkerNeXt with Grad-CAM maps, and standard backbones on the same patient partitions.

The clinical osteoid osteoma (OO) CT images are not distributed (patient privacy, ethics approval of Firat University, decision 2025/04-25). They are available from the corresponding author on reasonable request.

## Requirements

MATLAB R2024a or later with Deep Learning Toolbox, Statistics and Machine Learning Toolbox and Image Processing Toolbox. `finetune_backbones.m` uses `imagePretrainedNetwork` (ResNet50, EfficientNet-B0) and, for ViT-B/16 and ConvNeXt-Tiny, a user-supplied `dlnetwork` file (see the script header).

## Pretrained weights

`weights/InceptionTurkerNeXt_multicancer_weights.mat` holds the multi-cancer network of the manuscript (26 classes, final validation accuracy 99.01%) as plain arrays, so it can be read in MATLAB (`load`) and in Python (`scipy.io.loadmat`):

| Variable | Content |
|---|---|
| `W.<layer>` | `Weights`/`Bias` (convolution, grouped convolution, fully connected) or `Scale`/`Offset`/`TrainedMean`/`TrainedVariance` (batch normalization) |
| `meanImage` | 1 x 1 x 3 channel mean of the zero-center input normalization |
| `classNames` | the 26 class names in network order (alphabetical folder names of the Kaggle Multi Cancer Dataset, doi:10.34740/KAGGLE/DSV/3415848) |
| `numLearnables` | 9,714,338 |

Number of learnable parameters: 9,714,338 for 26 classes, 9,695,882 for 2 classes and 10,463,344 for 1000 classes (only the fully connected layer depends on the number of classes, 769 parameters per class).

```matlab
net = load_pretrained_itn('weights/InceptionTurkerNeXt_multicancer_weights.mat');
f   = activations(net, img, 'gapool');   % 768 GAP features used by the DFE
```

## Architecture (as stored in the trained network)

- Stem: 4x4 convolution, 96 filters, stride 4 -> BN -> GELU (224x224x3 -> 56x56x96).
- ITN block with input t and width C (96, 192, 384, 768):
  1. pooling branch: 5x5 average pooling (stride 1, same padding) + t;
  2. grouped branch: 3x3 depthwise convolution + t;
  3. ConvNeXt branch: 3x3 depthwise convolution -> BN -> 1x1 grouped convolution to 4C -> GELU -> 1x1 convolution to C, + t;
  4. ResNet bottleneck: 1x1 convolution to C/4 -> BN -> 3x3 depthwise convolution -> GELU -> 1x1 grouped convolution to C, + t;
  5. depth concatenation (4C) -> BN -> GELU -> 1x1 convolution to C, + t;
  6. inverted bottleneck: BN -> 1x1 grouped convolution to 4C -> GELU -> 1x1 convolution to C, + output of step 5;
  7. BN.
- Downsampling: 2x2 grouped convolution, stride 2, 2C filters -> BN -> GELU.
- Output: global average pooling (768) -> fully connected -> softmax.

## Data layout

`data/oo_manifest.csv` with one row per axial CT image:

```
path,label,patient_id
data/oo/P001_s1.png,OO,P001
data/control/C017_s3.png,Control,C017
```

All slices of one patient must share the same `patient_id`; the patient-level protocols use it to keep a patient in one partition.

## Running

```matlab
test_smoke            % architecture, weights file and synthetic run of the pipeline
main                  % DFE features, protocols O/A/B/C, CIs, ablations -> results/
finetune_itn_oo       % end-to-end fine-tuning (no DFE) + Grad-CAM maps and rating sheet
finetune_backbones    % ResNet50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16 on the Protocol B partitions
train_backbone        % optional: retrain InceptionTurkerNeXt on the multi-cancer dataset
```

Settings are in `config_itn.m`. Values that are not reported in the manuscript are marked `[default]` there; set them to the values of your own run before comparing numbers.

## Validation protocols

| Protocol | Partitioning | Where CWINCA and tkNN/IMV are fitted |
|---|---|---|
| O | image-level 10-fold CV | all images (original analysis, optimistic reference) |
| A | patient-grouped 10-fold CV | all images |
| B (primary) | patient-grouped 10-fold CV | nine training folds only (min–max limits, NCA weights, subset size, tkNN/IMV output) |
| C | patient-level hold-out (20% of patients) | development patients only |

Outputs in `results/`: `summary_metrics.csv` (image- and patient-level measures), `predictions_<protocol>.csv` (per-image predictions with patient ID and test fold), `protocolB_bootstrap_ci.mat`, `ablation_DFE.csv`, and the fine-tuning, Grad-CAM and baseline files.

## Notes

- The code was written from the manuscript and from the structure of the released network. It has not been executed in this repository; please run `test_smoke.m` first.
- In Protocol B, normalization and NCA are fitted once per outer training set and are not refit inside the internal CWINCA evaluation, as described in the manuscript.

## License

MIT (see `LICENSE`). Please cite the article when you use the code or the weights (see `CITATION.cff`).
