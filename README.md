This repository contains accompanying code for the article introducing
Meta-Dataset,
[https://arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

This code is provided here in order to give more details on the implementation
of the data-providing pipeline, our back-bones and models, as well as the
experimental setting.

See below for [user instructions](#user-instructions), including how to [install](#installation) the software, [download and convert](#downloading-and-converting-datasets) the data, and [train](#training) implemented models.

We are currently working on updating the code and improving the instructions to
facilitate designing and running new experiments.

This is not an officially supported Google product.

## Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples

_Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Kelvin Xu,
Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo
Larochelle_

Few-shot classification refers to learning a classifier for new classes given
only a few examples. While a plethora of models have emerged to tackle this
recently, we find the current procedure and datasets that are used to
systematically assess progress in this setting lacking. To address this, we
propose Meta-Dataset: a new benchmark for training and evaluating few-shot
classifiers that is large-scale, consists of multiple datasets, and presents
more natural and realistic tasks. The aim is to measure the ability of
state-of-the-art models to leverage diverse sources of data to achieve higher
generalization, and to evaluate that generalization ability in a more
challenging and realistic setting. We additionally measure robustness to
variations in the number of available examples and the number of classes.
Finally our extensive empirical evaluation leads us to identify weaknesses in
Prototypical Networks and MAML, two popular few-shot classification methods, and
to propose a new method, Proto-MAML, which achieves improved performance on our
benchmark.

# User instructions
## Installation
Meta-Dataset currently supports Python 2 only, and has not been tested with TensorFlow 2 yet.

- We recommend you follow [these instructions](https://www.tensorflow.org/install/pip?lang=python2) to install TensorFlow.
- A list of packages to install is available in `requirements.txt`, you can install them using `pip`.
- Clone the `meta-dataset` repository. Most command lines start with `python -m meta_dataset.<something>`, and should be typed from within that clone (where a `meta_dataset` Python module should be visible).

## Downloading and converting datasets

Meta-Dataset uses several established datasets, that are available from different sources.
You can find below a summary of these datasets, as well as instructions to download them and convert them into a common format.

For brevity of the command line examples, we assume the following environment variables are defined:

- `$DATASRC`: root of where the original data is downloaded and potentially extracted from compressed files. This directory does not need to be available after the data conversion is done.
- `$SPLITS`: directory where `*_splits.pkl` files will be created, one per dataset. For instance, `$SPLITS/fungi_splits.pkl` contains information about which classes are part of the meta-training, meta-validation, and meta-test set. This is only used during the dataset conversion phase, but can help troubleshooting later.
- `$RECORDS`: root directory that will contain the converted datasets (one per sub-directory). This directory needs to be available during training and evaluation.

### Dataset summary

Dataset (other names) | Number of classes (train/valid/test) | Size on disk  | Conversion time
----------------------|--------------------------------------|---------------|--------------------------
ilsvrc\_2012 (ImageNet, ILSVRC) \[[instructions](doc/dataset_conversion.md#ilsvrc_2012)\] | 1000 (712/158/130, hierarchical) | \~140 GiB   | 5 to 13 hours
omniglot \[[instructions](doc/dataset_conversion.md#omniglot)\] | 1623 (883/81/659, by alphabet: 25/5/20) | \~60 MiB | few seconds
aircraft (FGVC-Aircraft) \[[instructions](doc/dataset_conversion.md#aircraft)\] | 100 (70/15/15) | \~470 MiB (2.6 GiB download) | 5 to 10 minutes
cu\_birds (Birds, CUB-200-2011) \[[instructions](doc/dataset_conversion.md#cu_birds)\] | 200 (140/30/30) | \~1.1 GiB | \~1 minute
dtd (Describable Textures, DTD) \[[instructions](doc/dataset_conversion.md#dtd)\] | 47 (33/7/7) | \~600 MiB | few seconds
quickdraw (Quick, Draw!) \[[instructions](doc/dataset_conversion.md#quickdraw)\] | 345 (241/52/52) | \~50 GiB | 3 to 4 hours
fungi (FGVCx Fungi) \[[instructions](doc/dataset_conversion.md#fungi)\] | 1394 (994/200/200) | \~13 GiB | 5 to 15 minutes
vgg\_flower (VGG Flower) \[[instructions](doc/dataset_conversion.md#vgg_flower)\] | 102 (71/15/16) | \~330 MiB | \~1 minute
traffic\_sign (Traffic Signs, German Traffic Sign Recognition Benchmark, GTSRB) \[[instructions](doc/dataset_conversion.md#traffic_sign)\] | 43 (0/0/43, test only) | \~50 MiB (263 MiB download) | \~1 minute
mscoco (Common Objects in Context, COCO) \[[instructions](doc/dataset_conversion.md#mscoco)\] | 80 (0/40/40, validation and test only) | \~5.3 GiB (18 GiB download) | 4 hours
*Total (All datasets)* | *4934 (3144/598/1192)* | *\~210 GiB* | *12 to 24 hours*


## Training

Experiments are defined via [gin](google/gin-config) configuration files, that are under `meta_dataset/learn/gin/`:

- `setups/` contain generic setups for classes of experiment, for instance which datasets to use (`imagenet` or `all`), parameters for sampling the number of ways and shots of episodes.
- `models/` define settings for different meta-learning algorithms (baselines, prototypical networks, MAML...)
- `default/` contains files that each correspond to one experiment, mostly defining a setup and a model, with default values for training hyperparameters.
- `best/` contains files with values for training hyperparameters that achieved the best performance during hyperparameter search.

There are two main architectures, or "backbones": `four_layer_convnet` (sometimes `convnet` for short) and `resnet`, that can be used in the baselines ("k-NN" and "Finetune"), ProtoNet, and MatchingNet. Their layers do not have a trainable bias since it would be negated by the use of batch normalization. For fo-MAML and ProtoMAML, each of the backbones have a version with trainable biases (due to the way batch normalization is handled), resp. `four_layer_convnet_maml` (or `mamlconvnet`) and `resnet_maml` (sometimes `mamlresnet`); these can also be used by the baseline for pre-training of the MAML models.

### Reproducing results

See [Reproducing best results](doc/reproducing_best_results.md) for instructions to launch training experiments with the values of hyperparameters that were selected in the paper. The hyperparameters (including the backbone, whether to train from scratch or from pre-trained weights, and the number of training updates) were selected using only the validation classes of the ILSVRC 2012 dataset for all experiments. Even when training on "all" datasets, the validation classes of the other datasets were not used.

We tried our best to reproduce the original results using the public code on Google Cloud VMs, but there is inherent noise and variability in the computation.
For transparency, we will include the results of the reproduced experiments as well as the original ones, which were reported in the article. We also include the validation errors on ILSVRC.

TODO: fill.

#### Models trained on ILSVRC-2012 only

Evaluation Dataset      | k-NN         | Finetune     | MatchingNet  | ProtoNet     | fo-MAML      | Proto-MAML
------------------------|--------------|--------------|--------------|--------------|--------------|------------
 ILSVRC valid           |  29.54       |  37.16       |  35.68       |  38.70       |  24.91       |  42.62
*ILSVRC valid (repro)*  | *29.23*      | *37.51*      | *35.15*      | *37.13*      | *26.14*      | *42.89* 
 ILSVRC test            |  38.16±1.01  |  47.47±1.10  |  43.89±1.05  |  43.43±1.07  |  29.22±1.00  |  50.23±1.13
*ILSVRC test (repro)*   | *37.54±1.00* | *46.83±1.01* | *42.34±1.09* | *43.14±1.03* | *30.36±1.05* | *50.41±1.06* 
 Omniglot               |  59.40±1.31  |  62.97±1.39  |  62.44±1.25  |  60.41±1.35  |  45.42±1.61  |  60.65±1.40
*Omniglot (repro)*      | *58.73±1.22* | *60.88±1.39* | *59.30±1.28* | *56.39±1.40* | *49.06±1.47* | *60.48±1.37* 
 Aircraft               |  44.41±0.92  |  56.35±1.03  |  50.64±0.95  |  48.60±0.88  |  33.81±0.91  |  54.53±0.95
*Aircraft (repro)*      | *45.22±0.88* | *55.94±1.04* | *48.85±0.96* | *46.32±0.87* | *40.81±0.84* | *53.90±0.96* 
 Birds                  |  45.75±0.98  |  61.63±1.03  |  56.36±1.03  |  63.73±1.00  |  39.04±1.17  |  69.71±1.04
*Birds (repro)*         | *46.20±0.95* | *61.10±1.06* | *58.35±1.05* | *63.50±0.95* | *44.36±0.84* | *68.16±0.98* 
 Textures               |  61.53±0.75  |  67.82±0.86  |  65.55±0.76  |  62.17±0.77  |  50.60±0.74  |  66.68±0.80
*Textures (repro)*      | *61.87±0.78* | *68.29±0.77* | *64.77±0.87* | *63.15±0.75* | *52.18±0.82* | *65.68±0.79* 
 Quick Draw             |  46.42±1.10  |  50.89±1.15  |  50.24±1.12  |  50.53±0.97  |  24.33±1.39  |  49.03±1.12
*Quick Draw (repro)*    | *51.42±1.05* | *58.11±1.02* | *54.35±1.02* | *53.62±0.98* | *36.15±1.33* | *56.57±1.02* 
 Fungi                  |  29.91±0.93  |  33.01±1.06  |  33.66±1.00  |  35.95±1.09  |  16.36±0.86  |  39.04±1.03
*Fungi (repro)*         | *29.88±1.00* | *32.81±1.00* | *32.65±1.00* | *36.02±1.05* | *19.43±0.93* | *39.66±1.12* 
 VGG Flower             |  77.23±0.74  |  82.30±0.85  |  80.21±0.74  |  79.47±0.81  |  56.01±1.22  |  85.78±0.80
*VGG Flower (repro)*    | *76.47±0.75* | *83.26±0.80* | *79.57±0.75* | *75.93±0.77* | *63.00±1.02* | *85.75±0.72* 
 Traffic signs          |  58.42±1.28  |  55.67±1.19  |  59.64±1.20  |  46.93±1.11  |  23.53±1.17  |  47.83±1.03
*Traffic signs (repro)* | *57.68±1.24* | *57.13±1.20* | *58.65±1.19* | *44.49±1.10* | *26.08±1.12* | *49.17±1.10* 
 MSCOCO                 |  31.46±1.00  |  33.77±1.37  |  29.83±1.15  |  35.24±1.11  |  13.47±1.04  |  38.06±1.17
*MSCOCO (repro)*        | *38.50±1.01* | *43.44±1.12* | *40.10±1.01* | *40.96±1.04* | *24.69±1.09* | *44.43±1.12* 

#### Models trained on all datasets

Evaluation Dataset      | k-NN         | Finetune     | MatchingNet  | ProtoNet     | fo-MAML      | Proto-MAML
------------------------|--------------|--------------|--------------|--------------|--------------|------------
 ILSVRC valid           |  25.26       |  32.43       |  34.70       |  37.22       |  21.12       |  39.67 
*ILSVRC valid (repro)*  | *23.94*      | *27.98*      | *34.06*      | *36.34*      | *19.73*      | *39.86* 
 ILSVRC test            |  28.46±0.83  |  39.68±1.02  |  40.81±1.02  |  41.82±1.06  |  22.41±0.80  |  45.48±1.02
*ILSVRC test (repro)*   | *±* | *±* | *±* | *±* | *±* | *±* 
 Omniglot               |  88.42±0.63  |  85.57±0.89  |  75.62±1.09  |  78.61±1.10  |  68.14±1.35  |  86.26±0.85
*Omniglot (repro)*      | *±* | *±* | *±* | *±* | *±* | *±* 
 Aircraft               |  70.10±0.73  |  69.81±0.93  |  60.68±0.87  |  66.57±0.92  |  44.48±0.91  |  79.15±0.67
*Aircraft (repro)*      | *±* | *±* | *±* | *±* | *±* | *±* 
 Birds                  |  47.34±0.97  |  54.07±1.08  |  57.09±0.95  |  63.57±1.02  |  36.70±1.13  |  72.67±0.96
*Birds (repro)*         | *±* | *±* | *±* | *±* | *±* | *±* 
 Textures               |  56.39±0.74  |  62.66±0.81  |  64.65±0.77  |  66.60±0.80  |  45.79±0.67  |  66.69±0.77
*Textures (repro)*      | *±* | *±* | *±* | *±* | *±* | *±* 
 Quick Draw             |  66.12±0.91  |  73.88±0.81  |  58.86±1.01  |  63.55±0.92  |  41.27±1.46  |  67.83±0.90
*Quick Draw (repro)*    | *±* | *±* | *±* | *±* | *±* | *±* 
 Fungi                  |  38.35±1.08  |  31.85±1.08  |  34.38±1.01  |  37.97±1.07  |  14.21±0.81  |  44.58±1.19
*Fungi (repro)*         | *±* | *±* | *±* | *±* | *±* | *±* 
 VGG Flower             |  73.21±0.75  |  77.55±0.94  |  82.60±0.66  |  84.43±0.69  |  61.10±1.11  |  88.21±0.68
*VGG Flower (repro)*    | *±* | *±* | *±* | *±* | *±* | *±* 
 Traffic signs          |  49.84±1.23  |  53.07±1.13  |  57.90±1.16  |  50.60±1.02  |  24.03±1.08  |  46.38±1.03
*Traffic signs (repro)* | *±* | *±* | *±* | *±* | *±* | *±* 
 MSCOCO                 |  24.29±0.92  |  27.71±1.20  |  30.20±1.13  |  37.58±1.14  |  13.63±0.96  |  35.12±1.20
*MSCOCO (repro)*        | *±* | *±* | *±* | *±* | *±* | *±* 

### Hyperparameter search

TODO: Range / distribution of tried hyperparameters
