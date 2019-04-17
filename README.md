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

Evaluation Dataset      | k-NN | Finetune | MatchingNet | ProtoNet | fo-MAML | Proto-MAML
------------------------|------|----------|-------------|----------|---------|------------
 ILSVRC valid           |  |  |  |  |  |  
*ILSVRC valid (repro)*  | ** | ** | ** | ** | ** | ** 
 ILSVRC test            | 38.16±1.01 | 47.47±1.10 | 43.89±1.05 | 43.43±1.07 | 29.22±1.00 | 50.23±1.13
*ILSVRC test (repro)*   | ** | ** | ** | ** | ** | ** 
 Omniglot               |  |  |  |  |  |  
*Omniglot (repro)*      | ** | ** | ** | ** | ** | ** 
 Aircraft               |  |  |  |  |  |  
*Aircraft (repro)*      | ** | ** | ** | ** | ** | ** 
 Birds                  |  |  |  |  |  |  
*Birds (repro)*         | ** | ** | ** | ** | ** | ** 
 Textures               |  |  |  |  |  |  
*Textures (repro)*      | ** | ** | ** | ** | ** | ** 
 Quick Draw             |  |  |  |  |  |  
*Quick Draw (repro)*    | ** | ** | ** | ** | ** | ** 
 Fungi                  |  |  |  |  |  |  
*Fungi (repro)*         | ** | ** | ** | ** | ** | ** 
 VGG Flower             |  |  |  |  |  |  
*VGG Flower (repro)*    | ** | ** | ** | ** | ** | ** 
 Traffic signs          |  |  |  |  |  |  
*Traffic signs (repro)* | ** | ** | ** | ** | ** | ** 
 MSCOCO                 |  |  |  |  |  |  
*MSCOCO (repro)*        | ** | ** | ** | ** | ** | ** 

#### Models trained on all datasets

Evaluation Dataset      | k-NN | Finetune | MatchingNet | ProtoNet | fo-MAML | Proto-MAML
------------------------|------|----------|-------------|----------|---------|------------
 ILSVRC valid           |  |  |  |  |  |  
*ILSVRC valid (repro)*  | ** | ** | ** | ** | ** | ** 
 ILSVRC test            | 28.46±0.83 | 39.68±1.02 | 40.81±1.02 | 41.82±1.06 | 22.41±0.80 | 45.48±1.02
*ILSVRC test (repro)*   | ** | ** | ** | ** | ** | ** 
 Omniglot               |  |  |  |  |  |  
*Omniglot (repro)*      | ** | ** | ** | ** | ** | ** 
 Aircraft               |  |  |  |  |  |  
*Aircraft (repro)*      | ** | ** | ** | ** | ** | ** 
 Birds                  |  |  |  |  |  |  
*Birds (repro)*         | ** | ** | ** | ** | ** | ** 
 Textures               |  |  |  |  |  |  
*Textures (repro)*      | ** | ** | ** | ** | ** | ** 
 Quick Draw             |  |  |  |  |  |  
*Quick Draw (repro)*    | ** | ** | ** | ** | ** | ** 
 Fungi                  |  |  |  |  |  |  
*Fungi (repro)*         | ** | ** | ** | ** | ** | ** 
 VGG Flower             |  |  |  |  |  |  
*VGG Flower (repro)*    | ** | ** | ** | ** | ** | ** 
 Traffic signs          |  |  |  |  |  |  
*Traffic signs (repro)* | ** | ** | ** | ** | ** | ** 
 MSCOCO                 |  |  |  |  |  |  
*MSCOCO (repro)*        | ** | ** | ** | ** | ** | ** 

### Hyperparameter search

TODO: Range / distribution of tried hyperparameters
