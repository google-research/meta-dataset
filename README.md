This repository contains accompanying code for the article introducing
Meta-Dataset,
[https://arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

This code is provided here in order to give more details on the implementation
of the data-providing pipeline, our back-bones and models, as well as the
experimental setting.

See below for [user instructions](#user-instructions), including how to [install](#installation) the software, [download and convert](#downloading-and-converting-datasets) the data, and [train](#launching-experiments) implemented models.

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
- `$SPLITS`: directory where `splits_*.pkl` files will be created, one per dataset. For instance, `$SPLITS/fungi_splits.pkl` contains information about which classes are part of the meta-training, meta-validation, and meta-test set. This is only used during the dataset conversion phase, but can help troubleshooting later.
- `$RECORDS`: root directory that will contain the converted datasets (one per sub-directory). This directory needs to be available during training and evaluation.

### Dataset summary

Dataset (other names) | Number of classes (train/valid/test) | Size on disk  | Expected conversion time
----------------------|--------------------------------------|---------------|--------------------------
ilsvrc\_2012 (ImageNet, ILSVRC) \[[instructions](doc/dataset_conversion.md#ilsvrc_2012)\] | 1000 (712/202/188, hierarchical) | \~140 GiB   | 5 to 13 hours
omniglot \[[instructions](doc/dataset_conversion.md#omniglot)]\] | 1623 (883/81/659, by alphabet: 25/5/20) | \~60 MiB | few seconds
aircraft (FGVC-Aircraft) \[[instructions](doc/dataset_conversion.md#aircraft)]\] | 100 (70/15/15) | \~470 MiB (2.6 GiB download) | 5 to 10 minutes
cu\_birds (Birds, CUB-200-2011) \[[instructions](doc/dataset_conversion.md#cu_birds)]\] | 200 (140/30/30) | \~1.1 GiB | \~1 minute
dtd (Describable Textures, DTD) \[[instructions](doc/dataset_conversion.md#dtd)]\] | 47 (33/7/7) | \~600 MiB | few seconds
quickdraw (Quick, Draw!) \[[instructions](doc/dataset_conversion.md#quickdraw)]\] | 345 (241/52/52) | \~50 GiB | 3 to 4 hours
fungi (FGVCx Fungi) \[[instructions](doc/dataset_conversion.md#fungi)]\] | 1394 (994/200/200) | \~13 GiB | 5 to 15 minutes
vgg\_flower (VGG Flower) \[[instructions](doc/dataset_conversion.md#vgg_flower)]\] | 102 (71/15/16) | \~330 MiB | \~1 minute
traffic\_sign (Traffic Signs, German Traffic Sign Recognition Benchmark, GTSRB) \[[instructions](doc/dataset_conversion.md#traffic_sign)]\] | 43 (0/0/43, test only) | \~50 MiB (263 MiB download) | \~1 minute
mscoco (Common Objects in Context, COCO) \[[instructions](doc/dataset_conversion.md#mscoco)]\] | 80 (0/40/40, validation and test only) | (18 GiB download) | \~5.3 GiB | 4 hours


## Launching experiments
