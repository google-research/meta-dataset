This repository contains accompanying code for the article introducing
Meta-Dataset,
[arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

This code is provided here in order to give more details on the implementation
of the data-providing pipeline, our back-bones and models, as well as the
experimental setting.

See below for [user instructions](#user-instructions), including how to:

1.  [install](#installation) the software,
2.  [download and convert](#downloading-and-converting-datasets) the data, and
3.  [train](#training) implemented models.

See this [introduction notebook](https://github.com/google-research/meta-dataset/blob/master/Intro_to_Metadataset.ipynb) for a demonstration of how to sample data from the pipeline (episodes or batches).

In order to run the experiments described in the first version of the arXiv
article, [arxiv.org/abs/1903.03096v1](https://arxiv.org/abs/1903.03096v1),
please use the instructions, code, and configuration files at version
[arxiv_v1](https://github.com/google-research/meta-dataset/tree/arxiv_v1) of
this repository.

We are currently working on updating the instructions, code, and configuration
files to reproduce the results in the second version of the article,
[arxiv.org/abs/1903.03096v2](https://arxiv.org/abs/1903.03096v2).
You can follow the progess in branch
[arxiv_v2_dev](https://github.com/google-research/meta-dataset/tree/arxiv_v2_dev)
of this repository.

This is not an officially supported Google product.

## Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples

_Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci,
Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol,
Hugo Larochelle_

Few-shot classification refers to learning a classifier for new classes given
only a few examples. While a plethora of models have emerged to tackle it, we
find the procedure and datasets that are used to assess their progress lacking.
To address this limitation, we propose Meta-Dataset: a new benchmark for
training and evaluating models that is large-scale, consists of diverse
datasets, and presents more realistic tasks. We experiment with popular
baselines and meta-learners on Meta-Dataset, along with a competitive method
that we propose. We analyze performance as a function of various characteristics
of test tasks and examine the models' ability to leverage diverse training
sources for improving their generalization. We also propose a new set of
baselines for quantifying the benefit of meta-learning in Meta-Dataset. Our
extensive experimentation has uncovered important research challenges and we
hope to inspire work in these directions.


# User instructions
## Installation
Meta-Dataset is now compatible with Python 2 and Python 3, please report any glitch with Python 3.
The code has not been tested with TensorFlow 2 yet.

- We recommend you follow [these instructions](https://www.tensorflow.org/install/pip) to install TensorFlow.
- A list of packages to install is available in `requirements.txt`, you can install them using `pip`.
- Clone the `meta-dataset` repository. Most command lines start with `python -m meta_dataset.<something>`, and should be typed from within that clone (where a `meta_dataset` Python module should be visible).

## Downloading and converting datasets

Meta-Dataset uses several established datasets, that are available from different sources.
You can find below a summary of these datasets, as well as instructions to download them and convert them into a common format.

For brevity of the command line examples, we assume the following environment variables are defined:

- `$DATASRC`: root of where the original data is downloaded and potentially extracted from compressed files. This directory does not need to be available after the data conversion is done.
- `$SPLITS`: directory where `*_splits.json` files will be created, one per dataset. For instance, `$SPLITS/fungi_splits.json` contains information about which classes are part of the meta-training, meta-validation, and meta-test set.
  These files are only used during the dataset conversion phase, but can help troubleshooting later.
  To re-use the [canonical splits](https://github.com/google-research/meta-dataset/tree/master/meta_dataset/dataset_conversion/splits) instead of re-generating them, you can make it point to `meta_dataset/dataset_conversion` in your checkout.
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

There are three main architectures, also called "backbones" (or "embedding networks"): `four_layer_convnet` (sometimes `convnet` for short), `resnet`, and `wide_resnet`. These architectures can be used by all baselines and episodic models.
Another backbone, `relationnet_embedding` (similar to `four_layer_convnet` but without pooling on the last layer), is only used by RelationNet (and baseline, for pre-training purposes).

### Reproducing results

See [Reproducing best results](doc/reproducing_best_results.md) for instructions to launch training experiments with the values of hyperparameters that were selected in the paper. The hyperparameters (including the backbone, whether to train from scratch or from pre-trained weights, and the number of training updates) were selected using only the validation classes of the ILSVRC 2012 dataset for all experiments. Even when training on "all" datasets, the validation classes of the other datasets were not used.

We will try our best to reproduce the original results using the public code on Google Cloud VMs, and we will include the results of the reproduced experiments in the future.

#### Models trained on ILSVRC-2012 only

Evaluation Dataset      | k-NN         | Finetune     | MatchingNet  | ProtoNet     | fo-MAML      | RelationNet  | fo-Proto-MAML
------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------
 ILSVRC (test)          |  41.03±1.01  |  45.78±1.10  |  45.00±1.10  |  50.50±1.08  |  36.09±1.01  |  34.69±1.01  |**51.01**±1.05
 Omniglot               |  37.07±1.15  |  60.85±1.58  |  52.27±1.28  |  59.98±1.35  |  38.67±1.39  |  45.35±1.36  |**62.00**±1.35
 Aircraft               |  46.81±0.89  |**68.69**±1.26|  48.97±0.93  |  53.10±1.00  |  34.50±0.90  |  40.73±0.83  |  55.31±0.96
 Birds                  |  50.13±1.00  |  57.31±1.26  |  62.21±0.95  |**68.79**±1.01|  49.10±1.18  |  49.51±1.05  |  66.87±1.04
 Textures               |  66.36±0.75  |**69.05**±0.90|  64.15±0.85  |  66.56±0.83  |  56.50±0.80  |  52.97±0.69  |  67.75±0.78
 Quick Draw             |  32.06±1.08  |  42.60±1.17  |  42.87±1.09  |  48.96±1.08  |  27.24±1.24  |  43.30±1.08  |**53.70**±1.06
 Fungi                  |  36.16±1.02  |  38.20±1.02  |  33.97±1.00  |**39.71**±1.11|  23.50±1.00  |  30.55±1.04  |  37.97±1.11
 VGG Flower             |  83.10±0.68  |  85.51±0.68  |  80.13±0.71  |  85.27±0.77  |  66.42±0.96  |  68.76±0.83  |**86.86**±0.75
 Traffic signs          |  44.59±1.19  |**66.79**±1.31|  47.80±1.14  |  47.12±1.10  |  33.23±1.34  |  33.67±1.05  |  51.19±1.11
 MSCOCO                 |  30.38±0.99  |  34.86±0.97  |  34.99±1.00  |  41.00±1.10  |  27.52±1.11  |  29.15±1.01  |**43.41**±1.06
**Average rank**        |   5.0        |   2.5        |   4.0        |   2.4        |   6.7        |   5.8        | **1.6**

#### Models trained on all datasets

Evaluation Dataset      | k-NN         | Finetune     | MatchingNet  | ProtoNet     | fo-MAML      | RelationNet  | fo-Proto-MAML
------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------
 ILSVRC (test)          |  38.55±0.94  |  43.08±1.08  |  36.08±1.00  |  44.50±1.05  |  32.36±1.02  |  30.89±0.93  |**47.85**±1.08
 Omniglot               |  74.60±1.08  |  71.11±1.37  |  78.25±1.01  |  79.56±1.12  |  71.91±1.20  |**86.57**±0.79|  82.86±0.94
 Aircraft               |  64.98±0.82  |  72.03±1.07  |  69.17±0.96  |  71.14±0.86  |  52.76±0.90  |  69.71±0.83  |**74.24**±0.77
 Birds                  |  66.35±0.92  |  59.82±1.15  |  56.40±1.00  |  67.01±1.02  |  47.24±1.14  |  54.14±0.99  |**69.97**±0.95
 Textures               |  63.58±0.79  |**69.14**±0.85|  61.80±0.74  |  65.18±0.84  |  56.66±0.74  |  56.56±0.73  |  67.94±0.82
 Quick Draw             |  44.88±1.05  |  47.05±1.16  |  60.81±1.03  |  64.88±0.89  |  50.50±1.19  |  61.75±0.97  |**66.57**±0.90
 Fungi                  |  37.12±1.06  |  38.16±1.04  |  33.70±1.04  |  40.26±1.13  |  21.02±0.99  |  32.56±1.08  |**41.99**±1.12
 VGG Flower             |  83.47±0.61  |  85.28±0.69  |  81.90±0.72  |  86.85±0.71  |  70.93±0.99  |  76.08±0.76  |**88.45**±0.67
 Traffic signs          |  40.11±1.10  |**66.74**±1.23|  55.57±1.08  |  46.48±1.00  |  34.18±1.26  |  37.48±0.93  |  52.32±1.08
 MSCOCO                 |  29.55±0.96  |  35.17±1.08  |  28.79±0.96  |  39.87±1.06  |  24.05±1.10  |  27.41±0.89  |**41.29**±1.03
**Average rank**        |   4.6        |   3.3        |   4.5        |   2.5        |   6.5        |   5.2        | **1.4**

### Hyperparameter search

TODO: Range / distribution of tried hyperparameters
