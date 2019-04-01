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

Dataset | Other names | # classes (train/valid/test) | Size on disk  | Expected conversion time
--------|-------------|------------------------------|---------------|--------------------------
[ilsvrc_2012](#ilsvrc_2012) | ImageNet | 1000 () | \~ 140 GiB   | 5 to 13 hours


### ilsvrc_2012

- Download `ilsvrc2012_img_train.tar`, from the [ILSVRC2012 website](http://www.image-net.org/challenges/LSVRC/2012/index)
- Uncompress it into `ILSVRC2012_img_train/`, which should contain 1000 files, named `n????????.tar` (expected time: \~30 minutes)
- Uncompress each of `ILSVRC2012_img_train/n????????.tar` in its own directory (expected time: \~30 minutes), for instance:
  ```bash
  for FILE in *.tar; do mkdir ${FILE/.tar/}; cd ${FILE/.tar/}; tar xvf ../$FILE; cd ..; done
  ```
- Download the following two files into `ILSVRC2012_img_train/`:
  - http://www.image-net.org/archive/wordnet.is_a.txt
  - http://www.image-net.org/archive/words.txt
- The conversion itself should take 4 to 12 hours, depending on the filesystem's latency and bandwidth:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=ilsvrc_2012 \
    --imagenet_data_root=$DATASRC/ILSVRC2012_img_train \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/ilsvrc_2012/`:
  - 1000 tfrecords files named `[0-999].tfrecords`
  - `dataset_spec.pkl`
  - `num_leaf_images.pkl`

## Launching experiments
