# Meta-Dataset in TFDS

This directory contains a Tensorflow Datasets implementation of Meta-Dataset. It
is meant to mirror the original Meta-Dataset implementation exactly (see
`meta_dataset_test.py`) and reuses code when it can.

**Warning: There are some implementation details that the user should keep in
mind in order to avoid running into unexpected behavior; see
[Important caveats](#imporant-caveats).**

## Automated data downloading and conversion

One major pain point that the TFDS implementation aims to solve is that of data
preparation. TFDS automates the vast majority of that work. The only manual
intervention required is to download the ILSVRC 2012 training data
(`ILSVRC2012_img_train.tar`) into TFDS's manual download directory (e.g.
`~/tensorflow_datasets/downloads/manual/`).

**Note: A registration at http://www.image-net.org/download-images is required in order
to get the link to download the dataset.**

First, make sure that `meta_dataset` and its dependencies are installed. This
can be done with

```bash
pip install <PATH_TO_META_DATASET_REPO>
```

Generating the tfrecord files associated with all data sources and storing
them in `~/tensorflow_datasets/meta_dataset` is done with a single command run
from the `<PATH_TO_META_DATASET_REPO>/meta_dataset/data/tfds` directory:

```bash
tfds build md_tfds --manual_dir=<MANUAL_DIR>
```

where `<MANUAL_DIR>` is the directory where the `ILSVRC2012_img_train.tar` file
was downloaded.

**Note: downloading and converting all the data takes a few hours.**

## Use cases: I want to...

### Evaluate on Meta-Dataset

```python
import gin
import meta_dataset
from meta_dataset.data.tfds import api
import tensorflow_datasets as tfds

# Set up a TFDS-compatible data configuration. The `data_config_tfds.gin` config
# sets up the episode pipeline according to the Meta-Dataset episode sampling
# protocol and should be left as-is for standard evaluation, but can be modified
# or replaced with another config file for non-standard evaluation.
gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                      'learn/gin/setups/data_config_tfds.gin')

for source_name in ('aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
                    'mscoco', 'omniglot', 'quickdraw', 'traffic_sign',
                    'vgg_flower'):
  episode_dataset = api.episode_dataset(
      tfds.builder('meta_dataset', config=source_name),
      # 'v1' here refers to the Meta-Dataset protocol version and means that we
      # are using the protocol defined in the original Meta-Dataset paper
      # (rather than in the VTAB+MD paper, which is the 'v2' protocol; see the
      # VTAB+MD paper for a detailed explanation). This is not to be confused
      # with the (unrelated) arXiv version of the Meta-Dataset paper.
      'v1',
      # This is where the meta-split ('train', 'valid', or 'test') is specified.
      'test')
  # Evaluation is usually performed on 600 episodes, but we use 4 here for
  # demonstration.
  for episode in episode_dataset.take(4).as_numpy_iterator():
    support_images, support_labels, _ = episode[:3]
    query_images, query_labels, _ = episode[3:]
    # Evaluate on the episode...
```

Note that there is an I/O overhead to instantiating the episode dataset,
especially for sources such as Omniglot which feature a large number of test
classes. It's possible to amortize that overhead across multiple evaluation runs
by creating an iterator from `episode_dataset` directly and calling `next` on
it:

```python
# ...
episode_iterator = episode_dataset.as_numpy_iterator()

for model in models_to_evaluate:
  for _ in range(600):
    episode = next(episode_iterator)
    # Evaluate on the episode...
```

### Evaluate on VTAB+MD

MD-v2 episodes are obtained by passing `'v2'` to `api.episode_dataset` instead
of `'v1'`. As explained in the [VTAB+MD paper](https://openreview.net/pdf?id=Q0hm0_G1mpH),
the MD-v2 evaluation protocol

1. removes `'vgg_flower'` as a data source and reserves it for VTAB-v2, and
2. assigns all `'ilsvrc_2012'` classes to the training split of classes.

```python
import gin
import meta_dataset
from meta_dataset.data.tfds import api
import tensorflow_datasets as tfds

# Set up a TFDS-compatible data configuration.
gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                      'learn/gin/setups/data_config_tfds.gin')

# In MD-v2, all classes in 'ilsvrc_2012' are assigned to the training split of
# classes, and 'vgg_flower' is not available, so we remove both from the list of
# sources to evaluate on.
for source_name in ('aircraft', 'cu_birds', 'dtd', 'fungi', 'mscoco',
                    'omniglot', 'quickdraw', 'traffic_sign'):
  episode_dataset = api.episode_dataset(
      tfds.builder('meta_dataset', config=source_name),
      # 'v2' again the Meta-Dataset protocol version. We are using the protocol
      # defined in the VTAB+MD paper.
      'v2',
      # This is where the meta-split ('train', 'valid', or 'test') is specified.
      'test')
  # Evaluation is usually performed on 600 episodes, but we use 4 here for
  # demonstration.
  for episode in episode_dataset.take(4).as_numpy_iterator():
    support_images, support_labels, _ = episode[:3]
    query_images, query_labels, _ = episode[3:]
    # Evaluate on the episode...
```

Some VTAB-v2 tasks have test sets that are too large to be used as a monolithic
query set for few-shot learners that expect learning episodes. Assuming the
few-shot learner is non-transductive, one approach around that is to create
episodes with the task's 1000 training examples as the support set and a subset
of the test set as the query set. By partitioning the test set into multiple
query sets which all share the same support set and aggregating query accuracies
across query sets, we can obtain a query accuracy measurement that is equivalent
to evaluating on the entire test set.

Here we assume that the [`task_adaptation`](https://github.com/google-research/task_adaptation)
repository was cloned locally and installed with

```bash
pip install <PATH_TO_TASK_ADAPTATION_REPO>
```

We can use the `meta_dataset.data.read_episodes.read_vtab_as_episode` function
provided by `meta_dataset` to perform the "episodification" of VTAB-v2 tasks.
As explained in the [VTAB+MD paper](https://openreview.net/pdf?id=Q0hm0_G1mpH),
the VTAB-v2 evaluation protocol removes `'dtd'` from the tasks, as it is
reserved for MD-v2 training and evaluation.


```python
from meta_dataset.data import read_episodes

# In VTAB+MD, 'dtd' task is used by MD-v2, so we remove it from the list of VTAB
# tasks.
task_names = [task_name for task_name in read_episodes.VTAB_DATASETS
              if task_name != 'dtd']
for task_name in task_names:
  (support_dataset, query_dataset,
   num_query, num_classes) = read_episodes.read_vtab_as_episode(
      task_name,
      # Images are resized to `image_size` and scaled between -1.0 and 1.0.
      image_size=126,
      # This is the maximum query size allowed. The task's test set will be
      # partitioned into query sets of size `query_size_limit`, with a potential
      # remainder query set if the test set's size is not divisible by
      # `query_size_limit`.
      query_size_limit=500)

  support_set = next(support_dataset.as_numpy_iterator())
  support_images = support_set['image']
  support_labels = support_set['label']

  for query_set in query_dataset.as_numpy_iterator():
    query_images = query_set['image']
    query_labels = query_set['label']
    # Evaluate on the episode...```
```

**Note: Tensorflow Datasets needs to download and prepare the datasets used by
VTAB. This is done on the very first time VTAB tasks are instantiated (unless
the underlying datasets have already been downloaded and prepared).**


### Train on Meta-Dataset episodes

```python
import gin
import meta_dataset
from meta_dataset.data.tfds import api
import tensorflow_datasets as tfds

# Set up a TFDS-compatible data configuration.
gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                      'learn/gin/setups/data_config_tfds.gin')

# 'v1' here refers to the Meta-Dataset protocol version and means that we
# are using the protocol defined in the original Meta-Dataset paper
# (rather than in the VTAB+MD paper, which is the 'v2' protocol; see the
# VTAB+MD paper for a detailed explanation). This is not to be confused
# with the (unrelated) arXiv version of the Meta-Dataset paper.
md_version = 'v1'
md_sources = ('aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot',
              'quickdraw'):
if md_version == 'v1':
  md_sources += ('vgg_flower',)

episode_dataset = api.meta_dataset(
    md_sources,
    md_version,
    # This is where the meta-split ('train', 'valid', or 'test') is specified.
    'train')

# We sample 4 episodes here for demonstration. `source_id` is the index (in
# `md_sources`) of the source that was sampled for the episode.
for episode, source_id in episode_dataset.take(4).as_numpy_iterator():
  support_images, support_labels, _ = episode[:3]
  query_images, query_labels, _ = episode[3:]
  # Train step...
```


### Train on full-ways data sampled from Meta-Dataset sources ("batch training")

```python
from meta_dataset.data.tfds import api
import tensorflow as tf
import tensorflow_datasets as tfds

md_source = 'aircraft'
md_version = 'v1'
meta_split = 'train'

# Due to how data is stored in Meta-Dataset's TFDS implementation, we need to
# be careful to choose a large-enough shuffle buffer size (see "Important
# caveats" for details). Here we use the lesser of 10,000 and the meta-train
# split's total number of examples.
metadata = tfds.builder(f'meta_dataset/{md_source}').info.metadata
start, stop = metadata['meta_split_slices'][md_version][meta_split]
total_images = stop - start
shuffle_buffer_size = min(10_000, total_images)

dataset = api.full_ways_dataset(
    md_source=md_source,
    md_version=md_version,
    meta_split=meta_split,
).shuffle(
    shuffle_buffer_size
# Here we use an image processing that is consistent with
# `meta_dataset.data.decoder.ImageDecoder`, we could use any
# processing+augmentation function that produces images of a fixed size before
# batching.
).map(
    lambda data: {
        'image': 2 * (tf.cast(
            tf.compat.v1.image.resize_images(
                data['image'],
                [126, 126],
                method=tf.compat.v1.image.ResizeMethod.BILINEAR,
                align_corners=True),
            tf.float32) / 255.0 - 0.5),
        'label': data['label']
    }
).batch(256, drop_remainder=False).prefetch(1)

for batch in dataset.as_numpy_iterator():
  images = batch['image']
  labels = batch['label']
  # Train step...
```

# Important caveats

Users need to be aware of the implementation strategy that was used, otherwise
they risk running into counterintuitive behavior.

## Data structure

The core of the Meta-Dataset input pipeline consists in a set of class-specific
datasets from which examples are drawn according to randomly-sampled episode
specifications. The official implementation makes this possible by storing
examples of each class in their own tfrecord file. Rather than partitioning the
examples into training, validation, and test sets, Meta-Dataset partitions the
*classes* themselves into training, validation, and test sets of classes.

TFDS offers a slicing API that allows users to read a contiguous slice of
examples in the dataset while ignoring the rest. This is used for instance to
partition MNIST's official training set into training and validation sets (there
is no canonical validation set for MNIST). We use this mechanism to our
advantage: if the examples are sorted by class in the tfrecords shards, and if
we know the boundary indices between classes in the dataset, then we can very
easily slice through the dataset to build a class-specific dataset.

**The TFDS implementation of Meta-Dataset therefore stores examples in a very
structured way**: the examples are sorted by class (to allow building
class-specific datasets through slicing), and the classes themselves are sorted
by split (to allow building a split-specific full-ways dataset through slicing).

This strategy offers great benefits when sampling episodes, but it does have
drawbacks in the full-ways setting that users need to be aware of:

-   Since examples are sorted by class, naively iterating over the full-ways
    dataset for a given split yields examples that are extremely
    class-correlated. We mitigate this by setting the interleave cycle length to
    the number of shards in the split and the block cycle length to a low value
    (which results in `tf.data.Dataset` reading a few examples at a time from
    all shards while iterating). This means that:
    -   **Users should be careful when passing a non-default `reader_config`
        kwarg.**
    -   **Memory usage is greater than with more conventional TFDS datasets.**
    -   **Users should be mindful of the shuffle buffer side they choose and
        pick larger values than usual.**
-   Slicing through the dataset behaves very differently than with more
    conventional TFDS datasets. Again, since examples are sorted by class,
    slicing through the last 20% of examples will include all examples for some
    classes and exclude all examples for some other classes. This is the
    unfortunate price to pay in order to be able to easily construct
    class-specific datasets. See
    `meta_dataset_test.MetaDatasetTest.test_slice_split_workaround` for an
    example of a workaround.

## Splits

The implementation provides a single `'all_classes'` split for each data source.
Since TFDS splits are expected to share the same label space, partitioning the
training, validation, and test sets of classes into their own TFDS split is not
possible. Even if it were possible, we would still avoid it in order not to
overload the semantics of TFDS' split mechanism (which is meant to partition
examples sharing the same label space rather than the label space itself).
