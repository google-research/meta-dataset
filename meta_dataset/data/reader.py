# coding=utf-8
# Copyright 2020 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Forming the first part of a tf.data pipeline, reading from a source on disk.

The data output by the Reader consists in episodes or batches (for EpisodeReader
and BatchReader respectively) from one source (one split of a dataset). They
contain strings represented images that have not been decoded yet, and can
contain dummy examples and examples to discard.
See data/pipeline.py for the next stage of the pipeline.
"""
# TODO(lamblinp): Update variable names to be more consistent
# - target, class_idx, label

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import os

from meta_dataset import data
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

# DUMMY_CLASS_ID will be used as the target of examples used for padding only.
DUMMY_CLASS_ID = -1


def _pad(dataset_indices, chunk_size, dummy_dataset_id):
  """Pads `dataset_indices` with dummy values so it has length `chunk_size`.

  Args:
    dataset_indices: list of int, dataset indices.
    chunk_size: int, size to pad to.
    dummy_dataset_id: int, dummy value to pad with.
  """
  pad_size = chunk_size - len(dataset_indices)
  assert pad_size >= 0
  dataset_indices.extend([dummy_dataset_id] * pad_size)


def dataset_id_generator(dataset_spec, split, pool, sampler):
  """Generates a stream of dataset IDs forming a sequence of episodes.

  Each episode is chunked into:

  * a "flush" chunk, which is meant to allow to flush examples, in case we are
    at the end of an epoch for one or more class in the episode (we want to
    avoid accidentally repeating an example due to epoch boundaries), and
  * some number of additional chunks (for example, a "support" chunk and a
    "query" chunk).

  To make sure the input pipeline knows where the episode boundary is within the
  stream (and where the boundary is between chunks in an episode), we enforce
  that each chunk has a fixed size by padding with dummy dataset IDs (of value
  `num_classes`) as needed (in some cases it's possible that no padding is ever
  needed). The size of each chunk is prescribed by the `compute_chunk_sizes`
  method of `sampler`, which also implicitly defines the number of additional
  chunks (i.e. `len(chunk_sizes) - 1`).

  This generator is meant to be used with
  `tf.data.experimental.choose_from_datasets` and assumes that the list of
  tf.data.Dataset objects corresponding to each class in the dataset (there are
  `num_classes` of them, which is determined by inspecting the `dataset_spec`
  argument using the `split` argument) is appended with a "dummy" Dataset (which
  has index `num_classes` in the list) which outputs a constant `(b'',
  DUMMY_CLASS_ID)` tuple).

  Note that a dataset ID is different from the (absolute) class ID: the dataset
  ID refers to the index of the Dataset in the list of Dataset objects, and the
  class ID (or label) refers to the second element of the tuple that the Dataset
  outputs.

  Args:
    dataset_spec: DatasetSpecification, dataset specification.
    split: one of Split.TRAIN, Split.VALID, or Split.TEST.
    pool: A string ('train' or 'test') or None, indicating which example-level
      split to select, if the current dataset has them.
    sampler: EpisodeDescriptionSampler instance.

  Yields:
    i: int, dataset ID.
  """
  chunk_sizes = sampler.compute_chunk_sizes()
  # An episode always starts with a "flush" chunk to allow flushing examples at
  # class epoch boundaries, and contains `len(chunk_sizes) - 1` additional
  # chunks.
  flush_chunk_size, other_chunk_sizes = chunk_sizes[0], chunk_sizes[1:]

  class_set = dataset_spec.get_classes(split)
  num_classes = len(class_set)
  dummy_dataset_id = num_classes

  total_images_per_class = dict(
      (class_idx,
       dataset_spec.get_total_images_per_class(class_set[class_idx], pool))
      for class_idx in range(num_classes))
  cursors = [0] * num_classes

  # Infinite loop over episodes.
  while True:
    flushed_dataset_indices = []
    selected_dataset_indices = [[] for _ in other_chunk_sizes]
    # Sample an episode description. A description is a tuple of
    # `(class_idx, ...)` tuples, where `class_idx` indicates the class to sample
    # from and the remaining `len(chunk_sizes) - 1` elements indicate how many
    # examples to allocate to each chunk.
    episode_description = sampler.sample_episode_description()
    for element in episode_description:
      class_idx, distribution = element[0], element[1:]
      total_requested = sum(distribution)
      if total_requested > total_images_per_class[class_idx]:
        raise ValueError("Requesting more images than what's available for the "
                         'whole class')
      # If the total number of requested examples is greater than the number of
      # examples remaining for the current pass over class `class_idx`, we flush
      # the remaining examples and start a new pass over class `class_idx`.
      # TODO(lamblinp): factor this out into its own tracker class for
      # readability and testability.
      remaining = total_images_per_class[class_idx] - cursors[class_idx]
      if total_requested > remaining:
        flushed_dataset_indices.extend([class_idx] * remaining)
        cursors[class_idx] = 0
      # Elements of `distribution` correspond to how many examples of class
      # `class_idx` to allocate for each chunk (e.g. in a few-shot learning
      # context `distribution = [5, 8]` would allocate 5 examples to the
      # "support" chunk and 8 examples to the "query" chunk). Elements of
      # `selected_dataset_indices` correspond to the list of dataset indices
      # that have so far been requested for each chunk.
      for num_to_allocate, dataset_indices in zip(distribution,
                                                  selected_dataset_indices):
        dataset_indices.extend([class_idx] * num_to_allocate)
      cursors[class_idx] += total_requested

    # An episode sequence is generated in multiple phases, each padded with an
    # agreed-upon number of dummy dataset IDs.

    _pad(flushed_dataset_indices, flush_chunk_size, dummy_dataset_id)
    for dataset_indices, chunk_size in zip(selected_dataset_indices,
                                           other_chunk_sizes):
      _pad(dataset_indices, chunk_size, dummy_dataset_id)

    # Yield dataset IDs one by one.
    # TODO(lamblinp): revisit yielding the whole list at once rather than
    # element by element for performance reasons.
    dataset_indices = itertools.chain(flushed_dataset_indices,
                                      *selected_dataset_indices)
    for i in dataset_indices:
      yield i


class Reader(object):
  """Class reading data from one source and assembling examples.

  Specifically, it holds part of a tf.data pipeline (the source-specific part),
  that reads data from TFRecords and assembles examples from them.
  """

  def __init__(self,
               dataset_spec,
               split,
               shuffle_buffer_size,
               read_buffer_size_bytes,
               num_prefetch,
               num_to_take=-1):
    """Initializes a Reader from a source.

    The source is identified by dataset_spec and split.

    Args:
      dataset_spec: DatasetSpecification, dataset specification.
      split: A learning_spec.Split object identifying the source split.
      shuffle_buffer_size: An integer, the shuffle buffer size for each Dataset
        object. If 0, no shuffling operation will happen.
      read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
      num_prefetch: int, the number of examples to prefetch for each class of
        each dataset. Prefetching occurs just after the class-specific Dataset
        object is constructed. If < 1, no prefetching occurs.
      num_to_take: Optional, an int specifying a number of elements to pick from
        each tfrecord. If specified, the available images of each class will be
        restricted to that int. By default (-1) no restriction is applied and
        all data is used.
    """
    self.dataset_spec = dataset_spec
    self.split = split
    self.shuffle_buffer_size = shuffle_buffer_size
    self.read_buffer_size_bytes = read_buffer_size_bytes
    self.num_prefetch = num_prefetch
    self.num_to_take = num_to_take

    self.base_path = self.dataset_spec.path
    self.class_set = self.dataset_spec.get_classes(self.split)
    self.num_classes = len(self.class_set)

  def construct_class_datasets(self,
                               pool=None,
                               repeat=True,
                               shuffle=True,
                               shuffle_seed=None):
    """Constructs the list of class datasets.

    Args:
      pool: A string (optional) indicating whether to only read examples from a
        given example-level split.
      repeat: Boolean indicating whether each of the class datasets should be
        repeated (to provide an infinite stream) or not.
      shuffle: Boolean indicating whether each of the class datasets should be
        shuffled or not.
      shuffle_seed: Optional, an int containing the seed passed to
        tf.data.Dataset.shuffle.

    Returns:
      class_datasets: list of tf.data.Dataset, one for each class.
    """
    file_pattern = self.dataset_spec.file_pattern
    # We construct one dataset object per class. Each dataset outputs a stream
    # of `(example_string, dataset_id)` tuples.
    class_datasets = []
    for dataset_id in range(self.num_classes):
      class_id = self.class_set[dataset_id]
      if pool:
        if not data.POOL_SUPPORTED:
          raise NotImplementedError(
              'Example-level splits or pools not supported.')
      else:
        if file_pattern.startswith('{}_{}'):
          # TODO(lamblinp): Add support for sharded files if needed.
          raise NotImplementedError('Sharded files are not supported yet. '
                                    'The code expects one dataset per class.')
        elif file_pattern.startswith('{}'):
          filename = os.path.join(self.base_path, file_pattern.format(class_id))
        else:
          raise ValueError('Unsupported file_pattern in DatasetSpec: %s. '
                           'Expected something starting with "{}" or "{}_{}".' %
                           file_pattern)

      example_string_dataset = tf.data.TFRecordDataset(
          filename, buffer_size=self.read_buffer_size_bytes)

      # Create a dataset containing only num_to_take elements from
      # example_string_dataset. By default, takes all elements.
      example_string_dataset = example_string_dataset.take(self.num_to_take)

      if self.num_prefetch > 0:
        example_string_dataset = example_string_dataset.prefetch(
            self.num_prefetch)
      if shuffle:
        # Do not set a buffer size greater than the number of examples in this
        # class, as it can result in unnecessary memory being allocated.
        num_examples = self.dataset_spec.get_total_images_per_class(
            class_id, pool=pool)
        shuffle_buffer_size = min(num_examples, self.shuffle_buffer_size)
        if shuffle_buffer_size > 1:
          example_string_dataset = example_string_dataset.shuffle(
              buffer_size=shuffle_buffer_size,
              seed=shuffle_seed,
              reshuffle_each_iteration=True)
      if repeat:
        example_string_dataset = example_string_dataset.repeat()

      # These are absolute, dataset-specific class IDs (not relative to a given
      # split). It is okay to have class ID collisions across datasets, since we
      # don't sample multi-dataset episodes.
      class_id_dataset = tf.data.Dataset.from_tensors(class_id).repeat()
      dataset = tf.data.Dataset.zip((example_string_dataset, class_id_dataset))
      class_datasets.append(dataset)

    assert len(class_datasets) == self.num_classes
    return class_datasets


class EpisodeReaderMixin(object):
  """Mixin class to assemble examples as episodes."""

  def create_dataset_input_pipeline(self,
                                    sampler,
                                    pool=None,
                                    shuffle_seed=None):
    """Creates a Dataset encapsulating the input pipeline for one data source.

    Args:
      sampler: EpisodeDescriptionSampler instance.
      pool: A string (optional) indicating whether to only read examples from a
        given example-level split.
      shuffle_seed: Optional, an int containing the seed passed to
        tf.data.Dataset.shuffle.

    Returns:
      dataset: a tf.data.Dataset instance which encapsulates episode creation
        for the data identified by `dataset_spec` and `split`. These episodes
        contain flushed examples and are internally padded with dummy examples.
        A later part of the pipeline, shared across all sources, will extract
        support and query sets and decode the example strings.
    """
    # Always shuffle, unless self.shuffle_buffer_size is 0
    shuffle = (self.shuffle_buffer_size and self.shuffle_buffer_size > 0)
    class_datasets = self.construct_class_datasets(
        pool=pool, shuffle=shuffle, shuffle_seed=shuffle_seed)

    # We also construct a dummy dataset which outputs `(b'', DUMMY_CLASS_ID)`
    # tuples.
    dummy_dataset = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensors(b'').repeat(),
         tf.data.Dataset.from_tensors(DUMMY_CLASS_ID).repeat()))
    class_datasets.append(dummy_dataset)

    # The "choice" dataset outputs a stream of dataset IDs which are used to
    # select which class dataset to sample from. We turn the stream of dataset
    # IDs into a stream of `(example_string, class_id)` tuples using
    # `choose_from_datasets`.
    choice_generator = functools.partial(
        dataset_id_generator,
        dataset_spec=self.dataset_spec,
        split=self.split,
        pool=pool,
        sampler=sampler)

    choice_dataset = tf.data.Dataset.from_generator(choice_generator,
                                                    (tf.int64),
                                                    tf.TensorShape([]))
    dataset = tf.data.experimental.choose_from_datasets(class_datasets,
                                                        choice_dataset)

    # Episodes have a fixed size prescribed by `sampler.compute_chunk_sizes`.
    dataset = dataset.batch(sum(sampler.compute_chunk_sizes()))
    # Overlap batching and episode processing.
    dataset = dataset.prefetch(1)

    return dataset


class EpisodeReader(Reader, EpisodeReaderMixin):
  """Subclass of Reader assembling the examples as Episodes."""


def add_offset_to_target(example_strings, targets, offset):
  """Adds offset to the targets.

  This function is intented to be passed to tf.data.Dataset.map.

  Args:
    example_strings: 1-D Tensor of dtype str, Example protocol buffers.
    targets: 1-D Tensor of dtype int, targets representing the absolute class
      IDs.
    offset: int, optional, number to add to class IDs to get targets.

  Returns:
    example_strings, labels: Tensors, a batch of examples and labels.
  """
  labels = targets + offset
  return (example_strings, labels)


class BatchReaderMixin(object):
  """Mixin class to assemble examples as batches."""

  def create_dataset_input_pipeline(self,
                                    batch_size,
                                    offset=0,
                                    pool=None,
                                    shuffle_seed=None):
    """Creates a Dataset encapsulating the input pipeline for one data source.

    Args:
      batch_size: An int representing the max number of examples in each batch.
      offset: An int, that is added to the value of all the targets. This makes
        it possible to have a unique range of targets for each dataset.
      pool: A string (optional) indicating whether to only read examples from a
        given example-level split. If it is provided, these examples will be
        used as 'real test data', and used once each for evaluation only. The
        accepted values are 'valid' and 'test'.
      shuffle_seed: Optional, an int containing the seed passed to
        tf.data.Dataset.shuffle.

    Returns:
      dataset: a tf.data.Dataset instance which encapsulates batch creation for
        the data identified by `dataset_spec` and `split`. These batches contain
        compressed image representations and (possibly offset) absolute class
        IDs. A later part of the pipeline, shared across all sources, will
        decode the example strings.

    Raises:
      ValueError: Invalid pool provided. The supported values are 'valid' and
        'test'.
    """
    if pool and pool not in ['valid', 'test']:
      raise ValueError('Invalid pool provided. The supported values '
                       'are "valid" and "test".')
    # Do not shuffle or repeat each class dataset, to avoid fuzzing epoch
    # boundaries.
    class_datasets = self.construct_class_datasets(
        pool=pool, repeat=False, shuffle=False)
    num_classes = len(class_datasets)

    if pool:
      if not data.POOL_SUPPORTED:
        raise NotImplementedError(
            'Example-level splits or pools not supported.')
    else:
      # To have labels start at 0 and be contiguous, subtracting the starting
      # index from all
      start_ind = self.class_set[0]
      class_set = [
          self.class_set[ds_id] - start_ind for ds_id in range(num_classes)
      ]
      if list(class_set) != list(range(num_classes)):
        raise NotImplementedError('Batch training currently assumes the class '
                                  'set is contiguous and starts at 0.')

      # Sample from each class dataset according to its proportion of examples,
      # so examples from one class should be spread across the whole epoch.
      # Then, shuffle and repeat the combined dataset.
      num_examples_per_class = [
          self.dataset_spec.get_total_images_per_class(class_id, pool=pool)
          for class_id in class_set
      ]
      num_examples_per_class = np.array(num_examples_per_class, 'float64')
      class_proportions = num_examples_per_class / num_examples_per_class.sum()

      dataset = tf.data.experimental.sample_from_datasets(
          class_datasets, weights=class_proportions, seed=shuffle_seed)
      if self.shuffle_buffer_size and self.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=self.shuffle_buffer_size,
            seed=shuffle_seed,
            reshuffle_each_iteration=True)

    # Using drop_remainder=False for two reasons:
    # - Most importantly, during established splits evaluation, we need to
    #   evaluate on all examples.
    # - Also during training, if the shuffle buffer does not hold all the data,
    #   the last examples are more likely to be dropped than the first ones.
    # In any case, we are handling variable-sized batches just fine, so there
    # is no real reason to drop data.
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if not pool:
      dataset = dataset.repeat()

    if offset:
      map_fn = functools.partial(add_offset_to_target, offset=offset)
      dataset = dataset.map(map_fn)

    # Overlap batching and episode processing.
    dataset = dataset.prefetch(1)

    return dataset


class BatchReader(Reader, BatchReaderMixin):
  """Subclass of Reader assembling the examples as Batches."""
