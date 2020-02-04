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
"""Tests for Readers and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import gin.tf
from meta_dataset.data import config
from meta_dataset.data import reader
from meta_dataset.data import sampling
from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

# DatasetSpecification to use in tests
DATASET_SPEC = DatasetSpecification(
    name=None,
    classes_per_split={
        Split.TRAIN: 15,
        Split.VALID: 5,
        Split.TEST: 10
    },
    images_per_class=dict(enumerate([10, 20, 30] * 10)),
    class_names=None,
    path=None,
    file_pattern='{}.tfrecords')

# Define defaults and set Gin configuration for EpisodeDescriptionConfig
MIN_WAYS = 5
MAX_WAYS_UPPER_BOUND = 50
MAX_NUM_QUERY = 10
MAX_SUPPORT_SET_SIZE = 500
MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
MIN_LOG_WEIGHT = np.log(0.5)
MAX_LOG_WEIGHT = np.log(2)

gin.bind_parameter('EpisodeDescriptionConfig.num_ways', None)
gin.bind_parameter('EpisodeDescriptionConfig.num_support', None)
gin.bind_parameter('EpisodeDescriptionConfig.num_query', None)
gin.bind_parameter('EpisodeDescriptionConfig.min_ways', MIN_WAYS)
gin.bind_parameter('EpisodeDescriptionConfig.max_ways_upper_bound',
                   MAX_WAYS_UPPER_BOUND)
gin.bind_parameter('EpisodeDescriptionConfig.max_num_query', MAX_NUM_QUERY)
gin.bind_parameter('EpisodeDescriptionConfig.max_support_set_size',
                   MAX_SUPPORT_SET_SIZE)
gin.bind_parameter(
    'EpisodeDescriptionConfig.max_support_size_contrib_per_class',
    MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS)
gin.bind_parameter('EpisodeDescriptionConfig.min_log_weight', MIN_LOG_WEIGHT)
gin.bind_parameter('EpisodeDescriptionConfig.max_log_weight', MAX_LOG_WEIGHT)
gin.bind_parameter('EpisodeDescriptionConfig.ignore_dag_ontology', False)
gin.bind_parameter('EpisodeDescriptionConfig.ignore_bilevel_ontology', False)


def split_into_chunks(batch, chunk_sizes):
  """Returns batch split in 3 according to chunk_sizes.

  Args:
    batch: A sequence of length sum(chunk_sizes), usually examples or targets.
    chunk_sizes: A tuple of 3 ints (flush_size, support_size, query_size).

  Returns:
    A tuple of 3 sequences (flush_chunk, support_chunk, query_chunk).
  """
  assert sum(chunk_sizes) == len(batch)
  flush_chunk_size, support_chunk_size, _ = chunk_sizes
  query_start = flush_chunk_size + support_chunk_size

  flush_chunk = batch[:flush_chunk_size]
  support_chunk = batch[flush_chunk_size:query_start]
  query_chunk = batch[query_start:]

  return (flush_chunk, support_chunk, query_chunk)


class DatasetIDGenTest(tf.test.TestCase):
  """Tests `reader.dataset_id_generator`."""

  def setUp(self):
    super(DatasetIDGenTest, self).setUp()
    self.dataset_spec = DATASET_SPEC
    self.split = Split.TRAIN

  def check_expected_structure(self, sampler):
    """Checks the stream of dataset indices is as expected."""
    chunk_sizes = sampler.compute_chunk_sizes()
    batch_size = sum(chunk_sizes)
    dummy_id = len(self.dataset_spec.get_classes(self.split))

    generator = reader.dataset_id_generator(self.dataset_spec, self.split, None,
                                            sampler)
    for _ in range(3):
      # Re-assemble batch.
      # TODO(lamblinp): update if we change dataset_id_generator to return
      # the whole batch at once
      batch = list(itertools.islice(generator, batch_size))

      self.assertEqual(len(batch), batch_size)
      flush_chunk, support_chunk, query_chunk = split_into_chunks(
          batch, chunk_sizes)

      # flush_chunk is slightly oversized: if we actually had support_chunk_size
      # + query_chunk_size examples remaining, we could have used them.
      # Therefore, the last element of flush_chunk should be padding.
      self.assertEqual(flush_chunk[-1], dummy_id)
      # TODO(lamblinp): check more about the content of flush_chunk

      # The padding should be at the end of each chunk.
      for chunk in (flush_chunk, support_chunk, query_chunk):
        num_actual_examples = sum(class_id != dummy_id for class_id in chunk)
        self.assertNotIn(dummy_id, chunk[:num_actual_examples])
        self.assertTrue(
            all(dummy_id == class_id
                for class_id in chunk[num_actual_examples:]))

  def test_default(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig())
    self.check_expected_structure(sampler)

  def test_fixed_query(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_query=5))
    self.check_expected_structure(sampler)

  def test_no_query(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_query=5))
    self.check_expected_structure(sampler)

  def test_fixed_shots(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(
            num_support=3, num_query=7))
    self.check_expected_structure(sampler)

  def test_fixed_ways(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_ways=12))
    self.check_expected_structure(sampler)

  def test_fixed_episodes(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(
            num_ways=12, num_support=3, num_query=7))
    self.check_expected_structure(sampler)


def construct_dummy_datasets(class_ids,
                             examples_per_class,
                             repeat=True,
                             shuffle=True,
                             shuffle_seed=None):
  """Construct a list of in-memory dummy datasets.

  Args:
    class_ids: A list of ints, one for each dataset to build.
    examples_per_class: A list of int, how many examples there are in each
      dataset.
    repeat: A Boolean indicating whether each of the datasets should be repeated
      (to provide an infinite stream).
    shuffle: A Boolean indicating whether each dataset should be shuffled.
    shuffle_seed: Optional, an int containing the seed passed to
      tf.data.Dataset.shuffle.

  Returns:
    A list of tf.data.Dataset. Each one contains a series of pairs:
    (a string formatted like '<class_id>.<example_id>', an int: class_id).
  """
  datasets = []
  for i, class_id in enumerate(class_ids):
    num_examples = examples_per_class[i]
    example_string_dataset = tf.data.Dataset.from_tensor_slices(
        ['{}.{}'.format(class_id, ex_id) for ex_id in range(num_examples)])
    if shuffle:
      example_string_dataset = example_string_dataset.shuffle(
          buffer_size=num_examples,
          seed=shuffle_seed,
          reshuffle_each_iteration=True)
    if repeat:
      example_string_dataset = example_string_dataset.repeat()

    class_id_dataset = tf.data.Dataset.from_tensors(class_id).repeat()
    dataset = tf.data.Dataset.zip((example_string_dataset, class_id_dataset))
    datasets.append(dataset)

  return datasets


class DummyEpisodeReader(reader.EpisodeReader):
  """Subclass of EpisodeReader that builds class datasets in-memory."""

  def construct_class_datasets(self,
                               pool=None,
                               repeat=True,
                               shuffle=True,
                               shuffle_seed=None):
    class_ids = [
        self.class_set[dataset_id] for dataset_id in range(self.num_classes)
    ]
    examples_per_class = [
        self.dataset_spec.get_total_images_per_class(class_id)
        for class_id in class_ids
    ]
    shuffle = self.shuffle_buffer_size > 0
    return construct_dummy_datasets(class_ids, examples_per_class, repeat,
                                    shuffle, shuffle_seed)


class EpisodeReaderTest(tf.test.TestCase):
  """Tests behaviour of Reader.

  To avoid reading from the filesystem, we actually test a subclass,
  DummyEpisodeReader, that overrides Reader.construct_class_datasets,
  replacing it with a method building small, in-memory datasets instead.
  """

  def setUp(self):
    super(EpisodeReaderTest, self).setUp()
    self.dataset_spec = DATASET_SPEC
    self.split = Split.TRAIN
    self.shuffle_buffer_size = 30
    self.read_buffer_size_bytes = None
    self.num_prefetch = 0

  def generate_episodes(self,
                        sampler,
                        num_episodes,
                        shuffle=True,
                        shuffle_seed=None):
    dataset_spec = sampler.dataset_spec
    split = sampler.split
    if shuffle:
      shuffle_buffer_size = self.shuffle_buffer_size
    else:
      shuffle_buffer_size = 0

    episode_reader = DummyEpisodeReader(dataset_spec, split,
                                        shuffle_buffer_size,
                                        self.read_buffer_size_bytes,
                                        self.num_prefetch)
    input_pipeline = episode_reader.create_dataset_input_pipeline(
        sampler, shuffle_seed=shuffle_seed)
    iterator = input_pipeline.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      episodes = [sess.run(next_element) for _ in range(num_episodes)]
    return episodes

  def check_episode_consistency(self, examples, targets, chunk_sizes):
    """Tests that a given episode is correctly built and consistent.

    In particular:
    - test that examples come from the right class
    - test that the overall "flush, support, query" structure is respected
    - test that within each chunk, the padding is at the end

    Args:
      examples: A 1D array of strings.
      targets: A 1D array of ints.
      chunk_sizes: A tuple of 3 ints, describing the structure of the episode.
    """
    self.check_consistent_class(examples, targets)

    batch_size = sum(chunk_sizes)
    self.assertEqual(batch_size, len(examples), len(targets))

    flush_examples, support_examples, query_examples = split_into_chunks(
        examples, chunk_sizes)
    flush_targets, support_targets, query_targets = split_into_chunks(
        targets, chunk_sizes)

    self.check_end_padding(flush_examples, flush_targets)
    self.check_end_padding(support_examples, support_targets)
    self.check_end_padding(query_examples, query_targets)

  def check_consistent_class(self, examples, targets):
    """Checks that the content of examples corresponds to the target.

    This assumes the datasets were generated from `construct_dummy_datasets`,
    with a dummy class of DUMMY_CLASS_ID with empty string examples.

    Args:
      examples: A 1D array of strings.
      targets: A 1D array of ints.
    """
    self.assertEqual(len(examples), len(targets))
    for (example, target) in zip(examples, targets):
      if example:
        expected_target, _ = example.decode().split('.')
        self.assertEqual(int(expected_target), target)
      else:
        self.assertEqual(target, reader.DUMMY_CLASS_ID)

  def check_end_padding(self, examples_chunk, targets_chunk):
    """Checks the padding is at the end of each chunk.

    Args:
      examples_chunk: A 1D array of strings.
      targets_chunk: A 1D array of ints.
    """
    num_actual = sum(
        class_id != reader.DUMMY_CLASS_ID for class_id in targets_chunk)
    self.assertNotIn(reader.DUMMY_CLASS_ID, targets_chunk[:num_actual])
    self.assertNotIn(b'', examples_chunk[:num_actual])
    self.assertTrue(
        all(reader.DUMMY_CLASS_ID == target
            for target in targets_chunk[num_actual:]))
    self.assertAllInSet(examples_chunk[num_actual:], [b''])

  def generate_and_check(self, sampler, num_episodes):
    chunk_sizes = sampler.compute_chunk_sizes()
    episodes = self.generate_episodes(sampler, num_episodes)
    for episode in episodes:
      examples, targets = episode
      self.check_episode_consistency(examples, targets, chunk_sizes)

  def test_train(self):
    """Tests that a few episodes are consistent."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        Split.TRAIN,
        episode_descr_config=config.EpisodeDescriptionConfig())
    self.generate_and_check(sampler, 10)

  def test_valid(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        Split.VALID,
        episode_descr_config=config.EpisodeDescriptionConfig())
    self.generate_and_check(sampler, 10)

  def test_test(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        Split.TEST,
        episode_descr_config=config.EpisodeDescriptionConfig())
    self.generate_and_check(sampler, 10)

  def test_fixed_query(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_query=5))
    self.generate_and_check(sampler, 10)

  def test_no_query(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_query=0))
    self.generate_and_check(sampler, 10)

  def test_fixed_shots(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(
            num_support=3, num_query=7))
    self.generate_and_check(sampler, 10)

  def test_fixed_ways(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(num_ways=12))
    self.generate_and_check(sampler, 10)

  def test_fixed_episodes(self):
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec,
        self.split,
        episode_descr_config=config.EpisodeDescriptionConfig(
            num_ways=12, num_support=3, num_query=7))
    self.generate_and_check(sampler, 10)

  def test_non_deterministic_shuffle(self):
    """Different Readers generate different episode compositions.

    Even with the same episode descriptions, the content should be different.
    """
    num_episodes = 10
    init_rng = sampling.RNG
    seed = 20181120
    episode_streams = []
    chunk_sizes = []
    try:
      for _ in range(2):
        sampling.RNG = np.random.RandomState(seed)
        sampler = sampling.EpisodeDescriptionSampler(
            self.dataset_spec,
            self.split,
            episode_descr_config=config.EpisodeDescriptionConfig())
        episodes = self.generate_episodes(sampler, num_episodes)
        episode_streams.append(episodes)
        chunk_size = sampler.compute_chunk_sizes()
        chunk_sizes.append(chunk_size)
        for examples, targets in episodes:
          self.check_episode_consistency(examples, targets, chunk_size)

    finally:
      # Restore the original RNG
      sampling.RNG = init_rng

    self.assertEqual(chunk_sizes[0], chunk_sizes[1])

    # It is unlikely that all episodes will be the same
    num_identical_episodes = 0
    for ((examples1, targets1), (examples2, targets2)) in zip(*episode_streams):
      self.check_episode_consistency(examples1, targets1, chunk_sizes[0])
      self.check_episode_consistency(examples2, targets2, chunk_sizes[1])
      self.assertAllEqual(targets1, targets2)
      if all(examples1 == examples2):
        num_identical_episodes += 1

    self.assertNotEqual(num_identical_episodes, num_episodes)

  def test_deterministic_noshuffle(self):
    """Tests episode generation determinism when there is noshuffle queue."""
    num_episodes = 10
    init_rng = sampling.RNG
    seed = 20181120
    episode_streams = []
    chunk_sizes = []
    try:
      for _ in range(2):
        sampling.RNG = np.random.RandomState(seed)
        sampler = sampling.EpisodeDescriptionSampler(
            self.dataset_spec,
            self.split,
            episode_descr_config=config.EpisodeDescriptionConfig())
        episodes = self.generate_episodes(sampler, num_episodes, shuffle=False)
        episode_streams.append(episodes)
        chunk_size = sampler.compute_chunk_sizes()
        chunk_sizes.append(chunk_size)
        for examples, targets in episodes:
          self.check_episode_consistency(examples, targets, chunk_size)

    finally:
      # Restore the original RNG
      sampling.RNG = init_rng

    self.assertEqual(chunk_sizes[0], chunk_sizes[1])

    for ((examples1, targets1), (examples2, targets2)) in zip(*episode_streams):
      self.assertAllEqual(examples1, examples2)
      self.assertAllEqual(targets1, targets2)

  def test_deterministic_tfseed(self):
    """Tests episode generation determinism when shuffle queues are seeded."""
    num_episodes = 10
    seed = 20181120
    episode_streams = []
    chunk_sizes = []
    init_rng = sampling.RNG
    try:
      for _ in range(2):
        sampling.RNG = np.random.RandomState(seed)
        sampler = sampling.EpisodeDescriptionSampler(
            self.dataset_spec,
            self.split,
            episode_descr_config=config.EpisodeDescriptionConfig())
        episodes = self.generate_episodes(
            sampler, num_episodes, shuffle_seed=seed)
        episode_streams.append(episodes)
        chunk_size = sampler.compute_chunk_sizes()
        chunk_sizes.append(chunk_size)
        for examples, targets in episodes:
          self.check_episode_consistency(examples, targets, chunk_size)

    finally:
      # Restore the original RNG
      sampling.RNG = init_rng

    self.assertEqual(chunk_sizes[0], chunk_sizes[1])

    for ((examples1, targets1), (examples2, targets2)) in zip(*episode_streams):
      self.check_episode_consistency(examples1, targets1, chunk_sizes[0])
      self.check_episode_consistency(examples2, targets2, chunk_sizes[1])
      self.assertAllEqual(examples1, examples2)
      self.assertAllEqual(targets1, targets2)

  def check_description_vs_target_chunks(
      self, description, target_support_chunk, target_query_chunk, offset):
    """Checks that target chunks are consistent with the description.

    The number of support and query exampes should correspond to the
    description, and no other class ID (except DUMMY_CLASS_ID) should be
    present.

    Args:
      description: A sequence of (class_id, num_support, num_query) tuples of
        ints, describing the content of an episode.
      target_support_chunk: A sequence of ints, padded.
      target_query_chunk: A sequence of ints, padded.
      offset: An int, the difference between the absolute class IDs in the
        target, and the relative class IDs in the episode description.
    """
    support_cursor = 0
    query_cursor = 0
    for class_id, num_support, num_query in description:
      self.assertAllEqual(
          target_support_chunk[support_cursor:support_cursor + num_support],
          [class_id + offset] * num_support)
      support_cursor += num_support
      self.assertAllEqual(
          target_query_chunk[query_cursor:query_cursor + num_query],
          [class_id + offset] * num_query)
      query_cursor += num_query

    self.assertTrue(
        all(target_support_chunk[support_cursor:] == reader.DUMMY_CLASS_ID))
    self.assertTrue(
        all(target_query_chunk[query_cursor:] == reader.DUMMY_CLASS_ID))

  def check_same_as_generator(self, split, offset):
    """Tests that the targets are the one requested by the generator.

    Args:
      split: A value of the Split enum, which split to generate from.
      offset: An int, the difference between the absolute class IDs in the
        source, and the relative class IDs in the episodes.
    """
    num_episodes = 10
    seed = 20181121
    init_rng = sampling.RNG
    try:
      sampling.RNG = np.random.RandomState(seed)
      sampler = sampling.EpisodeDescriptionSampler(
          self.dataset_spec,
          split,
          episode_descr_config=config.EpisodeDescriptionConfig())
      # Each description is a (class_id, num_support, num_query) tuple.
      descriptions = [
          sampler.sample_episode_description() for _ in range(num_episodes)
      ]

      sampling.RNG = np.random.RandomState(seed)
      sampler = sampling.EpisodeDescriptionSampler(
          self.dataset_spec,
          split,
          episode_descr_config=config.EpisodeDescriptionConfig())
      episodes = self.generate_episodes(sampler, num_episodes)
      chunk_sizes = sampler.compute_chunk_sizes()
      self.assertEqual(len(descriptions), len(episodes))
      for (description, episode) in zip(descriptions, episodes):
        examples, targets = episode
        self.check_episode_consistency(examples, targets, chunk_sizes)
        _, targets_support_chunk, targets_query_chunk = split_into_chunks(
            targets, chunk_sizes)
        self.check_description_vs_target_chunks(
            description, targets_support_chunk, targets_query_chunk, offset)
    finally:
      sampling.RNG = init_rng

  def test_same_as_generator(self):
    # The offset corresponds to the difference between the absolute class ID as
    # used in the episode pipeline, and class ID relative to the split (provided
    # by the episode generator).
    offset = 0
    for split in Split:
      self.check_same_as_generator(split, offset)
      offset += len(self.dataset_spec.get_classes(split))

  def test_flush_logic(self):
    """Tests the "flush" logic avoiding example duplication in an episode."""
    # Generate two episodes from un-shuffled data sources. For classes where
    # there are enough examples for both, new examples should be used for the
    # second episodes. Otherwise, the first examples should be re-used.
    # A data_spec with classes between 10 and 29 examples.
    num_classes = 30
    dataset_spec = DatasetSpecification(
        name=None,
        classes_per_split={
            Split.TRAIN: num_classes,
            Split.VALID: 0,
            Split.TEST: 0
        },
        images_per_class={i: 10 + i for i in range(num_classes)},
        class_names=None,
        path=None,
        file_pattern='{}.tfrecords')
    # Sample from all train classes, 5 + 5 examples from each episode
    sampler = sampling.EpisodeDescriptionSampler(
        dataset_spec,
        Split.TRAIN,
        episode_descr_config=config.EpisodeDescriptionConfig(
            num_ways=num_classes, num_support=5, num_query=5))
    episodes = self.generate_episodes(sampler, num_episodes=2, shuffle=False)

    # The "flush" part of the second episode should contain 0 from class_id 0, 1
    # for 1, ..., 9 for 9, and then 0 for 10 and the following.
    chunk_sizes = sampler.compute_chunk_sizes()
    _, episode2 = episodes
    examples2, targets2 = episode2
    flush_target2, _, _ = split_into_chunks(targets2, chunk_sizes)
    for class_id in range(10):
      self.assertEqual(
          sum(target == class_id for target in flush_target2), class_id)
    for class_id in range(10, num_classes):
      self.assertEqual(sum(target == class_id for target in flush_target2), 0)

    # The "support" part of the second episode should start at example 0 for
    # class_ids from 0 to 9 (included), and at example 10 for class_id 10 and
    # higher.
    _, support_examples2, query_examples2 = split_into_chunks(
        examples2, chunk_sizes)

    def _build_class_id_to_example_ids(examples):
      # Build a mapping: class_id -> list of example ids
      mapping = collections.defaultdict(list)
      for example in examples:
        if not example:
          # Padding is at the end
          break
        class_id, example_id = example.decode().split('.')
        mapping[int(class_id)].append(int(example_id))
      return mapping

    support2_example_ids = _build_class_id_to_example_ids(support_examples2)
    query2_example_ids = _build_class_id_to_example_ids(query_examples2)

    for class_id in range(10):
      self.assertCountEqual(support2_example_ids[class_id], list(range(5)))
      self.assertCountEqual(query2_example_ids[class_id], list(range(5, 10)))

    for class_id in range(10, num_classes):
      self.assertCountEqual(support2_example_ids[class_id], list(range(10, 15)))
      self.assertCountEqual(query2_example_ids[class_id], list(range(15, 20)))


if __name__ == '__main__':
  tf.test.main()
