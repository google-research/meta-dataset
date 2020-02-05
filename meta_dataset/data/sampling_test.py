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
"""Tests for `sampling` module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import gin.tf
from meta_dataset.data import config
from meta_dataset.data import sampling
from meta_dataset.data import test_utils
from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

test_utils.set_episode_descr_config_defaults()


class SampleNumWaysUniformlyTest(tf.test.TestCase):
  """Tests for the `sample_num_ways_uniformly` function."""

  def test_min_ways_respected(self):
    for _ in range(10):
      num_ways = sampling.sample_num_ways_uniformly(
          10,
          min_ways=test_utils.MIN_WAYS,
          max_ways=test_utils.MAX_WAYS_UPPER_BOUND)
      self.assertGreaterEqual(num_ways, test_utils.MIN_WAYS)

  def test_num_classes_respected(self):
    num_classes = 10
    for _ in range(10):
      num_ways = sampling.sample_num_ways_uniformly(
          num_classes,
          min_ways=test_utils.MIN_WAYS,
          max_ways=test_utils.MAX_WAYS_UPPER_BOUND)
      self.assertLessEqual(num_ways, num_classes)

  def test_max_ways_upper_bound_respected(self):
    num_classes = 2 * test_utils.MAX_WAYS_UPPER_BOUND
    for _ in range(10):
      num_ways = sampling.sample_num_ways_uniformly(
          num_classes,
          min_ways=test_utils.MIN_WAYS,
          max_ways=test_utils.MAX_WAYS_UPPER_BOUND)
      self.assertLessEqual(num_ways, test_utils.MAX_WAYS_UPPER_BOUND)


class SampleClassIDsUniformlyTest(tf.test.TestCase):
  """Tests for the `sample_class_ids_uniformly` function."""

  def test_num_ways_respected(self):
    num_classes = test_utils.MAX_WAYS_UPPER_BOUND
    num_ways = test_utils.MIN_WAYS
    for _ in range(10):
      class_ids = sampling.sample_class_ids_uniformly(num_ways, num_classes)
      self.assertLen(set(class_ids), num_ways)
      self.assertLen(class_ids, num_ways)

  def test_num_classes_respected(self):
    num_classes = test_utils.MAX_WAYS_UPPER_BOUND
    num_ways = test_utils.MIN_WAYS
    for _ in range(10):
      class_ids = sampling.sample_class_ids_uniformly(num_ways, num_classes)
      self.assertContainsSubset(class_ids, list(range(num_classes)))

  def test_unique_class_ids(self):
    num_classes = test_utils.MAX_WAYS_UPPER_BOUND
    num_ways = test_utils.MIN_WAYS
    for _ in range(10):
      class_ids = sampling.sample_class_ids_uniformly(num_ways, num_classes)
      self.assertCountEqual(class_ids, set(class_ids))


class ComputeNumQueryTest(tf.test.TestCase):
  """Tests for the `compute_num_query` function."""

  def test_max_num_query_respected(self):
    images_per_class = np.array([30, 45, 35, 50])
    num_query = sampling.compute_num_query(
        images_per_class, max_num_query=test_utils.MAX_NUM_QUERY)
    self.assertEqual(num_query, test_utils.MAX_NUM_QUERY)

  def test_at_most_half(self):
    images_per_class = np.array([10, 9, 20, 21])
    num_query = sampling.compute_num_query(
        images_per_class, max_num_query=test_utils.MAX_NUM_QUERY)
    self.assertEqual(num_query, 4)

  def test_raises_error_on_one_image_per_class(self):
    images_per_class = np.array([1, 3, 8, 8])
    with self.assertRaises(ValueError):
      sampling.compute_num_query(
          images_per_class, max_num_query=test_utils.MAX_NUM_QUERY)


class SampleSupportSetSizeTest(tf.test.TestCase):
  """Tests for the `sample_support_set_size` function."""

  def test_max_support_set_size_respected(self):
    num_remaining_per_class = np.array([test_utils.MAX_SUPPORT_SET_SIZE] * 10)
    for _ in range(10):
      support_set_size = sampling.sample_support_set_size(
          num_remaining_per_class,
          max_support_size_contrib_per_class=(
              test_utils.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS),
          max_support_set_size=test_utils.MAX_SUPPORT_SET_SIZE)
      self.assertLessEqual(support_set_size, test_utils.MAX_SUPPORT_SET_SIZE)

  def test_raises_error_max_support_too_small(self):
    num_remaining_per_class = np.array([5] * 10)
    with self.assertRaises(ValueError):
      sampling.sample_support_set_size(
          num_remaining_per_class,
          max_support_size_contrib_per_class=(
              test_utils.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS),
          max_support_set_size=len(num_remaining_per_class) - 1)


class SampleNumSupportPerClassTest(tf.test.TestCase):
  """Tests for the `sample_num_support_per_class` function."""

  def test_support_set_size_respected(self):
    num_images_per_class = np.array([50, 40, 30, 20])
    num_remaining_per_class = np.array([40, 30, 20, 10])
    support_set_size = 50
    for _ in range(10):
      num_support_per_class = sampling.sample_num_support_per_class(
          num_images_per_class,
          num_remaining_per_class,
          support_set_size,
          min_log_weight=test_utils.MIN_LOG_WEIGHT,
          max_log_weight=test_utils.MAX_LOG_WEIGHT)
      self.assertLessEqual(num_support_per_class.sum(), support_set_size)

  def test_at_least_one_example_per_class(self):
    num_images_per_class = np.array([10, 10, 10, 10, 10])
    num_remaining_per_class = np.array([5, 5, 5, 5, 5])
    support_set_size = 5
    for _ in range(10):
      num_support_per_class = sampling.sample_num_support_per_class(
          num_images_per_class,
          num_remaining_per_class,
          support_set_size,
          min_log_weight=test_utils.MIN_LOG_WEIGHT,
          max_log_weight=test_utils.MAX_LOG_WEIGHT)
      self.assertTrue((num_support_per_class > 0).any())

  def test_complains_on_too_small_support_set_size(self):
    num_images_per_class = np.array([10, 10, 10, 10, 10])
    num_remaining_per_class = np.array([5, 5, 5, 5, 5])
    support_set_size = 3
    with self.assertRaises(ValueError):
      sampling.sample_num_support_per_class(
          num_images_per_class,
          num_remaining_per_class,
          support_set_size,
          min_log_weight=test_utils.MIN_LOG_WEIGHT,
          max_log_weight=test_utils.MAX_LOG_WEIGHT)

  def test_complains_on_zero_remaining(self):
    num_images_per_class = np.array([10, 10, 10, 10, 10])
    num_remaining_per_class = np.array([5, 0, 5, 5, 5])
    support_set_size = 5
    with self.assertRaises(ValueError):
      sampling.sample_num_support_per_class(
          num_images_per_class,
          num_remaining_per_class,
          support_set_size,
          min_log_weight=test_utils.MIN_LOG_WEIGHT,
          max_log_weight=test_utils.MAX_LOG_WEIGHT)


# TODO(vdumoulin): move this class into `config_test.py`.
class EpisodeDescrSamplerErrorTest(parameterized.TestCase, tf.test.TestCase):
  """Episode sampler should verify args when ways/shots are sampled."""
  dataset_spec = test_utils.DATASET_SPEC
  split = Split.VALID

  @parameterized.named_parameters(('num_ways_none', None, 5, 10, {}),
                                  ('num_ways_none2', None, 5, 10, {
                                      'min_ways': 3
                                  }), ('num_support_none', 5, None, 10, {}),
                                  ('num_support_none2', 5, None, 10, {
                                      'max_support_set_size': 3
                                  }), ('num_query_none', 5, 5, None, {}))
  def test_runtime_errors(self, num_ways, num_support, num_query, kwargs):
    """Testing run-time errors thrown when arguments are not set correctly."""
    # The following scope removes the gin-config set.
    with gin.config_scope('none'):
      with self.assertRaises(RuntimeError):
        _ = sampling.EpisodeDescriptionSampler(
            self.dataset_spec,
            self.split,
            episode_descr_config=config.EpisodeDescriptionConfig(
                num_ways=num_ways,
                num_support=num_support,
                num_query=num_query,
                **kwargs))

  @parameterized.named_parameters(('num_ways_none', None, 5, 10, {
      'min_ways': 3,
      'max_ways_upper_bound': 5
  }), ('num_support_none', 5, None, 10, {
      'max_support_set_size': 3,
      'max_support_size_contrib_per_class': 5,
      'min_log_weight': 0.5,
      'max_log_weight': 0.5
  }), ('num_query_none', 5, 5, None, {
      'max_num_query': 3
  }))
  def test_runtime_no_error(self, num_ways, num_support, num_query, kwargs):
    """Testing run-time errors thrown when arguments are not set correctly."""
    # The following scope removes the gin-config set.
    with gin.config_scope('none'):
      # No error thrown
      _ = sampling.EpisodeDescriptionSampler(
          self.dataset_spec,
          self.split,
          episode_descr_config=config.EpisodeDescriptionConfig(
              num_ways=num_ways,
              num_support=num_support,
              num_query=num_query,
              **kwargs))


class EpisodeDescrSamplerTest(tf.test.TestCase):
  """Tests EpisodeDescriptionSampler defaults.

  This class provides some tests to be run by inherited classes.
  """

  dataset_spec = test_utils.DATASET_SPEC
  split = Split.VALID

  def setUp(self):
    super(EpisodeDescrSamplerTest, self).setUp()
    self.sampler = self.make_sampler()

  def make_sampler(self):
    """Helper function to make a new instance of the tested sampler."""
    return sampling.EpisodeDescriptionSampler(self.dataset_spec, self.split,
                                              config.EpisodeDescriptionConfig())

  def test_max_examples(self):
    """The number of requested examples per class should not be too large."""
    class_set = self.dataset_spec.get_classes(self.split)
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      self.assertTrue(
          all(s +
              q <= self.dataset_spec.get_total_images_per_class(class_set[cid])
              for cid, s, q in episode_description))

  def test_min_examples(self):
    """There should be at least 1 support and query example per class."""
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      self.assertTrue(
          all(s >= 1 and q >= 1 for cid, s, q in episode_description))

  def test_non_deterministic(self):
    """By default, generated episodes should be different across Samplers."""
    reference_sample = self.sampler.sample_episode_description()
    for _ in range(10):
      sampler = self.make_sampler()
      sample = sampler.sample_episode_description()
      if sample != reference_sample:
        # Test should pass
        break
    else:
      # The end of the loop was reached with no "break" triggered.
      # If it generated the same description 11 times, this is an error.
      raise AssertionError('Different EpisodeDescriptionSamplers generate '
                           'the same sequence of episode descriptions.')

  def test_setting_randomstate(self):
    """Setting the RNG state should make episode generation deterministic."""
    init_rng = sampling.RNG
    seed = 20181113
    try:
      sampling.RNG = np.random.RandomState(seed)
      sampler = self.make_sampler()
      reference_sample = sampler.sample_episode_description()
      for _ in range(10):
        sampling.RNG = np.random.RandomState(seed)
        sampler = self.make_sampler()
        sample = sampler.sample_episode_description()
        self.assertEqual(reference_sample, sample)

    finally:
      # Restore the original RNG
      sampling.RNG = init_rng

  def assert_expected_chunk_sizes(self, expected_support_chunk_size,
                                  expected_query_chunk_size):
    rval = self.sampler.compute_chunk_sizes()
    flush_chunk_size, support_chunk_size, query_chunk_size = rval

    expected_flush_chunk_size = (
        expected_support_chunk_size + expected_query_chunk_size)
    self.assertEqual(flush_chunk_size, expected_flush_chunk_size)
    self.assertEqual(support_chunk_size, expected_support_chunk_size)
    self.assertEqual(query_chunk_size, expected_query_chunk_size)

  def test_correct_chunk_sizes(self):
    self.assert_expected_chunk_sizes(
        test_utils.MAX_SUPPORT_SET_SIZE,
        test_utils.MAX_WAYS_UPPER_BOUND * test_utils.MAX_NUM_QUERY)


class FixedQueryEpisodeDescrSamplerTest(EpisodeDescrSamplerTest):
  """Tests EpisodeDescriptionSampler with fixed query set.

  Inherits from EpisodeDescrSamplerTest so:
    - Tests defined in the parent class will be run
    - parent setUp method will be called
    - make_sampler is overridden.
  """

  split = Split.TRAIN
  num_query = 5

  def make_sampler(self):
    return sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(num_query=self.num_query))

  def test_num_query_examples(self):
    class_set = self.dataset_spec.get_classes(self.split)
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      for cid, _, q in episode_description:
        self.assertIn(cid, class_set)
        self.assertEqual(q, self.num_query)

  def test_query_too_big(self):
    """Asserts failure if all examples of a class are selected for query."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(num_query=10))
    with self.assertRaises(ValueError):
      # Sample enough times that we encounter a class with only 10 examples.
      for _ in range(10):
        sampler.sample_episode_description()

  def test_correct_chunk_sizes(self):
    self.assert_expected_chunk_sizes(
        test_utils.MAX_SUPPORT_SET_SIZE,
        test_utils.MAX_WAYS_UPPER_BOUND * self.num_query)


class NoQueryEpisodeDescrSamplerTest(FixedQueryEpisodeDescrSamplerTest):
  """Tests EpisodeDescriptionSampler with no query set.

  Special case of FixedQueryEpisodeDescrSamplerTest with num_query = 0.
  """
  num_query = 0

  def test_min_examples(self):
    """Overrides base class because 0 query examples is actually expected."""
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      self.assertTrue(all(s >= 1 for cid, s, q in episode_description))


class FixedShotsEpisodeDescrSamplerTest(FixedQueryEpisodeDescrSamplerTest):
  """Tests EpisodeDescriptionSampler with fixed support and query size.

  Inherits form FixedQueryEpisodeDescrSamplerTest, so parent tests, including
  test_num_query_examples will be run as well.
  """
  # Chosen so num_support + num_query <= 10, since some classes have 10 ex.
  num_support = 3
  num_query = 7

  def make_sampler(self):
    return sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(
            num_support=self.num_support, num_query=self.num_query))

  def test_num_support_examples(self):
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      for _, s, _ in episode_description:
        self.assertEqual(s, self.num_support)

  def test_shots_too_big(self):
    """Asserts failure if not enough examples to fulfill support and query."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(num_support=5, num_query=15))
    with self.assertRaises(ValueError):
      sampler.sample_episode_description()

  def test_correct_chunk_sizes(self):
    self.assert_expected_chunk_sizes(
        test_utils.MAX_WAYS_UPPER_BOUND * self.num_support,
        test_utils.MAX_WAYS_UPPER_BOUND * self.num_query)


class FixedWaysEpisodeDescrSamplerTest(EpisodeDescrSamplerTest):
  """Tests EpisodeDescriptionSampler with fixed number of classes."""
  split = Split.TRAIN
  num_ways = 12

  def make_sampler(self):
    return sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(num_ways=self.num_ways))

  def test_num_ways(self):
    for _ in range(10):
      episode_description = self.sampler.sample_episode_description()
      self.assertLen((episode_description), self.num_ways)

  def test_ways_too_big(self):
    """Asserts failure if more ways than classes are available."""
    # Use Split.VALID as it only has 10 classes.
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, Split.VALID,
        config.EpisodeDescriptionConfig(num_ways=self.num_ways))
    with self.assertRaises(ValueError):
      sampler.sample_episode_description()

  def test_correct_chunk_sizes(self):
    self.assert_expected_chunk_sizes(test_utils.MAX_SUPPORT_SET_SIZE,
                                     self.num_ways * test_utils.MAX_NUM_QUERY)


class FixedEpisodeDescrSamplerTest(FixedShotsEpisodeDescrSamplerTest,
                                   FixedWaysEpisodeDescrSamplerTest):
  """Tests EpisodeDescriptionSampler with fixed shots and ways."""

  def make_sampler(self):
    return sampling.EpisodeDescriptionSampler(
        self.dataset_spec, self.split,
        config.EpisodeDescriptionConfig(
            num_ways=self.num_ways,
            num_support=self.num_support,
            num_query=self.num_query))

  def test_correct_chunk_sizes(self):
    self.assert_expected_chunk_sizes(self.num_ways * self.num_support,
                                     self.num_ways * self.num_query)


class ChunkSizesTest(tf.test.TestCase):
  """Tests the boundaries of compute_chunk_sizes."""

  def setUp(self):
    super(ChunkSizesTest, self).setUp()
    # Set up a DatasetSpecification with lots of classes and samples.
    self.dataset_spec = DatasetSpecification(
        name=None,
        classes_per_split=dict(zip(Split, [1000, 0, 0])),
        images_per_class={i: 1000 for i in range(1000)},
        class_names=None,
        path=None,
        file_pattern='{}.tfrecords')

  def test_large_support(self):
    """Support set larger than MAX_SUPPORT_SET_SIZE with fixed shots."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, Split.TRAIN,
        config.EpisodeDescriptionConfig(num_ways=30, num_support=20))
    _, support_chunk_size, _ = sampler.compute_chunk_sizes()
    self.assertGreater(support_chunk_size, test_utils.MAX_SUPPORT_SET_SIZE)
    sampler.sample_episode_description()

  def test_large_ways(self):
    """Fixed num_ways above MAX_WAYS_UPPER_BOUND."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, Split.TRAIN,
        config.EpisodeDescriptionConfig(num_ways=60, num_support=10))
    _, support_chunk_size, query_chunk_size = sampler.compute_chunk_sizes()
    self.assertGreater(support_chunk_size, test_utils.MAX_SUPPORT_SET_SIZE)
    self.assertGreater(
        query_chunk_size,
        test_utils.MAX_WAYS_UPPER_BOUND * test_utils.MAX_NUM_QUERY)
    sampler.sample_episode_description()

  def test_large_query(self):
    """Query set larger than MAX_NUM_QUERY per class."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, Split.TRAIN,
        config.EpisodeDescriptionConfig(num_query=60))
    _, _, query_chunk_size = sampler.compute_chunk_sizes()
    self.assertGreater(
        query_chunk_size,
        test_utils.MAX_WAYS_UPPER_BOUND * test_utils.MAX_NUM_QUERY)
    sampler.sample_episode_description()

  def test_too_many_ways(self):
    """Too many ways to have 1 example per class with default variable shots."""
    sampler = sampling.EpisodeDescriptionSampler(
        self.dataset_spec, Split.TRAIN,
        config.EpisodeDescriptionConfig(num_ways=600))
    with self.assertRaises(ValueError):
      sampler.sample_episode_description()


# TODO(lamblinp)
# - test with use_hierarchy=True

if __name__ == '__main__':
  tf.test.main()
