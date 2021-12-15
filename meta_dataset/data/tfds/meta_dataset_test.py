# coding=utf-8
# Copyright 2022 The Meta-Dataset Authors.
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

r"""Integration tests for the TFDS implementation of Meta-Dataset.

Verifies that the data output by TFDS matches the Meta-Dataset tfrecord files,
and that both implementations return the same episodes for a given random seed.

Follow the instructions in the `meta_dataset` module to download and convert the
data to ~/tensorflow_datasets/, and make sure that the `meta_dataset_path` flag
points to Meta-Dataset's records root directory.
"""

import collections
import concurrent.futures
import functools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import gin.tf
import meta_dataset
from meta_dataset import learners
from meta_dataset.data import config as config_lib
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import providers
from meta_dataset.data import sampling
from meta_dataset.data.tfds import api
from meta_dataset.data.tfds import test_utils
from meta_dataset.models import functional_backbones
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS
# Make the pipeline fully deterministic and lower memory consumption for tests.
_DETERMINISTIC_CONFIG = """\
DataConfig.shuffle_buffer_size = 0
DataConfig.read_buffer_size_bytes = 1024
DataConfig.num_prefetch = 4
"""
_NUM_WORKERS = 10
_DEFAULT_TFDS_PATH = f"{os.environ['HOME']}/tensorflow_datasets"
_DEFAULT_META_DATASET_PATH = f"{os.environ['HOME']}/meta_dataset/records"

flags.DEFINE_string('tfds_path', _DEFAULT_TFDS_PATH, 'TFDS data path.')
flags.DEFINE_string('meta_dataset_path', _DEFAULT_META_DATASET_PATH,
                    'Meta-Dataset data path.')


class MetaDatasetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *test_utils.make_class_dataset_comparison_test_cases())
  def test_as_class_dataset_matches_meta_dataset(self,
                                                 source,
                                                 md_source,
                                                 md_version,
                                                 meta_split,
                                                 num_labels,
                                                 offset,
                                                 remap_labels):

    builder = tfds.builder(
        'meta_dataset', config=source, data_dir=FLAGS.tfds_path)

    # For MD-v2's ilsvrc_2012 source, train classes are sorted by class name,
    # whereas the TFDS implementation intentionally keeps the v1 class order.
    if remap_labels:
      # The first argsort tells us where to look in class_names for position i
      # in the sorted class list. The second argsort reverses that: it tells us,
      # for position j in class_names, where to place that class in the sorted
      # class list.
      label_map = np.argsort(np.argsort(builder.info.metadata['class_names']))

    def _compare_class_data(relative_label):
      tfds_dataset = builder.as_class_dataset(
          md_version=md_version,
          meta_split=meta_split,
          relative_label=relative_label,
          decoders={'image': tfds.decode.SkipDecoding()}
      ).batch(500_000, drop_remainder=False)
      tfds_data, = tfds.as_numpy(tfds_dataset)

      absolute_label = relative_label + offset
      md_label = label_map[absolute_label] if remap_labels else absolute_label
      md_dataset = tf.data.TFRecordDataset(
          f'{FLAGS.meta_dataset_path}/{md_source}/{md_label}.tfrecords'
      ).map(test_utils.parse_example).batch(500_000, drop_remainder=False)
      md_data, = tfds.as_numpy(md_dataset)

      if remap_labels:
        tfds_data['label'] = np.vectorize(
            lambda x: label_map[x])(tfds_data['label'])

      np.testing.assert_array_equal(tfds_data['image'], md_data['image'])
      np.testing.assert_array_equal(tfds_data['label'], md_data['label'])

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_NUM_WORKERS) as executor:
      _ = list(executor.map(_compare_class_data, range(num_labels)))

  def test_nonexistent_md_version_raises_value_error(self):
    with self.assertRaises(ValueError):
      tfds.builder(
          'meta_dataset', config='vgg_flower', data_dir=FLAGS.tfds_path
      ).get_start_stop('v2', 'train')
    with self.assertRaises(ValueError):
      tfds.builder(
          'meta_dataset', config='vgg_flower', data_dir=FLAGS.tfds_path
      ).get_start_stop('v2', 'train', 0)

  @parameterized.named_parameters(
      ('ilsvrc_2012_v2_valid', 'ilsvrc_2012', 'v2', 'valid'),
      ('ilsvrc_2012_v2_test', 'ilsvrc_2012', 'v2', 'test'),
      ('mscoco_2012_v1_train', 'mscoco', 'v1', 'train'),
      ('mscoco_2012_v2_train', 'mscoco', 'v2', 'train'),
      ('traffic_sign_v1_train', 'traffic_sign', 'v1', 'train'),
      ('traffic_sign_v1_valid', 'traffic_sign', 'v1', 'valid'),
      ('traffic_sign_v2_train', 'traffic_sign', 'v2', 'train'),
      ('traffic_sign_v2_valid', 'traffic_sign', 'v2', 'valid'),
  )
  def test_nonexistent_meta_split_raises_value_error(self,
                                                     source,
                                                     md_version,
                                                     meta_split):
    with self.assertRaises(ValueError):
      tfds.builder(
          'meta_dataset', config=source, data_dir=FLAGS.tfds_path
      ).get_start_stop(md_version, meta_split)
    with self.assertRaises(ValueError):
      tfds.builder(
          'meta_dataset', config=source, data_dir=FLAGS.tfds_path
      ).get_start_stop(md_version, meta_split, 0)

  def test_meta_split_supports_both_lower_upper_case(self):

    def _load_data(meta_split):
      tfds_dataset = tfds.builder(
          'meta_dataset',
          config='dtd',
          data_dir=FLAGS.tfds_path
      ).as_class_dataset(
          'v1', meta_split, 0, decoders={'image': tfds.decode.SkipDecoding()}
      ).batch(500_000, drop_remainder=False)
      tfds_data, = tfds.as_numpy(tfds_dataset)
      return tfds_data

    tfds_data_lower, tfds_data_upper = _load_data('train'), _load_data('TRAIN')

    np.testing.assert_array_equal(tfds_data_lower['image'],
                                  tfds_data_upper['image'])
    np.testing.assert_array_equal(tfds_data_lower['label'],
                                  tfds_data_upper['label'])

  def test_disable_ordering_guard(self):
    # Shuffling a sliced dataset should still yield the same examples.
    source = 'fungi'
    split = 'valid'
    label = 13
    offset = 994
    shuffle_seed = 1234

    tfds_dataset = tfds.builder(
        'meta_dataset',
        config=source,
        data_dir=FLAGS.tfds_path
    ).as_class_dataset(
        md_version='v1',
        meta_split=split,
        relative_label=label,
        decoders={'image': tfds.decode.SkipDecoding()},
        batch_size=-1,
        shuffle_files=True,
        read_config=tfds.ReadConfig(
            shuffle_seed=shuffle_seed,
            enable_ordering_guard=False))
    tfds_images = tuple(tfds.as_numpy(tfds_dataset)['image'])

    md_dataset = tf.data.TFRecordDataset(
        f'{FLAGS.meta_dataset_path}/{source}/{label + offset}.tfrecords'
    ).map(test_utils.parse_example).batch(10_000, drop_remainder=False)
    md_data, = tfds.as_numpy(md_dataset)
    md_images = tuple(md_data['image'])

    # The order is not the same (`shuffle_files = True`)...
    self.assertNotEqual(tfds_images, md_images)
    # ... but the contents are the same.
    self.assertSameElements(tfds_images, md_images)

  def test_slice_split_workaround(self):
    md_source = 'aircraft'
    md_version = 'v1'
    meta_split = 'train'

    # Infer how many training examples there are, and decide how many to reserve
    # as validation examples.
    builder = tfds.builder(
        'meta_dataset', config=md_source, data_dir=FLAGS.tfds_path)
    start, stop = builder.get_start_stop(md_version, meta_split)
    num_examples = stop - start
    num_valid = int(0.2 * num_examples)

    dataset = builder.as_dataset(
        split=f'all_classes[{start}:{stop}]',
        shuffle_files=False,
        decoders={'image': tfds.decode.SkipDecoding()},
        read_config=tfds.ReadConfig(
            # Adding TFDS IDs allows us to select a random subset of the
            # training split and filter it out to create an unofficial
            # validation split.
            add_tfds_id=True,
            interleave_cycle_length=builder.info.splits[
                f'all_classes[{start}:{stop}]'].num_shards,
            interleave_block_length=4,
            enable_ordering_guard=False)
        # A large shuffle buffer is required in order to ensure that labels are
        # uncorrelated.
    ).shuffle(num_examples)

    # The validation set is usually small, so we can afford to cache it in
    # memory.
    valid_dataset = dataset.take(num_valid).cache().batch(num_valid)
    # Retrieve the TFDS IDs of the validation examples.
    valid_tfds_ids_np = next(valid_dataset.as_numpy_iterator())['tfds_id']

    # Validation examples are the same across iterations and are not affected
    # by the shuffling operation used to instantiate the dataset.
    valid_1, = valid_dataset.as_numpy_iterator()
    valid_2, = valid_dataset.as_numpy_iterator()
    self.assertEqual(set(valid_1['tfds_id']), set(valid_2['tfds_id']))

    def filter_fn(e):
      return tf.math.reduce_all(
          tf.math.not_equal(e['tfds_id'], tf.constant(valid_tfds_ids_np)))

    train_dataset = dataset.filter(filter_fn).batch(num_examples - num_valid)

    # Training examples are the same across iterations, but the order is not the
    # same from one iteration to the next due to the shuffling operation used to
    # instantiate the dataset.
    train_1, = train_dataset.as_numpy_iterator()
    train_2, = train_dataset.as_numpy_iterator()
    self.assertEqual(set(train_1['tfds_id']), set(train_2['tfds_id']))
    self.assertNotEqual(list(train_1['tfds_id']), list(train_2['tfds_id']))

    # The training and validation sets defined partition the training split.
    all_1, = dataset.batch(num_examples).as_numpy_iterator()
    self.assertEqual(set(train_1['tfds_id']).union(set(valid_1['tfds_id'])),
                     set(all_1['tfds_id']))
    self.assertEmpty(
        set(train_1['tfds_id']).intersection(set(valid_1['tfds_id'])))


class APITest(parameterized.TestCase):

  def test_meta_dataset(self):
    gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                          'learn/gin/setups/data_config_tfds.gin')
    gin.parse_config(_DETERMINISTIC_CONFIG)
    data_config = config_lib.DataConfig()
    seed = 20210917
    num_episodes = 10
    meta_split = 'valid'
    md_sources = ('aircraft', 'cu_birds', 'vgg_flower')

    sampling.RNG = np.random.RandomState(seed)
    tfds_episode_dataset = api.meta_dataset(
        md_sources=md_sources, md_version='v1', meta_split=meta_split,
        source_sampling_seed=seed + 1, data_dir=FLAGS.tfds_path)
    tfds_episodes = list(
        tfds_episode_dataset.take(num_episodes).as_numpy_iterator())

    sampling.RNG = np.random.RandomState(seed)
    dataset_spec_list = [
        dataset_spec_lib.load_dataset_spec(os.path.join(FLAGS.meta_dataset_path,
                                                        md_source))
        for md_source in md_sources
    ]
    # We should not skip TFExample decoding in the original Meta-Dataset
    # implementation. The kwarg defaults to False when the class is defined, but
    # the call to `gin.parse_config_file` above changes the default value to
    # True, which is why we have to explicitly bind a new default value here.
    gin.bind_parameter('ImageDecoder.skip_tfexample_decoding', False)
    md_episode_dataset = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=dataset_spec_list,
        use_dag_ontology_list=['ilsvrc_2012' in dataset_spec.name
                               for dataset_spec in dataset_spec_list],
        use_bilevel_ontology_list=[dataset_spec.name == 'omniglot'
                                   for dataset_spec in dataset_spec_list],
        split=getattr(learning_spec.Split, meta_split.upper()),
        episode_descr_config=config_lib.EpisodeDescriptionConfig(),
        pool=None,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        image_size=data_config.image_height,
        source_sampling_seed=seed + 1
    )
    md_episodes = list(
        md_episode_dataset.take(num_episodes).as_numpy_iterator())

    for (tfds_episode, tfds_source_id), (md_episode, md_source_id) in zip(
        tfds_episodes, md_episodes):

      np.testing.assert_equal(tfds_source_id, md_source_id)

      for tfds_tensor, md_tensor in zip(tfds_episode, md_episode):
        np.testing.assert_allclose(tfds_tensor, md_tensor)

  def test_episode_dataset_no_source_id(self):

    gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                          'learn/gin/setups/data_config_tfds.gin')
    gin.parse_config(_DETERMINISTIC_CONFIG)
    tfds_episode_dataset = api.episode_dataset(
        tfds.builder(
            'meta_dataset',
            config='vgg_flower',
            data_dir=FLAGS.tfds_path),
        'v1',
        'valid')
    self.assertEqual(
        tuple(spec.dtype for spec in tfds_episode_dataset.element_spec),
        (tf.float32, tf.int32, tf.int64, tf.float32, tf.int32, tf.int64))

  @parameterized.named_parameters(*(
      args[:5] + args[-1:] for args in
      test_utils.make_class_dataset_comparison_test_cases()))
  def test_episode_dataset_matches_meta_dataset(self,
                                                source,
                                                md_source,
                                                md_version,
                                                meta_split,
                                                remap_labels):

    gin.parse_config_file(tfds.core.as_path(meta_dataset.__file__).parent /
                          'learn/gin/setups/data_config_tfds.gin')
    gin.parse_config(_DETERMINISTIC_CONFIG)
    data_config = config_lib.DataConfig()
    seed = 20210917
    num_episodes = 10

    builder = tfds.builder(
        'meta_dataset', config=source, data_dir=FLAGS.tfds_path)
    sampling.RNG = np.random.RandomState(seed)
    tfds_episode_dataset = api.episode_dataset(
        builder, md_version, meta_split, source_id=0)
    # For MD-v2's ilsvrc_2012 source, train classes are sorted by class name,
    # whereas the TFDS implementation intentionally keeps the v1 class order.
    if remap_labels:
      # The first argsort tells us where to look in class_names for position i
      # in the sorted class list. The second argsort reverses that: it tells us,
      # for position j in class_names, where to place that class in the sorted
      # class list.
      label_map = np.argsort(np.argsort(builder.info.metadata['class_names']))
      label_remap_fn = np.vectorize(lambda x: label_map[x])
    tfds_episodes = list(
        tfds_episode_dataset.take(num_episodes).as_numpy_iterator())

    dataset_spec = dataset_spec_lib.load_dataset_spec(
        os.path.join(FLAGS.meta_dataset_path, md_source))
    sampling.RNG = np.random.RandomState(seed)
    # We should not skip TFExample decoding in the original Meta-Dataset
    # implementation.
    gin.bind_parameter('ImageDecoder.skip_tfexample_decoding', False)
    md_episode_dataset = pipeline.make_one_source_episode_pipeline(
        dataset_spec,
        use_dag_ontology='ilsvrc_2012' in dataset_spec.name,
        use_bilevel_ontology=dataset_spec.name == 'omniglot',
        split=getattr(learning_spec.Split, meta_split.upper()),
        episode_descr_config=config_lib.EpisodeDescriptionConfig(),
        pool=None,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        image_size=data_config.image_height
    )
    md_episodes = list(
        md_episode_dataset.take(num_episodes).as_numpy_iterator())

    for (tfds_episode, tfds_source_id), (md_episode, md_source_id) in zip(
        tfds_episodes, md_episodes):

      np.testing.assert_equal(tfds_source_id, md_source_id)

      if remap_labels:
        tfds_episode = list(tfds_episode)
        tfds_episode[2] = label_remap_fn(tfds_episode[2])
        tfds_episode[5] = label_remap_fn(tfds_episode[5])
        tfds_episode = tuple(tfds_episode)

      for tfds_tensor, md_tensor in zip(tfds_episode, md_episode):
        np.testing.assert_allclose(tfds_tensor, md_tensor)

  @parameterized.named_parameters(*(
      args[:5] + args[-1:] for args in
      test_utils.make_class_dataset_comparison_test_cases()))
  def test_full_ways_dataset(self, source, md_source, md_version, meta_split,
                             remap_labels):

    dataset_spec = dataset_spec_lib.load_dataset_spec(
        os.path.join(FLAGS.meta_dataset_path, md_source))
    allowed_classes = set()
    forbidden_classes = set()
    for ms in ('train', 'valid', 'test'):
      (allowed_classes if ms == meta_split else forbidden_classes).update(
          dataset_spec.get_classes(getattr(learning_spec.Split, ms.upper())))

    # For MD-v2's ilsvrc_2012 source, train classes are sorted by class name,
    # whereas the TFDS implementation intentionally keeps the v1 class order.
    if remap_labels:
      # The first argsort tells us where to look in class_names for position i
      # in the sorted class list. The second argsort reverses that: it tells us,
      # for position j in class_names, where to place that class in the sorted
      # class list.
      class_names = tfds.builder(
          'meta_dataset/ilsvrc_2012',
          data_dir=FLAGS.tfds_path
      ).info.metadata['class_names']
      label_map = np.argsort(np.argsort(class_names))
      label_remap_fn = np.vectorize(lambda x: label_map[x])

    total_images = sum(dataset_spec.get_total_images_per_class(l)
                       for l in allowed_classes)
    shuffle_buffer_size = min(10_000, total_images)
    batch_size = min(1024, total_images)
    dataset = api.full_ways_dataset(
        md_source=source,
        md_version=md_version,
        meta_split=meta_split,
        decoders={'image': tfds.decode.SkipDecoding()},
        data_dir=FLAGS.tfds_path,
    ).shuffle(
        shuffle_buffer_size
    ).batch(batch_size, drop_remainder=False).prefetch(1)

    entropies = []
    for batch in dataset.as_numpy_iterator():
      labels = batch['label']
      if remap_labels:
        labels = label_remap_fn(labels)

      # With the exception of the remainder batch, we expect classes to be
      # distributed more or less uniformly within batches.
      if len(labels) == batch_size:
        label_counts = collections.Counter(labels)
        p = np.array(
            [label_counts[l] for l in allowed_classes], dtype='float32'
        ) / batch_size
        entropies.append(-np.nan_to_num(p * np.log(p)).sum())

      label_set = set(labels)
      self.assertContainsSubset(label_set, allowed_classes)
      self.assertNoCommonElements(label_set, forbidden_classes)

    # We are happy if at least 75% of the (non-remainder) batches have a
    # class label entropy is at least 50% of the maximum value possible (which
    # corresponds to a uniform distribution).
    self.assertGreater(
        (np.array(entropies) > 0.5 * np.log(len(allowed_classes))).mean(),
        0.75)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  absltest.main()
