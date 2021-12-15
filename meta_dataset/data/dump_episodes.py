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

# Lint as: python3
r"""Dumps Meta-Dataset episodes to disk as tfrecords files.

Episodes are stored as a pair of `{episode_number}-train.tfrecords` and
`{episode_number}-test.tfrecords` files, each of which contains serialized
TFExample strings for the support and query set, respectively.

python -m meta_dataset.data.dump_episodes \
--gin_config=meta_dataset/learn/gin/setups/\
data_config_string.gin --gin_config=meta_dataset/learn/gin/\
setups/variable_way_and_shot.gin \
--gin_bindings="DataConfig.num_prefetch=<num_prefetch>"
"""
import json
import os
from absl import app
from absl import flags
from absl import logging

import gin
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import utils
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

flags.DEFINE_multi_string('gin_config', None,
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', None,
                          'List of Gin parameter bindings.')
flags.DEFINE_string('output_dir', '/tmp/cached_episodes/',
                    'Root directory for saving episodes.')
flags.DEFINE_integer('num_episodes', 600, 'Number of episodes to sample.')
flags.DEFINE_string('dataset_name', 'omniglot', 'Dataset name to create '
                    'episodes from.')
flags.DEFINE_enum_class('split', learning_spec.Split.TEST, learning_spec.Split,
                        'See learning_spec.Split for '
                        'allowed values.')
flags.DEFINE_boolean(
    'ignore_dag_ontology', False, 'If True the dag ontology'
    ' for Imagenet dataset is not used.')
flags.DEFINE_boolean(
    'ignore_bilevel_ontology', False, 'If True the bilevel'
    ' sampling for Omniglot dataset is not used.')
tf.flags.DEFINE_string('records_root_dir', '',
                       'Root directory containing a subdirectory per dataset.')
FLAGS = flags.FLAGS


def main(unused_argv):
  logging.info(FLAGS.output_dir)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  gin.parse_config_files_and_bindings(
      FLAGS.gin_config, FLAGS.gin_bindings, finalize_config=True)
  dataset_spec = dataset_spec_lib.load_dataset_spec(
      os.path.join(FLAGS.records_root_dir, FLAGS.dataset_name))
  data_config = config.DataConfig()
  episode_descr_config = config.EpisodeDescriptionConfig()
  use_dag_ontology = (
      FLAGS.dataset_name in ('ilsvrc_2012', 'ilsvrc_2012_v2') and
      not FLAGS.ignore_dag_ontology)
  use_bilevel_ontology = (
      FLAGS.dataset_name == 'omniglot' and not FLAGS.ignore_bilevel_ontology)
  data_pipeline = pipeline.make_one_source_episode_pipeline(
      dataset_spec,
      use_dag_ontology=use_dag_ontology,
      use_bilevel_ontology=use_bilevel_ontology,
      split=FLAGS.split,
      episode_descr_config=episode_descr_config,
      # TODO(evcu) Maybe set the following to 0 to prevent shuffling and check
      # reproducibility of dumping.
      shuffle_buffer_size=data_config.shuffle_buffer_size,
      read_buffer_size_bytes=data_config.read_buffer_size_bytes,
      num_prefetch=data_config.num_prefetch)
  dataset = data_pipeline.take(FLAGS.num_episodes)

  images_per_class_dict = {}
  # Ignoring dataset number since we are loading one dataset.
  for episode_number, (episode, _) in enumerate(dataset):
    logging.info('Dumping episode %d', episode_number)
    train_imgs, train_labels, _, test_imgs, test_labels, _ = episode
    path_train = utils.get_file_path(FLAGS.output_dir, episode_number, 'train')
    path_test = utils.get_file_path(FLAGS.output_dir, episode_number, 'test')
    utils.dump_as_tfrecord(path_train, train_imgs, train_labels)
    utils.dump_as_tfrecord(path_test, test_imgs, test_labels)
    images_per_class_dict[os.path.basename(path_train)] = (
        utils.get_label_counts(train_labels))
    images_per_class_dict[os.path.basename(path_test)] = (
        utils.get_label_counts(test_labels))
  info_path = utils.get_info_path(FLAGS.output_dir)
  with tf.io.gfile.GFile(info_path, 'w') as f:
    f.write(json.dumps(images_per_class_dict, indent=2))


if __name__ == '__main__':
  app.run(main)
