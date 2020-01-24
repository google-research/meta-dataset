# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
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
r"""Script for training models on the benchmark.

Launching command for batch baseline:
# pylint: disable=line-too-long
python -m meta_dataset.train \
  --train_checkpoint_dir=/tmp/bench --summary_dir=/tmp/bench \
  --records_root_dir=<records_root> \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/default/<exp_name>.gin
  --gin_bindings="LearnerConfig.experiment_name='<exp_name>'"
# pylint: enable=line-too-long

where:
  <exp_name> is e.g. 'debug_proto_mini_imagenet'

To override elements from the config, you can use arguments of the form:
  For gin: --gin_bindings='foo = 1000000'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import logging
import gin.tf
from meta_dataset import data
from meta_dataset import trainer
from meta_dataset.data import config  # pylint: disable=unused-import
import tensorflow as tf

DEFAULT_SAVING_DIRECTORY = '/tmp/metadataset'

tf.flags.DEFINE_string('train_checkpoint_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'checkpoints'),
                       'The directory to save checkpoints.')
tf.flags.DEFINE_string('summary_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'summaries'),
                       'The directory for writing summaries.')

tf.flags.DEFINE_bool(
    'is_training', True, 'Whether we are in the training phase. '
    'Used to control whether to perform training or evaluation.')

tf.flags.DEFINE_multi_string('gin_config', None,
                             'List of paths to the config files.')
tf.flags.DEFINE_multi_string('gin_bindings', None,
                             'List of Gin parameter bindings.')

tf.flags.DEFINE_string(
    'eval_imbalance_dataset', '', 'A dataset on which to perform evaluation '
    'for assessing how class imbalance affects performance in binary episodes. '
    'By default it is empty and no imbalance analysis is performed.')

tf.flags.DEFINE_bool(
    'eval_finegrainedness', False, 'Whether to perform only 2-way ImageNet '
    'evaluation for assessing performance as a function of how finegrained '
    'each task is. This differs from usual ImageNet eval in the sampling '
    'procedure used to get episodes, and therefore requires its own setting.')

tf.flags.DEFINE_enum(
    'eval_finegrainedness_split', 'train', ['train', 'valid', 'test'], 'The '
    'split whose results we want to use for the fine-grainedness analysis.'
    'Contrary to most analyses which are performed on the test split only, the '
    'fine-grainedness analysis may also be performed on the train or valid '
    'sub-graphs of ImageNet too, since the test sub-graph evidently does not '
    'exhibit enough variation in the fine-grainedness of its different tasks '
    'to allow for a meaningful analysis.')

tf.flags.DEFINE_multi_enum(
    'omit_from_saving_and_reloading', [
        'num_left_in_epoch', 'fc_finetune', 'linear_classifier', 'adam_opt',
        'weight_copy'
    ], [
        'num_left_in_epoch', 'fc_finetune', 'linear_classifier', 'adam_opt',
        'weight_copy', 'fc'
    ],
    'A comma-separated list of substrings such that all variables containing '
    'them should not be saved and reloaded.')

FLAGS = tf.flags.FLAGS


def main(unused_argv):

  logging.info('FLAGS.gin_config: %s', FLAGS.gin_config)
  logging.info('FLAGS.gin_bindings: %s', FLAGS.gin_bindings)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

  learner_config = trainer.LearnerConfig()

  # Check for inconsistent or contradictory flag settings.
  if (learner_config.checkpoint_for_eval and
      learner_config.pretrained_checkpoint):
    raise ValueError('Can not define both checkpoint_for_eval and '
                     'pretrained_checkpoint. The difference between them is '
                     'that in the former all variables are restored (including '
                     'global step) whereas the latter is only applicable to '
                     'the start of training for initializing the model from '
                     'pre-trained weights. It is also only applicable to '
                     'episodic models and restores only the embedding weights.')


  (train_datasets, eval_datasets, restrict_classes,
   restrict_num_per_class) = trainer.get_datasets_and_restrictions()

  train_learner = None
  if FLAGS.is_training or (FLAGS.eval_finegrainedness and
                           FLAGS.eval_finegrainedness_split == 'train'):
    # If eval_finegrainedness is True, even in pure evaluation mode we still
    # require a train learner, since we may perform this analysis on the
    # training sub-graph of ImageNet too.
    train_learner = trainer.NAME_TO_LEARNER[learner_config.train_learner]
  eval_learner = trainer.NAME_TO_LEARNER[learner_config.eval_learner]

  # Get a trainer or evaluator.
  trainer_kwargs = {
      'train_learner': train_learner,
      'eval_learner': eval_learner,
      'is_training': FLAGS.is_training,
      'train_dataset_list': train_datasets,
      'eval_dataset_list': eval_datasets,
      'restrict_classes': restrict_classes,
      'restrict_num_per_class': restrict_num_per_class,
      'checkpoint_dir': FLAGS.train_checkpoint_dir,
      'summary_dir': FLAGS.summary_dir,
      'records_root_dir': FLAGS.records_root_dir,
      'eval_finegrainedness': FLAGS.eval_finegrainedness,
      'eval_finegrainedness_split': FLAGS.eval_finegrainedness_split,
      'eval_imbalance_dataset': FLAGS.eval_imbalance_dataset,
      'omit_from_saving_and_reloading': FLAGS.omit_from_saving_and_reloading,
  }
  if learner_config.episodic:
    trainer_instance = trainer.EpisodicTrainer(**trainer_kwargs)
    if learner_config.train_learner not in trainer.EPISODIC_LEARNER_NAMES:
      raise ValueError(
          'When "episodic" is True, "train_learner" should be an episodic one, '
          'among {}.'.format(trainer.EPISODIC_LEARNER_NAMES))
  else:
    trainer_instance = trainer.BatchTrainer(**trainer_kwargs)
    if learner_config.train_learner not in trainer.BATCH_LEARNER_NAMES:
      raise ValueError(
          'When "episodic" is False, "train_learner" should be a batch one, '
          'among {}.'.format(trainer.BATCH_LEARNER_NAMES))

  mode = 'training' if FLAGS.is_training else 'evaluation'
  datasets = train_datasets if FLAGS.is_training else eval_datasets
  logging.info('Starting %s for dataset(s) %s...', mode, datasets)

  # Record gin operative config string after the setup, both in the logs and in
  # the checkpoint directory.
  gin_operative_config = gin.operative_config_str()
  logging.info('gin configuration:\n%s', gin_operative_config)
  if FLAGS.train_checkpoint_dir:
    gin_log_file = os.path.join(FLAGS.train_checkpoint_dir,
                                'operative_config.gin')
    # If it exists already, rename it instead of overwriting it.
    # This just saves the previous one, not all the ones before.
    if tf.io.gfile.exists(gin_log_file):
      tf.io.gfile.rename(gin_log_file, gin_log_file + '.old', overwrite=True)
    with tf.io.gfile.GFile(gin_log_file, 'w') as f:
      f.write(gin_operative_config)

  if FLAGS.is_training:
    trainer_instance.train()
  elif set(datasets).intersection(trainer.DATASETS_WITH_EXAMPLE_SPLITS):
    if not data.POOL_SUPPORTED:
      raise NotImplementedError('Example-level splits or pools not supported.')
  else:
    if len(datasets) != 1:
      raise ValueError('Requested datasets {} for evaluation, but evaluation '
                       'should be performed on individual datasets '
                       'only.'.format(datasets))

    eval_split = 'test'
    if FLAGS.eval_finegrainedness:
      eval_split = FLAGS.eval_finegrainedness_split

    trainer_instance.evaluate(eval_split)

  # Flushes the event file to disk and closes the file.
  if trainer_instance.summary_writer:
    trainer_instance.summary_writer.close()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
