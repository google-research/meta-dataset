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

import os

import gin.tf
from meta_dataset import data
from meta_dataset import learner
from meta_dataset import trainer
import tensorflow as tf

tf.flags.DEFINE_string('train_checkpoint_dir', '/tmp/metadataset',
                       'The directory to save checkpoints.')

tf.flags.DEFINE_string('summary_dir', '/tmp/metadataset',
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

FLAGS = tf.flags.FLAGS

NAME_TO_LEARNER = {
    'Baseline': learner.BaselineLearner,
    'BaselineFinetune': learner.BaselineFinetuneLearner,
    'MatchingNet': learner.MatchingNetworkLearner,
    'PrototypicalNet': learner.PrototypicalNetworkLearner,
    'MAML': learner.MAMLLearner,
}

BATCH_LEARNERS = ['Baseline', 'BaselineFinetune']
EPISODIC_LEARNERS = ['MatchingNet', 'PrototypicalNet', 'MAML']


@gin.configurable('benchmark')
def get_datasets(train_datasets='', eval_datasets=''):
  """Gets the lists of dataset names.

  Args:
    train_datasets: A string of comma separated dataset names for training.
    eval_datasets: A string of comma separated dataset names for evaluation.

  Returns:
    Two lists of dataset names, to be used for training and validation, resp.
    The second one might be empty.
  """
  return [d.strip() for d in train_datasets.split(',')
         ], [d.strip() for d in eval_datasets.split(',')]


def main(unused_argv):
  tf.logging.info('FLAGS.gin_config: %s', FLAGS.gin_config)
  tf.logging.info('FLAGS.gin_bindings: %s', FLAGS.gin_bindings)
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

  train_datasets, eval_datasets = get_datasets()

  train_learner = None
  if FLAGS.is_training or (FLAGS.eval_finegrainedness and
                           FLAGS.eval_finegrainedness_split == 'train'):
    # If eval_finegrainedness is True, even in pure evaluation mode we still
    # require a train learner, since we may perform this analysis on the
    # training sub-graph of ImageNet too.
    train_learner = NAME_TO_LEARNER[learner_config.train_learner]
  eval_learner = NAME_TO_LEARNER[learner_config.eval_learner]

  # Get a trainer or evaluator.
  if learner_config.episodic:
    trainer_instance = trainer.EpisodicTrainer(
        train_learner, eval_learner, FLAGS.is_training, train_datasets,
        eval_datasets, FLAGS.train_checkpoint_dir, FLAGS.summary_dir,
        FLAGS.eval_finegrainedness, FLAGS.eval_finegrainedness_split,
        FLAGS.eval_imbalance_dataset)
    if learner_config.train_learner not in EPISODIC_LEARNERS:
      raise ValueError(
          'When "episodic" is True, "train_learner" should be an episodic one, '
          'among {}.'.format(EPISODIC_LEARNERS))
  else:
    trainer_instance = trainer.BatchTrainer(
        train_learner, eval_learner, FLAGS.is_training, train_datasets,
        eval_datasets, FLAGS.train_checkpoint_dir, FLAGS.summary_dir,
        FLAGS.eval_finegrainedness, FLAGS.eval_finegrainedness_split,
        FLAGS.eval_imbalance_dataset)
    if learner_config.train_learner not in BATCH_LEARNERS:
      raise ValueError(
          'When "episodic" is False, "train_learner" should be a batch one, '
          'among {}.'.format(BATCH_LEARNERS))

  mode = 'training' if FLAGS.is_training else 'evaluation'
  datasets = train_datasets if FLAGS.is_training else eval_datasets
  tf.logging.info('Starting %s for dataset(s) %s...' % (mode, datasets))

  # Record gin operative config string after the setup, both in the logs and in
  # the checkpoint directory.
  gin_operative_config = gin.operative_config_str()
  tf.logging.info('gin configuration:\n%s', gin_operative_config)
  if FLAGS.train_checkpoint_dir:
    gin_log_file = os.path.join(FLAGS.train_checkpoint_dir,
                                'operative_config.gin')
    # If it exists already, rename it instead of overwriting it.
    # This just saves the previous one, not all the ones before.
    if tf.gfile.Exists(gin_log_file):
      tf.gfile.Rename(gin_log_file, gin_log_file + '.old', overwrite=True)
    with tf.gfile.Open(gin_log_file, 'w') as f:
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
  trainer_instance.summary_writer.close()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
