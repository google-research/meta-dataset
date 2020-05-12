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
r"""Script for training models on the benchmark.

Launching command for batch baseline:
# pylint: disable=line-too-long
python -m meta_dataset.train \
  --train_checkpoint_dir=/tmp/bench --summary_dir=/tmp/bench \
  --records_root_dir=<records_root> \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/default/<exp_name>.gin
  --gin_bindings="Trainer.experiment_name='<exp_name>'"
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

from absl import app
from absl import logging
import gin
import gin.tf
from meta_dataset import data
from meta_dataset import trainer
from meta_dataset.data import config  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf

DEFAULT_SAVING_DIRECTORY = '/tmp/metadataset'

tf.flags.DEFINE_string('train_checkpoint_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'checkpoints'),
                       'The directory to save checkpoints.')
tf.flags.DEFINE_string('summary_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'summaries'),
                       'The directory for writing summaries.')
tf.flags.DEFINE_bool(
    'reload_checkpoint_gin_config', False,
    'Whether to reload an operative Gin configuration along with a checkpoint.')
tf.flags.DEFINE_string('records_root_dir', '',
                       'Root directory containing a subdirectory per dataset.')
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

# TODO(crisnv): eval_split is supposed to substitute eval_finegrainedness and
# eval_finegrainedness_split in the future
tf.flags.DEFINE_enum(
    'eval_split', None,
    [trainer.TRAIN_SPLIT, trainer.VALID_SPLIT, trainer.TEST_SPLIT],
    'Override the evaluation split. If None (default), regular logic is used, '
    'that is, if "is_training" is True, "trainer.VALID_SPLIT" is used, '
    'otherwise "trainer.TEST_SPLIT" is used. The "is_training" case also uses '
    'the value of eval_finegrainedness_split if eval_finegrainedness is True.')

tf.flags.DEFINE_bool(
    'eval_finegrainedness', False, 'Whether to perform only 2-way ImageNet '
    'evaluation for assessing performance as a function of how finegrained '
    'each task is. This differs from usual ImageNet eval in the sampling '
    'procedure used to get episodes, and therefore requires its own setting.')

tf.flags.DEFINE_enum(
    'eval_finegrainedness_split', trainer.TRAIN_SPLIT,
    [trainer.TRAIN_SPLIT, trainer.VALID_SPLIT, trainer.TEST_SPLIT], 'The '
    'split whose results we want to use for the fine-grainedness analysis.'
    'Contrary to most analyses which are performed on the test split only, the '
    'fine-grainedness analysis may also be performed on the train or valid '
    'sub-graphs of ImageNet too, since the test sub-graph evidently does not '
    'exhibit enough variation in the fine-grainedness of its different tasks '
    'to allow for a meaningful analysis.')
# The following flag specifies substrings of variable names that should not be
# reloaded. `num_left_in_epoch' is a variable that influences the behavior of
# the EpochTrackers. Since the state of those trackers is not reloaded, neither
# should this variable. `fc_finetune' is a substring of the names of the
# variables in the episode-specific linear layer of the finetune baseline (used
# at meta-validation and meta-test times). Since this layer gets re-initialized
# to random weights in each new episode, there is no need to ever restore these
# weights. `linear_classifier' plays that role but for the MAML model: similarly
# in each new episode it is re-initialized (e.g. set to zeros or to the
# prototypes in the case of proto-MAML), so there is no need to restore these
# weights. `adam_opt' captures the variables of the within-episode optimizer of
# the finetune baseline when it is configured to perform that finetuning with
# adam. `fc' captures the variable names of the fully-connected layer for the
# all-way classification problem that the baselines solve at training time.
# There are certain situations where we need to omit reloading these weights to
# avoid getting an error. Consider for example the experiments where we train
# a baseline model, starting from weights that were previously trained on
# ImageNet. If this training now takes place on all datasets, the size of the
# all-way classification layer is now different (equal to the number of
# meta-training classes of all datasets not just of ImageNet). Thus when
# training baselines from pre-trained weights, we only reload the backbone and
# not the `fc' all-way classification layer (similarly for inference-only
# experiments for the same reason).
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


def parse_cmdline_gin_configurations():
  """Parse Gin configurations from all command-line sources."""
  with gin.unlock_config():
    gin.parse_config_files_and_bindings(
        FLAGS.gin_config, FLAGS.gin_bindings, finalize_config=True)


def operative_config_path(operative_config_dir,
                          operative_config_filename='operative_config.gin'):
  return os.path.join(operative_config_dir, operative_config_filename)


def load_operative_gin_configurations(operative_config_dir):
  """Load operative Gin configurations from the given directory."""
  gin_log_file = operative_config_path(operative_config_dir)
  with gin.unlock_config():
    gin.parse_config_file(gin_log_file)
  gin.finalize()
  logging.info('Operative Gin configurations loaded from %s.', gin_log_file)


def record_operative_gin_configurations(operative_config_dir):
  """Record operative Gin configurations in the given directory."""
  gin_log_file = operative_config_path(operative_config_dir)
  # If it exists already, rename it instead of overwriting it.
  # This just saves the previous one, not all the ones before.
  if tf.io.gfile.exists(gin_log_file):
    tf.io.gfile.rename(gin_log_file, gin_log_file + '.old', overwrite=True)
  with tf.io.gfile.GFile(gin_log_file, 'w') as f:
    f.write(gin.operative_config_str())


def main(unused_argv):

  # Parse Gin configurations passed to this script.
  parse_cmdline_gin_configurations()

  if FLAGS.reload_checkpoint_gin_config:
    # Try to reload a previously recorded Gin configuration from an operative
    # Gin configuration file in one of the provided directories.
    # TODO(eringrant): Allow querying of a value to be bound without binding it
    # to avoid the redundant call to `parse_cmdline_gin_configurations` below.
    try:
      checkpoint_to_restore = gin.query_parameter(
          'Trainer.checkpoint_to_restore')
    except ValueError:
      checkpoint_to_restore = None

    # Load the operative Gin configurations from the checkpoint directory.
    if checkpoint_to_restore:
      restore_checkpoint_dir = os.path.dirname(checkpoint_to_restore)
      load_operative_gin_configurations(restore_checkpoint_dir)

      # Reload the command-line Gin configuration to allow overriding of the Gin
      # configuration loaded from the checkpoint directory.
      parse_cmdline_gin_configurations()

  # Wrap object instantiations to print out full Gin configuration on failure.
  try:
    (train_datasets, eval_datasets, restrict_classes,
     restrict_num_per_class) = trainer.get_datasets_and_restrictions()

    # Get a trainer or evaluator.
    trainer_instance = trainer.Trainer(
        is_training=FLAGS.is_training,
        train_dataset_list=train_datasets,
        eval_dataset_list=eval_datasets,
        restrict_classes=restrict_classes,
        restrict_num_per_class=restrict_num_per_class,
        checkpoint_dir=FLAGS.train_checkpoint_dir,
        summary_dir=FLAGS.summary_dir,
        records_root_dir=FLAGS.records_root_dir,
        eval_finegrainedness=FLAGS.eval_finegrainedness,
        eval_finegrainedness_split=FLAGS.eval_finegrainedness_split,
        eval_imbalance_dataset=FLAGS.eval_imbalance_dataset,
        omit_from_saving_and_reloading=FLAGS.omit_from_saving_and_reloading,
        eval_split=FLAGS.eval_split,
    )
  except ValueError as e:
    logging.info('Full Gin configurations:\n%s', gin.config_str())
    raise e

  # All configurable objects/functions should have been instantiated/called.
  # TODO(evcu): Tie saving of Gin configuration at training and evaluation time.
  logging.info('Operative Gin configurations:\n%s', gin.operative_config_str())
  if FLAGS.is_training and FLAGS.train_checkpoint_dir:
    record_operative_gin_configurations(FLAGS.train_checkpoint_dir)
  elif not FLAGS.is_training and FLAGS.summary_dir:
    record_operative_gin_configurations(FLAGS.summary_dir)

  datasets = train_datasets if FLAGS.is_training else eval_datasets
  logging.info('Starting %s for dataset(s) %s...',
               'training' if FLAGS.is_training else 'evaluation', datasets)
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

    if FLAGS.eval_finegrainedness:
      eval_split = FLAGS.eval_finegrainedness_split
    elif FLAGS.eval_split:
      eval_split = FLAGS.eval_split
    else:
      eval_split = trainer.TEST_SPLIT

    _, _, acc_summary, ci_acc_summary = trainer_instance.evaluate(eval_split)
    if trainer_instance.summary_writer:
      trainer_instance.summary_writer.add_summary(acc_summary)
      trainer_instance.summary_writer.add_summary(ci_acc_summary)

  # Flushes the event file to disk and closes the file.
  if trainer_instance.summary_writer:
    trainer_instance.summary_writer.close()


program = main


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(program)
