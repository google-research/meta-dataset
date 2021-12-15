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

# Lint as: python2, python3
r"""Script for training / evaluating FLUTE.

# pylint: disable=line-too-long
Launching command for training FLUTE:
python -m meta_dataset.train_flute \
  --train_checkpoint_dir=/tmp/bench \
  --summary_dir=/tmp/bench \
  --records_root_dir=<records_root> \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/default/flute.gin \
  --gin_bindings="Trainer_flute.experiment_name='flute'"

Launching command for training FLUTE's dataset classifier (used in the 'Blender
network'):
python -m meta_dataset.train_flute \
  --train_checkpoint_dir=/tmp/bench \
  --summary_dir=/tmp/bench \
  --records_root_dir=<records_root> \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/default/flute_dataset_classifier.gin
  \
  --gin_bindings="Trainer_flute.experiment_name='flute_dataset_classifier'"

Testing FLUTE:
python -m meta_dataset.train_flute \
  --is_training=False \
  --records_root_dir=<records_root> \
  --summary_dir=/tmp/bench \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/best/flute.gin \
  --gin_bindings="Trainer_flute.experiment_name='flute'" \
  --gin_bindings="Trainer_flute.checkpoint_to_restore='<FLUTE_CKPT>'" \
  --gin_bindings="Trainer_flute.dataset_classifier_to_restore='<DATAST_CLASSIFIER_CKPT>'"
  \
  --gin_bindings="benchmark.eval_datasets='<DATASET>'"
# pylint: enable=line-too-long

where <DATASET> is the dataset to run evaluation on, and <FLUTE_CKPT> and
<DATAST_CLASSIFIER_CKPT> are the checkpoints of the trained FLUTE and dataset
classifier models. These can be either the checkpoints produced via the first 2
commands, or the trained checkpoints that we have released (see the README to
obtain those).

To override elements from the config, you can use arguments of the form:
  For gin: --gin_bindings='foo = 1000000'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import logging
import gin.tf
# TODO(lamblinp): Better organize module imports for exposure of Gin
# configurables.
from meta_dataset import data  # pylint: disable=unused-import
from meta_dataset import learners  # pylint: disable=unused-import
from meta_dataset import train
from meta_dataset import trainer
from meta_dataset import trainer_flute
from meta_dataset.data import config  # pylint: disable=unused-import
from meta_dataset.data import sur_decoder  # pylint: disable=unused-import
from meta_dataset.learners import experimental as experimental_learners  # pylint: disable=unused-import
from meta_dataset.models.experimental import parameter_adapter  # pylint: disable=unused-import

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

DEFAULT_SAVING_DIRECTORY = '/tmp/metadataset'

tf.flags.DEFINE_integer('num_gpus', 1, 'The number of available gpus.')
FLAGS = tf.flags.FLAGS


def main(unused_argv):

  # Parse Gin configurations passed to this script.
  train.parse_cmdline_gin_configurations()

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
      train.load_operative_gin_configurations(restore_checkpoint_dir)

      # Reload the command-line Gin configuration to allow overriding of the Gin
      # configuration loaded from the checkpoint directory.
      train.parse_cmdline_gin_configurations()

  # Wrap object instantiations to print out full Gin configuration on failure.
  try:
    (train_datasets, eval_datasets, restrict_classes,
     restrict_num_per_class) = trainer.get_datasets_and_restrictions()

    # Get a trainer or evaluator.
    trainer_instance = trainer_flute.Trainer(
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
        num_gpus=FLAGS.num_gpus)
  except ValueError as e:
    logging.info('Full Gin configurations:\n%s', gin.config_str())
    raise e

  # All configurable objects/functions should have been instantiated/called.
  # TODO(evcu): Tie saving of Gin configuration at training and evaluation time.
  logging.info('Operative Gin configurations:\n%s', gin.operative_config_str())
  if FLAGS.is_training and FLAGS.train_checkpoint_dir:
    train.record_operative_gin_configurations(FLAGS.train_checkpoint_dir)
  elif not FLAGS.is_training and FLAGS.summary_dir:
    train.record_operative_gin_configurations(FLAGS.summary_dir)

  datasets = train_datasets if FLAGS.is_training else eval_datasets
  logging.info('Starting %s for dataset(s) %s...',
               'training' if FLAGS.is_training else 'evaluation', datasets)
  if FLAGS.is_training:
    trainer_instance.train()
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
