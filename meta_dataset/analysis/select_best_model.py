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
r"""A script for choosing the best variant of a model automatically.

It takes as input the root directory of all experiments, and a list of names of
directories in that root, each storing the data of an experiment with multiple
variants accross which we want to select the best. Each experiment directory
should contain a directoy named 'summaries' that hosts subdirectories for the
different runs with each one containing event files. These event files are read
to figure out which is best in terms of mean validation accuracy, and at which
step of that run this best value occurs in.

For each of the experiment directories provided, the output information is saved
in a 'best.pklz' file in that directory. This file contains a dict with keys
'best_variant', 'best_valid_acc', and 'best_update_num' where the name of the
variant is simply the name of the sub-directory corresponding to that variant.

Example directory structure (after the script is ran):
Root contains: 'Exp1', 'Exp2'.
  Exp1 contains: 'checkpoints', 'summaries', and best.pklz
    summaries contains: '1', '2', '3', ..., '20'
      '1' contains event files
      '2' contains event files
      ...
      '20' contains event files

Sample command:
# pylint: disable=line-too-long
python -m meta_dataset.analysis.select_best_model \
  --alsologtostderr \
  --all_experiments_root=<experiments_root> \
  --experiment_dir_basenames=baseline_imagenet_icml2019_1/3602170,baselinefinetune_imagenet_icml2019_1/3581340
# pylint: enable=line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import logging
import numpy as np
from six.moves import range
from six.moves import zip
import six.moves.cPickle as pkl
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'all_experiments_root',
    '',
    'The overall experiments directory root.')

tf.flags.DEFINE_string(
    'experiment_dir_basenames', ''
    'baseline_imagenet_icml2019_1/3602170,'
    'baselinefinetune_imagenet_icml2019_1/3581340',
    'A comma-separated list of directory basenames. Adding each basename as a '
    'suffix to FLAGS.all_experiments_root forms a path that stores the data of '
    'an experiment with multiple variants accross which we want to select the '
    'best. Each such path is expected to host a directory named "summaries" '
    'that contains subdirectories for the different runs with each such '
    'subdirectory containing event files.')

# TODO(etriantafillou): This assumes the variants to omit are the same for all
# experiments that model selection will be ran for which doesn't make much
# sense. Maybe just remove this altogether?
tf.flags.DEFINE_string(
    'restrict_to_variants', '', 'A comma-separated list of '
    'variants to restrict to for model selection. This is '
    'useful for example for finding the best out of all '
    'variants that use a specific embedding or image size.')

tf.flags.DEFINE_string(
    'restrict_to_variants_by_range', '', 'A comma-separated list of '
    'two integers that represent the start and end range (both inclusive) '
    'of variant ids to restrict to.')

tf.flags.DEFINE_string(
    'description', 'best', 'The description for the output. The output will '
    'then be named as description.pklz and description.txt. For example, this '
    'can be used to reflect that some variants were omitted.')

# The following two flags assume that the parameters of the experiments have
# been logged (they attempt to read from them). If this is not the case, the
# restrict_to_variants flag should be used instead.
tf.flags.DEFINE_string(
    'restrict_to_architectures', '', 'The comma-separated names of the '
    'embedding networks to restrict to for model selection.')

tf.flags.DEFINE_enum(
    'restrict_to_pretrained_source', '', ['', 'scratch', 'imagenet'],
    'The name of a  pretrained_source to '
    'restrict to for model selection.')


def get_value_from_params_dir(params_dir, param_name):
  """Gets the value for param_name in the params file in params_dir."""

  def _load_params(params_file, loader, mode):
    with tf.io.gfile.GFile(params_file, mode) as f:
      params = loader(f)
    logging.info('Found params file %s', params_file)
    return params[param_name]

  try:
    try:
      return _load_params(
          os.path.join(params_dir, 'params.json'), json.load, 'r')
    except tf.errors.NotFoundError:
      logging.info('%s does not exist in %s', 'params.json', params_dir)

    try:
      return _load_params(
          os.path.join(params_dir, 'params.pkl'), pkl.load, 'rb')
    except tf.errors.NotFoundError:
      logging.info('%s does not exist in %s', 'params.pkl', params_dir)

  except KeyError:
    logging.info('The params file does not have the key %s', param_name)
    return None


def get_paths_to_events(root_dir,
                        restrict_to_architectures,
                        restrict_to_pretrained_source,
                        restrict_to_variants=None):
  """Returns a dict that maps each variant name to its event file.

  The name of the variant is the basename of the directory where it's stored.
  Assumes the following directory organization root_dir contains a sub-directory
  for every variant where event files can be found.

  There may be more than one event file for each variant, e.g. a new one will be
  created upon restarting an experiment that was pre-empted. So later event
  files contain the summaries for larger values of 'step'. We need all of them
  for determining the global 'best'.

  Args:
    root_dir: A str. The root directory of experiments of all models variants.
    restrict_to_architectures: A list of names of architectures to restrict to
      when choosing the best variant.
    restrict_to_pretrained_source: A string. The pretrained_source to restrict
      to when choosing the best variant.
    restrict_to_variants: Optionally, a set of variant names to restrict to.
  """
  params_dir = os.path.join(root_dir, 'params')
  summary_dir = os.path.join(root_dir, 'summaries')

  def get_variant_architecture(name):
    """Get the architecture of the given variant if it's recorded, o/w None."""
    variant_params_dir = os.path.join(params_dir, name)
    return get_value_from_params_dir(variant_params_dir,
                                     '_gin.LearnerConfig.embedding_network')

  def get_variant_pretrained_source(name):
    variant_params_dir = os.path.join(params_dir, name)
    return get_value_from_params_dir(variant_params_dir,
                                     '_gin.LearnerConfig.pretrained_source')

  def keep_variant(name):
    """Determine if the variant in directory name should be considered."""
    value_error_msg = (
        'Requested to restrict to an architecture or '
        'pretrained_source but the given experiment does not '
        'have its params recorded. Looked in: {}'.format(params_dir))

    if restrict_to_architectures:
      architecture = get_variant_architecture(name)
      if architecture is None:
        raise ValueError(value_error_msg)
    valid_architecture = (not restrict_to_architectures or
                          architecture in restrict_to_architectures)

    if restrict_to_pretrained_source:
      pretrained_source = get_variant_pretrained_source(name)
      if pretrained_source is None:
        raise ValueError(value_error_msg)
    valid_pretrained_source = (
        not restrict_to_pretrained_source or
        pretrained_source == restrict_to_pretrained_source)

    valid_variant_name = True
    if restrict_to_variants is not None:
      valid_variant_name = name in restrict_to_variants

    return (valid_architecture and valid_pretrained_source and
            valid_variant_name)

  variant_names = [
      fname for fname in tf.io.gfile.listdir(summary_dir)
      if tf.io.gfile.isdir(os.path.join(summary_dir, fname))
  ]

  if not variant_names:
    # Maybe there are no variants, and we are already in the directory that
    # contains the summaries. In this case, we consider that the current
    # directory (.) is the only variant.
    variant_names = ['.']

  # Further filter variant names based on the given restrictions.
  variant_names = [name for name in variant_names if keep_variant(name)]

  if not variant_names:
    raise ValueError('Found no subdirectories in {}. Was expecting a '
                     'subdirectory per variant.'.format(summary_dir))
  variant_paths = [
      os.path.join(summary_dir, variant_dir) for variant_dir in variant_names
  ]

  event_paths = {}
  for variant_path, variant_name in zip(variant_paths, variant_names):
    event_filenames = [
        f_name for f_name in tf.io.gfile.listdir(variant_path)
        if f_name.startswith('events.out.tfevents')
    ]

    if len(event_filenames) < 1:
      logging.warn('Skipping empty variant %s.', variant_path)
      logging.info(
          'Was expecting at least one event file '
          'in directory %s. Instead, found %d.', variant_path,
          len(event_filenames))
      continue
    event_paths[variant_name] = [
        os.path.join(variant_path, event_filename)
        for event_filename in event_filenames
    ]

  logging.info('Found event files for variants: %s', list(event_paths.keys()))
  return event_paths


def extract_best_from_event_file(event_path, log_details=False):
  """Returns the best accuracy and the step it occurs in in the given events.

  This searches the summaries written in a given event file, which may be only a
  subset of the total summaries of a run, since the summaries of a run are
  sometimes split into multiple event files.

  Args:
    event_path: A string. The path to an event file.
    log_details: A boolean. Whether to log details regarding skipped event paths
      in which locating the tag "mean valid acc" failed.
  """
  steps, valid_accs = [], []
  try:
    for event in tf.train.summary_iterator(event_path):
      step = event.step
      for value in event.summary.value:
        if value.tag == 'mean valid acc':
          steps.append(step)
          valid_accs.append(value.simple_value)
  except tf.errors.DataLossError:
    if log_details:
      tf.logging.info(
          'Omitting events from event_path {} because '
          'tf.train.summary_iterator(event_path) failed.'.format(event_path))
    return 0, 0
  if not valid_accs:
    # Could happen if there is no DataLossError above but for some reason
    # there is no 'mean valid acc' tag found in the summary values.
    tf.logging.info(
        'Did not find any "mean valid acc" tags in event_path {}'.format(
            event_path))
    return 0, 0
  argmax_ind = np.argmax(valid_accs)
  best_acc = valid_accs[argmax_ind]
  best_step = steps[argmax_ind]
  if log_details:
    tf.logging.info('Successfully read event_path {} with best_acc {}'.format(
        event_path, best_acc))
  return best_acc, best_step


def extract_best_from_variant(event_paths):
  """Returns the best accuracy and the step it occurs in for the given run.

  Args:
    event_paths: A list of strings. The event files of the given run.

  Raises:
    RuntimeError: No 'valid' event file for the given variant ('valid' here
      refers to an event file that has a "mean valid acc" tag).
  """
  best_acc = -1
  for event_path in event_paths:
    best_acc_, best_step_ = extract_best_from_event_file(event_path)
    if best_acc_ > best_acc:
      best_acc = best_acc_
      best_step = best_step_
  if best_acc <= 0:
    raise RuntimeError('Something went wrong with the summary event reading.')
  return best_acc, best_step


def main(argv):
  del argv
  experiment_paths = [
      os.path.join(FLAGS.all_experiments_root, basename)
      for basename in FLAGS.experiment_dir_basenames.split(',')
  ]
  # Perform model selection for each provided experiment root.
  for root_experiment_dir in experiment_paths:
    stars_string = '**************************************\n'
    architecture_string = ''
    if FLAGS.restrict_to_architectures:
      architecture_string = ' out of the {} variants'.format(
          FLAGS.restrict_to_architectures)
    logging.info('%sSelecting the best variant for: %s%s.%s', stars_string,
                 root_experiment_dir, architecture_string, stars_string)

    if FLAGS.restrict_to_variants_by_range and FLAGS.restrict_to_variants:
      raise ValueError('Please provide only one of '
                       'FLAGS.restrict_to_variants_by_range and '
                       'FLAGS.restrict_to_variants, not both.')

    restrict_to_variants = None
    if FLAGS.restrict_to_variants_by_range:
      start, end = FLAGS.restrict_to_variants_by_range.split(',')
      start, end = int(start), int(end)
      restrict_to_variants = set(
          [str(variant_id) for variant_id in range(start, end + 1)])
    if FLAGS.restrict_to_variants:
      restrict_to_variants = set(FLAGS.restrict_to_variants.split(','))

    restrict_to_architectures = []
    if FLAGS.restrict_to_architectures:
      restrict_to_architectures = FLAGS.restrict_to_architectures.split(',')

    event_paths = get_paths_to_events(
        root_experiment_dir,
        restrict_to_architectures,
        FLAGS.restrict_to_pretrained_source,
        restrict_to_variants=restrict_to_variants)
    # Read the event file of each variant to find the highest mean validation
    # accuracy reached with it.
    best_variant = ''
    best_valid_acc = -1
    best_step = -1
    for variant_name, event_path in event_paths.items():
      best_valid_acc_, best_step_ = extract_best_from_variant(event_path)
      if best_valid_acc_ > best_valid_acc:
        best_variant = variant_name
        best_valid_acc = best_valid_acc_
        best_step = best_step_

    output_dict = {
        'best_variant': best_variant,
        'best_valid_acc': best_valid_acc,
        'best_update_num': best_step
    }

    # Create a more informative description if necessary.
    description = FLAGS.description
    if FLAGS.restrict_to_architectures and FLAGS.description == 'best':
      description += '_{}'.format(FLAGS.restrict_to_architectures)

    if (FLAGS.restrict_to_pretrained_source and FLAGS.description == 'best'):
      if FLAGS.restrict_to_pretrained_source == 'scratch':
        description += '_trained_from_scratch'
      else:
        description += '_pretrained_on_{}'.format(
            FLAGS.restrict_to_pretrained_source)

    output_path_pklz = os.path.join(root_experiment_dir,
                                    '{}.pklz'.format(description))
    with tf.io.gfile.GFile(output_path_pklz, 'wb') as f:
      pkl.dump(output_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Also write this info as a .txt file for easier reading.
    output_path_txt = os.path.join(root_experiment_dir,
                                   '{}.txt'.format(description))
    with tf.io.gfile.GFile(output_path_txt, 'w') as f:
      f.write(
          'best_variant: {}\nbest_valid_acc: {}\nbest_update_num: {}\n'.format(
              best_variant, best_valid_acc, best_step))
    logging.info(
        'Best variant: %s. Best valid acc: %s. Best update num: %d. '
        'Just wrote this info to %s and %s', best_variant, best_valid_acc,
        best_step, output_path_pklz, output_path_txt)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
