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
r"""Script for aggregating the eval summaries into dicts.

This script assumes that the evaluation has already been ran (and has produced
the eval summaries from which this script reads).

Creates four dicts: One maps each observed 'shot' to a list of class precisions
obtained by classes that had that shot (regardless of shots of other classes in
the same episode). The second maps each observed 'way' of an episode to a list
of accuracies of the episodes with that way. The third maps each observed height
(of the lowest common ancestor of pairs of leaves corresponding to the Synsets
of ImageNet binary classification tasks from the training subgraph) to the
accuracy of those tasks, aiming to study how the fine- or coarse- grainedness of
a task affects its difficulty. The fourth maps each observed degree of imbalance
(w.r.t the numbers of shots of the different classes in the task) to the
accuracy of the corresponding episodes.
Summarized versions are also created that keep only the mean and confidence
intervals instead of the list of all precisons or accuracies, resp. as the
values of these dicts.

Sample command:
# pylint: disable=line-too-long
python -m meta_dataset.analyze \
  --alsologtostderr \
  --eval_finegrainedness \
  --eval_finegrainedness_split=test \
  --root_dir=<root_dir> \
# pylint: enable=line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import logging
from meta_dataset.data import dataset_spec
from meta_dataset.data import imagenet_specification as imagenet_spec
from meta_dataset.data import learning_spec
import numpy as np
from six.moves import range
import six.moves.cPickle as pkl
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

# Will search for all event files in this root dir.
tf.flags.DEFINE_string(
    'root_dir',
    '',
    'The root '
    'directory to look in for sub-directory trees containing event files.')

tf.flags.DEFINE_bool(
    'eval_imbalance', False, 'Whether to perform only 2-way evaluation for '
    'assessing performance as a function of how imbalanced each task is.')

tf.flags.DEFINE_bool(
    'eval_finegrainedness', False, 'Whether to perform only 2-way ImageNet '
    'evaluation for assessing performance as a function of how finegrained '
    'each task is. This differs from usual ImageNet eval in the sampling '
    'procedure used to get episodes, and therefore requires its own setting.')

tf.flags.DEFINE_enum(
    'eval_finegrainedness_split', 'test', ['train', 'valid', 'test'], 'The '
    'split whose results we want to use for the fine-grainedness analysis.'
    'Contrary to most analyses which are performed on the test split only, the '
    'fine-grainedness analysis may also be performed on the train or valid '
    'sub-graphs of ImageNet too, since the test sub-graph evidently does not '
    'exhibit enough variation in the fine-grainedness of its different tasks '
    'to allow for a meaningful analysis.')

# To restrict to evaluating on ImageNet, the following should be set to '2'.
# The valid sub-experiment id's start from '1'.
# TODO(etriantafillou): Adapt the following for external users. In particular,
# we shouldn't necessarily assume the directory structure where there is one
# directory per experiment id, which corresponds to different hyperparams.
tf.flags.DEFINE_enum(
    'restrict_to_subexperiment', '0', [str(num) for num in range(11)], 'If '
    'positive, restricts to using the summaries in the sub-experiment whose id '
    'is the given number. This corresponds to a specific hyper (e.g. choice of '
    'evaluation dataset). Valid experiment ids start from "1".')

tf.flags.DEFINE_bool(
    'force_recompute', False, 'Whether to always re-compute (and overwrite) '
    'the dictionaries regardless of whether they have already been computed.')

FLAGS = tf.flags.FLAGS


def compute_class_precision(class_id, logits, targets):
  """Computes the precision for class_id.

  The precision for a class is defined as the number of examples of that class
  that are correctly classified over its total number of examples.

  Args:
    class_id: An int, in the range between 0 and the number of classes.
    logits: A float array, of shape [num_test_examples, num_classes].
    targets: An int array, of the same shape as logits.

  Returns:
    precision: A float. The precision for the given class.
  """
  # Get the section of the logits that correspond to class_id.
  class_logits_ids = np.where(targets == class_id)[0]
  # [# test examples of class_id, way].
  class_logits = logits[class_logits_ids]
  # [# test examples of class_id]
  class_preds = np.argmax(class_logits, axis=1)
  precision = np.mean(np.equal(class_preds, class_id))
  return precision


def compute_episode_accuracy(logits, targets):
  """Computes the accuracy for the episode.

  The accuracy for the episode is the proportion of correctly classified test
  examples from the overall number of test examples.

  Args:
    logits: A float array, of shape [num_test_examples, num_classes].
    targets: An int array, of the same shape as logits.

  Returns:
    accuracy: A float. The precision for the given class.
  """
  preds = np.argmax(logits, axis=1)
  return np.mean(np.equal(preds, targets))


def get_shot_to_precision(shots, logits, targets):
  """Performance of a particular class as a function of its 'shot'.

  Args:
    shots: A list containing a np.array per episode. The shape of an episode's
      array is the [way]. Stores the 'shot' of each class, ie. the number of
      training examples that that class has in the support set.
    logits: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set, way].
    targets: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set]. This contains integers in the range from 0 to
      the way of the episode.

  Returns:
    shot_to_precision: A dict that maps each 'observed' shot to a list that
      stores the precision obtained for that shot (each entry in this list is
      the precision of a particular class that had this shot, regardless of the
      shots of the other classes in the episode.).
  """
  shot_to_precision = collections.defaultdict(list)
  for episode_num, episode_shots in enumerate(shots):
    episode_logits = logits[episode_num]
    episode_targets = targets[episode_num]
    for class_id, class_shot in enumerate(episode_shots):
      class_precision = compute_class_precision(class_id, episode_logits,
                                                episode_targets)
      shot_to_precision[class_shot].append(class_precision)
  return shot_to_precision


def get_imbalance_to_accuracy(class_props, logits, targets):
  """Accuracy as a function of imabalance.

  Args:
    class_props: A list containing a np.array per episode. The shape of an
      episode's array is the [way]. Stores the 'normalized shot' of each class,
      ie. the proportion of the examples of that class that are in the support
      set of the episode.
    logits: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set, way].
    targets: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set]. This contains integers in the range from 0 to
      the way of the episode.

  Returns:
    imbalance_to_accuracy: A dict mapping each observed imbalance (a float) to a
    list that stores the accuracy of the episodes characterized by that degree
    of imbalance.

  Raises:
    ValueError: There should have been exactly 2 elements in the list of each
      episode's class id's.
  """
  imbalance_to_accuracy = collections.defaultdict(list)
  for episode_num, episode_class_props in enumerate(class_props):
    if len(episode_class_props) != 2:
      raise ValueError(
          'There should have been exactly 2 elements in the list '
          "of each episode's class_props (we only perform the "
          'imbalance analysis on binary tasks). Instead, found: {}'.format(
              len(episode_class_props)))
    # Compute imbalance.
    imbalance = max(episode_class_props) - min(episode_class_props)
    # Compute the accuracy of the episode.
    episode_logits = logits[episode_num]
    episode_targets = targets[episode_num]
    episode_acc = compute_episode_accuracy(episode_logits, episode_targets)
    imbalance_to_accuracy[imbalance].append(episode_acc)
  return imbalance_to_accuracy


def get_way_to_accuracy(ways, logits, targets):
  """Accuracy as a function of the episode's way.

  Args:
    ways: A list containing the 'way' of each episode.
    logits: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set, way].
    targets: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set]. This contains integers in the range from 0 to
      the way of the episode.

  Returns:
    way_to_accuracy: A dict that maps each 'observed' way to a list that
      stores the accuracy obtained for different episodes of that way.
  """
  way_to_accuracy = collections.defaultdict(list)
  for episode_num, episode_way in enumerate(ways):
    episode_logits = logits[episode_num]
    assert episode_way == episode_logits.shape[1]
    episode_targets = targets[episode_num]
    episode_acc = compute_episode_accuracy(episode_logits, episode_targets)
    way_to_accuracy[episode_way].append(episode_acc)
  return way_to_accuracy


def get_finegrainedness_split_enum():
  """Returns the Split corresponding to FLAGS.eval_finegrainedness_split."""
  if FLAGS.eval_finegrainedness_split == 'train':
    split_enum = learning_spec.Split.TRAIN
  elif FLAGS.eval_finegrainedness_split == 'valid':
    split_enum = learning_spec.Split.VALID
  elif FLAGS.eval_finegrainedness_split == 'test':
    split_enum = learning_spec.Split.TEST
  return split_enum


def get_synsets_from_class_ids(class_ids):
  """Returns the Synsets of the appropriate subgraph corresponding to class_ids.

  For each class id in class_ids, the corresponding Synset is found among the
  Synsets of the subgraph corresponding to the split that is chosen for the
  fine-grainedness analysis.

  Args:
    class_ids: A np.array of ints in the range between 1 and the total number of
      classes that contains the two class id's chosen for an episode.

  Returns:
    A list of Synsets.

  Raises:
    ValueError: The dataset specification is not found in the expected location.
  """
  # First load the DatasetSpecification of ImageNet.
  dataset_records_path = os.path.join(FLAGS.records_root_dir, 'ilsvrc_2012')
  imagenet_data_spec = dataset_spec.load_dataset_spec(dataset_records_path)

  # A set of Synsets of the split's subgraph.
  split_enum = get_finegrainedness_split_enum()
  split_subgraph = imagenet_data_spec.split_subgraphs[split_enum]

  # Go from class_ids (integers in the range from 1 to the total number of
  # classes in the Split) to WordNet id's, e.g n02075296.
  wn_ids = []
  for class_id in class_ids:
    wn_ids.append(imagenet_data_spec.class_names[class_id])

  # Find the Synsets in split_subgraph whose WordNet id's are wn_ids.
  synsets = imagenet_spec.get_synsets_from_ids(wn_ids, split_subgraph)
  return [synsets[wn_id] for wn_id in wn_ids]


def get_height_to_accuracy(class_ids, logits, targets):
  """Accuracy as a function of the height of class' the lowest common ancestor.

  This is only applicable to 2-way ImageNet episodes. Given the class set of
  each episode, we find the corresponding 2 leaves of the ImageNet graph and
  compute the lowest common ancestor of those leaves. Its height is computed as
  the maximum over the length of the paths from that node to each of the two
  leaves. This height is the estimate of fine-grainedness. Intuitively, the
  larger the height, the more coarse-grained the episode's classification task.

  Args:
    class_ids: A list containing a np.array per episode that contains the two
      class id's chosen for the episode's binary classification task. These id's
      are ints in the range between 1 and the total number of classes.
    logits: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set, way].
    targets: A list containing a np.array per episode. The shape of an episode's
      array is [size of test set]. This contains integers in the range from 0 to
      the way of the episode.

  Returns:
    height_to_accuracy: A dict that maps each 'observed' height to a list that
      contains the accuracies obtained for different episodes with that height.

  Raises:
    ValueError: There should have been exactly 2 elements in the list of each
      episode's class id's.
  """
  height_to_accuracy = collections.defaultdict(list)
  for episode_num, episode_class_ids in enumerate(class_ids):
    if len(episode_class_ids) != 2:
      raise ValueError('There should have been exactly 2 elements in the list '
                       "of each episode's class id's.")
    # Get the Synsets corresponding to the class id's episode_class_ids.
    episode_synsets = get_synsets_from_class_ids(episode_class_ids)
    assert len(episode_synsets) == 2, ('Fine- vs coarse- grained analysis '
                                       'should be restricted to binary tasks.')
    # Compute the height of the lowest common ancestor of the episode's Synsets.
    _, height = imagenet_spec.get_lowest_common_ancestor(
        episode_synsets[0], episode_synsets[1])
    # Compute the accuracy of the episode.
    episode_logits = logits[episode_num]
    episode_targets = targets[episode_num]
    episode_acc = compute_episode_accuracy(episode_logits, episode_targets)
    height_to_accuracy[height].append(episode_acc)
  return height_to_accuracy


def summarize_values_stats(d):
  """Summarizes each list value of dict d into a mean and confidence interval.

  The summarized version of an empty dictionary, is also empty.

  Args:
    d: A dict where each value is a list.

  Returns:
    d_mean_ci: If d is not empty, a dict with the same keys as d but with each
      value which was originally a list replaced with a tuple of the mean of
      that list and the corresponding confidence interval.

  Raises:
    ValueError: The values of d are not lists.
  """
  if not d:
    return {}

  for v in d.values():
    if not isinstance(v, list):
      raise ValueError('The values of the provided dict are not lists.')

  d_mean_ci = {}
  for k, v in d.items():
    mean = np.mean(v)
    ci = np.std(v) * 1.96 / np.sqrt(len(v))
    d_mean_ci[k] = (mean, ci)
  return d_mean_ci


def read_data(input_path, do_finegrainedness_analysis, do_imbalance_analysis):
  """Reads the data from the evaluation files.

  Args:
    input_path: The path to the event file to read from.
    do_finegrainedness_analysis: Whether to perform analysis of fine- vs coarse-
      grained tasks. This affects the tags that are necessary to find in the
      event files.
    do_imbalance_analysis: Whether to analyze performance for episodes that are
      characterized by different degrees of imbalance.

  Returns:
    ways: A list containing the 'way' of each episode.
    shots: A list containing a np.array per episode. The shape of an episode's
      array is the [way].
    class_ids: A list containing a np.array per episode which contains two class
      id's representing the two classes chosen for that binary classification.
    test_logits: A list containing a np.array per episode. The shape of an
      episode's array is [size of test set, way].
    test_targets: A list containing a np.array per episode. The shape of an
      episode's array is [size of test set]. This contains integers in the range
      from 0 to the way of the episode.

  Raises:
    ValueError: Finegrainedness analysis is requested but no summaries of
      class_ids are found for the provided split, or imbalance analysis is
      requested but no summaries of class_props are found.
  """
  split = (
      FLAGS.eval_finegrainedness_split
      if FLAGS.eval_finegrainedness else 'test')
  logging.info('Reading event file %s for summaries of split %s.', input_path,
               split)
  (ways, shots, class_props, class_ids, test_logits,
   test_targets) = [], [], [], [], [], []
  tags = set()
  for e in tf.train.summary_iterator(input_path):
    for v in e.summary.value:
      tags.add(v.tag)
      if v.tag == '{}_way'.format(split):
        ways.append(v.simple_value)
      elif v.tag == '{}_shots'.format(split):
        shots.append(tf.make_ndarray(v.tensor))
      elif v.tag == '{}_class_props'.format(split):
        class_props.append(tf.make_ndarray(v.tensor))
      elif v.tag == '{}_class_ids'.format(split):
        class_ids.append(tf.make_ndarray(v.tensor))
      elif v.tag == '{}_test_logits'.format(split):
        test_logits.append(tf.make_ndarray(v.tensor))
      elif v.tag == '{}_test_targets'.format(split):
        test_targets.append(tf.make_ndarray(v.tensor))
  if do_finegrainedness_analysis and not class_ids:
    raise ValueError(
        'No summaries found with tag: {}_class_ids. The tags that exist in the '
        'event file are: {}.'.format(split, list(tags)))
  if do_imbalance_analysis and not class_props:
    raise ValueError(
        'No summaries found with tag: {}_class_props. The tags that exist in '
        'the event file are: {}.'.format(split, list(tags)))
  return ways, shots, class_props, class_ids, test_logits, test_targets


def write_pkl(output_data, output_path):
  """Save output_data to the pickle at output_path."""
  with tf.io.gfile.GFile(output_path, 'wb') as f:
    pkl.dump(output_data, f, protocol=pkl.HIGHEST_PROTOCOL)
  logging.info('Dumped data with keys: %s to location %s',
               list(output_data.keys()), output_path)


def read_pkl(output_path):
  """Returns the contents of a pickle file or False if it doesn't exist."""
  if tf.io.gfile.exists(output_path):
    with tf.io.gfile.GFile(output_path, 'rb') as f:
      data = pkl.load(f)
      logging.info('Read data with keys: %s', list(data.keys()))
      return data
  else:
    return False


def get_event_files(root_dir):
  """Returns all event files from the subdirectories of root_dir.

  Args:
    root_dir: A str. The root directory of evaluation experiments.
  Assumes the following directory organization: root_dir contains a sub-
    directory for every dataset, and each of those contains a directory named
    'summaries' where an event file can be found.
  """
  paths_to_events = []
  summaries_dir = os.path.join(root_dir, 'summaries')
  assert tf.io.gfile.isdir(summaries_dir), ('Could not find summaries in %s.' %
                                            root_dir)

  if int(FLAGS.restrict_to_subexperiment) > 0:
    child_dirs = [os.path.join(summaries_dir, FLAGS.restrict_to_subexperiment)]
  else:
    child_dirs = [
        os.path.join(summaries_dir, f)
        for f in tf.io.gfile.listdir(summaries_dir)
    ]
  # Filter out non-directory files, if any.
  child_dirs = [child for child in child_dirs if tf.io.gfile.isdir(child)]
  logging.info('Looking for events in dirs: %s', child_dirs)
  for child_dir in child_dirs:
    for file_name in tf.io.gfile.listdir(child_dir):
      if 'event' in file_name:
        paths_to_events.append(os.path.join(child_dir, file_name))
  logging.info('Found events: %s', paths_to_events)
  return paths_to_events


def get_output_path(path_to_event_file):
  """Returns the path where the pickle of output data will be stored.

  Args:
    path_to_event_file: The path where the event file lives. Used so that the
      output pickle is stored in that same directory.
  """
  # Get the directory where the event file was found.
  event_dir, _ = os.path.split(path_to_event_file)
  out_pickle_path = os.path.join(event_dir, 'aggregated_summary_dicts.pklz')
  return out_pickle_path


def combine_dicts(dict_list):
  """Combines the dictionaries in dict_list.

  Args:
    dict_list: A list of dicts. Each dict maps integers to lists.

  Returns:
    combined: A dict that has for every key the 'combined' values of all dicts
    in dict list that have that key. Combining the values for a key amounts to
    concatenating the corresponding lists.
  """
  combined = collections.defaultdict(list)
  for d in dict_list:
    for k, v in d.items():
      combined[k].extend(v)
  return combined


def analyze_events(paths_to_event_files, experiment_root_dir,
                   do_finegrainedness_analysis, do_imbalance_analysis,
                   force_recompute):
  """Analyzes each event file and stores the .pklz in the corresponding dir."""

  # Aggregate stats across all event files and write (the summarized version of
  # those) to the root directory.
  shot_to_precision_all = []
  way_to_accuracy_all = []
  height_to_accuracy_all = []
  imbalance_to_accuracy_all = []

  for path_to_event in paths_to_event_files:
    output_pickle = get_output_path(path_to_event)

    # First check if the required data is already computed and written.
    maybe_data = False if force_recompute else read_pkl(output_pickle)
    if maybe_data:
      logging.info('Output %s already exists. Skipping it.', output_pickle)
      shot_to_precision = maybe_data['shot_to_precision']
      way_to_accuracy = maybe_data['way_to_accuracy']
      height_to_accuracy = maybe_data['height_to_accuracy']
      imbalance_to_accuracy = maybe_data['imbalance_to_accuracy']

    else:
      # Read the data from the event files.
      (ways, shots, class_props, class_ids, test_logits,
       test_targets) = read_data(path_to_event, do_finegrainedness_analysis,
                                 do_imbalance_analysis)

      # A dict mapping each observed 'shot' to a list of class precisions
      # obtained by classes that had that shot (regardless of shots of other
      # classes in the same episode).
      shot_to_precision = get_shot_to_precision(shots, test_logits,
                                                test_targets)

      # A dict mapping each observed 'way' of an episode to a list of accuracies
      # of the episodes with that way.
      way_to_accuracy = get_way_to_accuracy(ways, test_logits, test_targets)

      # A dict mapping the height of the lowest common ancestor of each pair of
      # leaves defining the binary classiifcation task to the task's accuracy.
      height_to_accuracy = {}
      if do_finegrainedness_analysis:
        height_to_accuracy = get_height_to_accuracy(class_ids, test_logits,
                                                    test_targets)

      # A dict mapping the degree of imabalance of tasks to their accuracy.
      imbalance_to_accuracy = {}
      if do_imbalance_analysis:
        imbalance_to_accuracy = get_imbalance_to_accuracy(
            class_props, test_logits, test_targets)

      # Keep only the mean and confidence intervals instead of the list of all
      # precisons or accuracies, resp. as the values of these dicts.
      shot_to_precision_summarized = summarize_values_stats(shot_to_precision)
      way_to_accuracy_summarized = summarize_values_stats(way_to_accuracy)
      height_to_accuracy_summarized = summarize_values_stats(height_to_accuracy)
      imbalance_to_accuracy_summarized = summarize_values_stats(
          imbalance_to_accuracy)

      # Save the two dicts to a pickle at the designated location.
      output_data = {
          'shot_to_precision': shot_to_precision,
          'way_to_accuracy': way_to_accuracy,
          'height_to_accuracy': height_to_accuracy,
          'imbalance_to_accuracy': imbalance_to_accuracy,
          'shot_to_precision_summarized': shot_to_precision_summarized,
          'way_to_accuracy_summarized': way_to_accuracy_summarized,
          'height_to_accuracy_summarized': height_to_accuracy_summarized,
          'imbalance_to_accuracy_summarized': imbalance_to_accuracy_summarized,
      }
      write_pkl(output_data, output_pickle)

    shot_to_precision_all.append(shot_to_precision)
    way_to_accuracy_all.append(way_to_accuracy)
    height_to_accuracy_all.append(height_to_accuracy)
    imbalance_to_accuracy_all.append(imbalance_to_accuracy)

  # Now aggregate the stats across datasets.
  shot_to_precision_all = combine_dicts(shot_to_precision_all)
  way_to_accuracy_all = combine_dicts(way_to_accuracy_all)
  height_to_accuracy_all = combine_dicts(height_to_accuracy_all)
  imbalance_to_accuracy_all = combine_dicts(imbalance_to_accuracy_all)

  # Summarize them.
  shot_to_precision_all_summarized = summarize_values_stats(
      shot_to_precision_all)
  way_to_accuracy_all_summarized = summarize_values_stats(way_to_accuracy_all)
  height_to_accuracy_all_summarized = summarize_values_stats(
      height_to_accuracy_all)
  imbalance_to_accuracy_all_summarized = summarize_values_stats(
      imbalance_to_accuracy_all)

  # Save the dicts to a pickle at the designated location.
  output_data = {
      'shot_to_precision': shot_to_precision_all,
      'way_to_accuracy': way_to_accuracy_all,
      'height_to_accuracy': height_to_accuracy_all,
      'shot_to_precision_summarized': shot_to_precision_all_summarized,
      'way_to_accuracy_summarized': way_to_accuracy_all_summarized,
      'height_to_accuracy_summarized': height_to_accuracy_all_summarized,
      'imbalance_to_accuracy_summarized': imbalance_to_accuracy_all_summarized
  }
  pickle_name_base = 'aggregated_summary_dicts'
  if int(FLAGS.restrict_to_subexperiment) > 0:
    pickle_name_base += '_eval_{}'.format(FLAGS.restrict_to_subexperiment)
  output_pickle = os.path.join(experiment_root_dir, pickle_name_base + '.pklz')
  write_pkl(output_data, output_pickle)


def main(argv):
  del argv
  paths_to_event_files = get_event_files(FLAGS.root_dir)
  if not paths_to_event_files:
    logging.info('No event files found.')
    return
  analyze_events(paths_to_event_files, FLAGS.root_dir,
                 FLAGS.eval_finegrainedness, FLAGS.eval_imbalance,
                 FLAGS.force_recompute)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
