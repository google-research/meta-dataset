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

r"""Check that a dataset's .tfrecords and datataset_spec.json are consistent.

In particular, check that:
- The name of the directory containing these files corresponds to the "dataset"
field of dataset_spec.
- The number of .tfrecords corresponds to the number of classes, and they are
numbered sequentially
- The number of examples in each .tfrecords file corresponds to the one in the
dataset_spec.
- The class label inside each tf.Example in i.tfrecords is i.

Example:

  TF_CPP_MIN_LOG_LEVEL=1 python -m meta_dataset.dataset_conversion.check_dataset_consistency \
    --dataset_records_path=$RECORDS/traffic_sign
"""

import json
import os

from absl import app
from absl import flags
from absl import logging
from meta_dataset.data import dataset_spec as dataset_spec_lib
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_records_path', None, 'The name of a directory '
    'containing .tfrecords files and a dataset_spec.json.')
flags.DEFINE_string(
    'dataset_spec_file', None, 'Optional, the name and path to '
    'the dataset spec JSON file to use instead of the one in '
    'dataset_records_path.')
flags.DEFINE_string(
    'label_field_name', 'label', 'The name of the field, '
    'inside each tf.Example, that represents the label.')
flags.mark_flag_as_required('dataset_records_path')


def count_records(tfrecords_path):
  """Returns the number of tf.Examples in a .tfrecords file."""
  dataset = tf.data.TFRecordDataset(tfrecords_path)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  count = 0
  with tf.Session() as sess:
    try:
      while True:
        sess.run(next_element)
        count += 1
    except tf.errors.OutOfRangeError:
      pass
  return count


def get_count_and_labels(tfrecords_path, label_field_name):
  """Returns the number of examples and the set of labels in a dataset."""
  dataset = tf.data.TFRecordDataset(tfrecords_path)

  def _parse_function(example_string):
    features = {label_field_name: tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(example_string, features=features)
    label = example[label_field_name]
    return label

  dataset = dataset.map(_parse_function)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  count = 0
  values = set()
  with tf.Session() as sess:
    try:
      while True:
        value = sess.run(next_element)
        count += 1
        values.add(value)
    except tf.errors.OutOfRangeError:
      pass
  return count, values


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Report failed checks as they occur and maintain a counter, instead of
  # raising exceptions right away, so all issues can be reported at once.
  num_failed_checks = 0

  # Load dataset_spec, this should fail if it is absent or incorrect.
  if FLAGS.dataset_spec_file is None:
    dataset_spec = dataset_spec_lib.load_dataset_spec(
        FLAGS.dataset_records_path)
  else:
    with tf.io.gfile.GFile(FLAGS.dataset_spec_file, 'r') as f:
      dataset_spec = json.load(f, object_hook=dataset_spec_lib.as_dataset_spec)

  dataset_spec.initialize()

  # 1. Check dataset name
  dir_name = os.path.basename(os.path.abspath(FLAGS.dataset_records_path))
  if dataset_spec.name != dir_name:
    num_failed_checks += 1
    logging.error(
        'The dataset name in "dataset_spec.json" (%s) does not match '
        'the name of the directory containing it (%s)', dataset_spec.name,
        dir_name)

  # 2. Check name and number of .tfrecords files
  num_classes = len(dataset_spec.class_names)
  try:
    expected_filenames = [
        dataset_spec.file_pattern.format(class_id)
        for class_id in range(num_classes)
    ]
  except IndexError:
    num_failed_checks += 1
    err_msg = (
        'The `file_pattern` (%s) did not accept the class number as its only '
        'formatting argument. Using the default (%s).')
    default_pattern = '{}.tfrecords'
    logging.error(err_msg, dataset_spec.file_pattern, default_pattern)

    expected_filenames = [
        default_pattern.format(class_id) for class_id in range(num_classes)
    ]

  all_filenames = tf.io.gfile.listdir(FLAGS.dataset_records_path)
  # Heuristic to exclude obviously-not-tfrecords files.
  tfrecords_filenames = [f for f in all_filenames if 'tfrecords' in f.lower()]

  expected_set = set(expected_filenames)
  present_set = set(tfrecords_filenames)
  if set(expected_set) != set(present_set):

    num_failed_checks += 1
    logging.error(
        'The tfrecords files in %s do not match the dataset_spec.\n'
        'Unexpected files present:\n'
        '%s\n'
        'Expected files not present:\n'
        '%s', FLAGS.dataset_records_path, sorted(present_set - expected_set),
        sorted(expected_set - present_set))

  # Iterate through each dataset, count examples and check set of targets.
  # List of (class_id, expected_count, actual_count) triples.
  bad_counts = []
  # List of (filename, class_id, labels).
  bad_labels = []

  for class_id, filename in enumerate(expected_filenames):
    expected_count = dataset_spec.get_total_images_per_class(class_id)
    if filename not in tfrecords_filenames:
      # The tfrecords does not exist, we use a negative count to denote it.
      bad_counts.append((class_id, expected_count, -1))
      bad_labels.append((filename, class_id, set()))
      continue
    full_filepath = os.path.join(FLAGS.dataset_records_path, filename)

    try:
      count, labels = get_count_and_labels(full_filepath,
                                           FLAGS.label_field_name)
    except tf.errors.InvalidArgumentError:
      logging.exception(
          'Unable to find label (%s) in the tf.Examples of file %s. '
          'Maybe try a different --label_field_name.', FLAGS.label_field_name,
          filename)
      # Fall back to counting examples only.
      count = count_records(full_filepath)
      labels = set()
    if count != expected_count:
      bad_counts.append((class_id, expected_count, count))
    if labels != {class_id}:
      # labels could include class_id among other, incorrect labels.
      bad_labels.append((filename, class_id, labels))

  # 3. Check number of examples
  if bad_counts:
    num_failed_checks += 1
    logging.error(
        'The number of tfrecords in the following files do not match '
        'the expected number of examples in that class.\n'
        '(filename, expected, actual)  # -1 denotes a missing file.\n'
        '%s', bad_counts)

  # 4. Check the targets stored in the tfrecords files.
  if bad_labels:
    num_failed_checks += 1
    logging.error(
        'The labels stored inside the tfrecords (in field %s) do not '
        'all match the expected value (class_id).\n'
        '(filename, class_id, values)\n'
        '%s', FLAGS.label_field_name, bad_labels)

  # Report results
  if num_failed_checks:
    raise ValueError('%d checks failed. See the error-level logs.' %
                     num_failed_checks)


if __name__ == '__main__':
  app.run(main)
