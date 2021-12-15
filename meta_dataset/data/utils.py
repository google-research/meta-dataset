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
r"""Utils for dumping Meta-Dataset episodes to disk as tfrecords files."""
import collections
import os

from absl import logging

from meta_dataset.dataset_conversion import dataset_to_records
import tensorflow.compat.v1 as tf

TRAIN_SUFFIX = 'train.tfrecords'
TEST_SUFFIX = 'test.tfrecords'
# For output file.
FILE_NAME_TEMPLATE = 'episode-{:04d}-{}'


def get_label_counts(labels):
  """Creates a JSON compatible dictionary of image per class counts."""
  # JSON does not support integer keys.
  counts = {str(k): v for k, v in collections.Counter(labels.numpy()).items()}
  return counts


def dump_as_tfrecord(path, images, labels):
  logging.info('Dumping records to: %s', path)
  with tf.io.TFRecordWriter(path) as writer:
    for image, label in zip(images, labels):
      dataset_to_records.write_example(image.numpy(), label.numpy(), writer)


def get_file_path(folder, idx, split):
  if split == 'train':
    suffix = TRAIN_SUFFIX
  elif split == 'test':
    suffix = TEST_SUFFIX
  else:
    raise ValueError('Split: %s, should be train or test.' % split)
  return os.path.join(folder, FILE_NAME_TEMPLATE.format(idx, suffix))


def get_info_path(folder):
  return os.path.join(folder, 'images_per_class.json')
