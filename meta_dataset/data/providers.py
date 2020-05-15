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
"""Interfaces for data returned by the pipelines.

TODO(lamblinp): Integrate better with pipeline.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow.compat.v1 as tf


def compute_shot(way, labels):
  """Computes the `shot` of the episode containing labels.

  Args:
    way: An int constant tensor. The number of classes in the episode.
    labels: A Tensor of labels of shape [batch_size].

  Returns:
    shots: An int 1D tensor: The number of support examples per class.
  """
  class_ids = tf.reshape(tf.range(way), [way, 1])
  class_labels = tf.reshape(labels, [1, -1])
  is_equal = tf.equal(class_labels, class_ids)
  return tf.reduce_sum(tf.cast(is_equal, tf.int32), axis=1)


def compute_unique_class_ids(class_ids):
  """Computes the unique class IDs of the episode containing `class_ids`.

  Args:
    class_ids: A 1D tensor representing class IDs, one per example in an
      episode.

  Returns:
    A 1D tensor of the unique class IDs whose size is equal to the way of an
    episode.
  """
  return tf.unique(class_ids)[0]


class EpisodeDataset(
    collections.namedtuple(
        'EpisodeDataset', 'train_images, test_images, '
        'train_labels, test_labels, train_class_ids, test_class_ids')):
  """Wraps an episode's data and facilitates creation of feed dict.

    Args:
      train_images: A Tensor of images for training.
      test_images: A Tensor of images for testing.
      train_labels: A 1D Tensor, the matching training labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      test_labels: A 1D Tensor, the matching testing labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      train_class_ids: A 1D Tensor, the matching training class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
      test_class_ids: A 1D Tensor, the matching testing class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
  """

  @property
  def unique_class_ids(self):
    return compute_unique_class_ids(
        tf.concat((self.train_class_ids, self.test_class_ids), -1))

  @property
  def train_shots(self):
    return compute_shot(self.way, self.train_labels)

  @property
  def test_shots(self):
    return compute_shot(self.way, self.test_labels)

  # TODO(evcu) We should probably calculate way from unique labels, not
  # class_ids.
  @property
  def way(self):
    return tf.size(self.unique_class_ids)

  @property
  def labels(self):
    """Return query labels to provide an episodic/batch-agnostic API."""
    return self.test_labels

  @property
  def onehot_labels(self):
    """Return one-hot query labels to provide an episodic/batch-agnostic API."""
    return self.onehot_test_labels

  @property
  def onehot_train_labels(self):
    return tf.one_hot(self.train_labels, self.way)

  @property
  def onehot_test_labels(self):
    return tf.one_hot(self.test_labels, self.way)


class Batch(collections.namedtuple('Batch', 'images, labels, n_classes')):
  """Wraps an batch's data and facilitates creation of feed dict.

    Args:
      images: a Tensor of images of shape [self.batch_size] + image shape.
      labels: a Tensor of labels of shape [self.batch_size].
      n_classes: a scalar int Tensor, the total number of available classes
        (labels). Used to express targets as 1-hot vectors.
  """

  @property
  def way(self):
    """Compute the way of the episode.

    Returns:
      way: An int, the number of possible classes in the dataset.
    """
    return self.n_classes

  @property
  def onehot_labels(self):
    return tf.one_hot(self.labels, self.way)
