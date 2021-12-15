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
"""Interfaces for data returned by the pipelines.

TODO(lamblinp): Integrate better with pipeline.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from meta_dataset import distribute_utils
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


class Episode(
    collections.namedtuple(
        'Episode', 'support_images, query_images, '
        'support_labels, query_labels, support_class_ids, query_class_ids')):
  """Wraps an episode's data and facilitates creation of feed dict.

    Args:
      support_images: A Tensor of images for fitting an episodic model.
      query_images: A Tensor of images for evaluating an episodic model.
      support_labels: A 1D Tensor, the matching support labels (numbers between
        0 and K-1, with K the number of classes involved in the episode).
      query_labels: A 1D Tensor, the matching query labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      support_class_ids: A 1D Tensor, the matching support class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
      query_class_ids: A 1D Tensor, the matching query class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
  """

  @property
  def unique_class_ids(self):
    return compute_unique_class_ids(
        tf.concat((self.support_class_ids, self.query_class_ids), -1))

  @property
  def support_shots(self):
    return compute_shot(self.way, self.support_labels)

  @property
  def query_shots(self):
    return compute_shot(self.way, self.query_labels)

  @property
  def way(self):
    return tf.size(self.unique_labels)

  @property
  def unique_labels(self):
    return tf.unique(tf.concat((self.support_labels, self.query_labels), -1))[0]

  @property
  def labels(self):
    """Return query labels to provide an episode/batch-independent API."""
    return self.query_labels

  @property
  def onehot_labels(self):
    """Return one-hot query labels to provide an episode/batch-independent API."""
    return self.onehot_query_labels

  @property
  def onehot_support_labels(self):
    return tf.one_hot(self.support_labels, self.way)

  @property
  def onehot_query_labels(self):
    return tf.one_hot(self.query_labels, self.way)


class EpisodePiece(
    collections.namedtuple(
        'Episode', 'support_images, query_images, support_labels, '
        'query_labels, support_class_ids, query_class_ids, way')):
  """Wraps an episode's data and facilitates creation of feed dict.

    This class provides the same functionality as an Episode, but it's intended
    for use in a distributed setting: it contains only a chunk of the support
    and query images in the episode (and their corresponding labels); basic
    accessor methods (support/query_images, support/query_labels,
    one_hot_support/query_labels, support/query_class_ids) will return only
    the local values.  The remaining properties, however, refer to the global
    episode stats (way, shots, unique_class_id's) and may be aggregated on
    demand.

    Args:
      support_images: A Tensor of images for fitting an episodic model.
      query_images: A Tensor of images for evaluating an episodic model.
      support_labels: A 1D Tensor, the matching support labels (numbers between
        0 and K-1, with K the number of classes involved in the episode).
      query_labels: A 1D Tensor, the matching query labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      support_class_ids: A 1D Tensor, the matching support class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
      query_class_ids: A 1D Tensor, the matching query class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
      way: number of classes in the original episode.
  """

  @property
  def unique_class_ids(self):
    """Return global unique class id's for the episode."""
    return compute_unique_class_ids(
        tf.concat((distribute_utils.aggregate(self.support_class_ids),
                   distribute_utils.aggregate(self.query_class_ids)), -1))

  @property
  def support_shots(self):
    """Return global support shots for the episode."""
    return compute_shot(self.way,
                        distribute_utils.aggregate(self.support_labels))

  @property
  def query_shots(self):
    """Return global query shots for the episode."""
    return compute_shot(self.way, distribute_utils.aggregate(self.query_labels))

  @property
  def labels(self):
    """Return local query labels to provide an episode/batch-independent API."""
    return self.query_labels

  @property
  def onehot_labels(self):
    """Return local one-hot query labels for episode/batch-independent API."""
    return self.onehot_query_labels

  @property
  def onehot_support_labels(self):
    """Return local one-hot support labels."""
    return tf.one_hot(self.support_labels, self.way)

  @property
  def onehot_query_labels(self):
    """Return local one-hot query labels."""
    return tf.one_hot(self.query_labels, self.way)


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
