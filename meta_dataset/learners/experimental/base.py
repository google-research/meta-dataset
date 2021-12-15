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
"""Abstract experimental learners that use `ReparameterizableModule`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf

from meta_dataset.learners import base as learner_base
import tensorflow as tf


class NotBuiltError(RuntimeError):

  def __init__(self):
    super(NotBuiltError, self).__init__(
        'The `build` method of `ExperimentalLearner` must be called before '
        'accessing its variables.')


def class_specific_data(onehot_labels, data, num_classes, axis=0):
  # TODO(eringrant): Deal with case of no data for a class in [1...num_classes].
  data_shape = [s for i, s in enumerate(data.shape) if i != axis]
  labels = tf.argmax(onehot_labels, axis=-1)
  class_idx = [tf.where(tf.equal(labels, i)) for i in range(num_classes)]
  return [
      tf.reshape(tf.gather(data, idx, axis=axis), [-1] + data_shape)
      for idx in class_idx
  ]


@gin.configurable
class ExperimentalLearner(learner_base.Learner):
  """An experimental learner."""

  def __init__(self, **kwargs):
    """Constructs an `ExperimentalLearner`.

    Args:
      **kwargs: Keyword arguments common to all `Learner`s.

    Raises:
      ValueError: If the `embedding_fn` provided is not an instance of
        `tf.Module`.
    """
    super(ExperimentalLearner, self).__init__(**kwargs)

    if not isinstance(self.embedding_fn, tf.Module):
      raise ValueError('The `embedding_fn` provided to `ExperimentalLearner`s '
                       'must be an instance of `tf.Module`.')

    self._built = False

  def compute_regularizer(self, onehot_labels, predictions):
    """Computes a regularizer, maybe using `predictions` and `onehot_labels`."""
    del onehot_labels
    del predictions
    return tf.reduce_sum(input_tensor=self.embedding_fn.losses)

  def build(self):
    """Instantiate the parameters belonging to this `ExperimentalLearner`."""
    if not self.embedding_fn.built:
      self.embedding_fn.build([None] + self.input_shape)
    self.embedding_shape = self.embedding_fn.compute_output_shape(
        [None] + self.input_shape)
    self._built = True

  @property
  def variables(self):
    """Returns a list of this `ExperimentalLearner`'s variables."""
    if not self._built:
      raise NotBuiltError
    return self.embedding_fn.variables

  @property
  def trainable_variables(self):
    """Returns a list of this `ExperimentalLearner`'s trainable variables."""
    if not self._built:
      raise NotBuiltError
    return self.embedding_fn.trainable_variables


class ExperimentalEpisodicLearner(ExperimentalLearner,
                                  learner_base.EpisodicLearner):
  """An experimental episodic learner."""

  pass


class ExperimentalBatchLearner(ExperimentalLearner, learner_base.BatchLearner):
  """An experimental batch learner."""

  pass
