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
# pyformat: disable
"""Implementation of the log expected empirical prediction measure (LEEP).

#### References

[1]: Nguyen, Cuong V., Tal Hassner, Matthias Seeger, and Cedric Archambeau.
     LEEP: A new measure to evaluate transferability of learned representations.
     In _Proceedings of 37th International Conference on Machine Learning_,
     2020.
     https://arxiv.org/abs/2002.12462
"""
# pyformat: enable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def compute_leep(source_predictions,
                 target_onehot_labels):
  """Compute LEEP using `source_predictions` and `target_onehot_labels`.

  Args:
    source_predictions: Predictions over the source label set for a batch of
      input data from the target dataset.
    target_onehot_labels: One-hot labels from the target label set.

  Returns:
    The log expected empirical prediction measure (LEEP) on the given batch.
  """
  batch_size = tf.cast(source_predictions.shape[0], tf.float32)
  if target_onehot_labels.shape[0] != batch_size:
    raise ValueError('`source_predictions` and `target_onehot_labels` must '
                     'represent the same number of examples.')

  # Normalize the predicted probabilities in log space.
  source_predictions -= tf.math.reduce_logsumexp(
      source_predictions, axis=1, keepdims=True)

  # p_model(y, z | x_i) for source labels `z`, target labels `y`, and each `i`.
  log_per_example_full_joint = tf.einsum('iz,iy->iyz', source_predictions,
                                         target_onehot_labels)

  # Workaround for one-hot indexing in log space.
  log_per_example_full_joint = (
      tf.where(
          tf.expand_dims(target_onehot_labels != 0, axis=2),
          log_per_example_full_joint,
          tf.ones_like(log_per_example_full_joint) * -np.inf))

  # Re-normalize the joint.
  log_per_example_full_joint -= tf.math.reduce_logsumexp(
      log_per_example_full_joint, axis=(1, 2), keepdims=True)

  # Average examples-wise probabilities (Eq. (1) in the paper).
  log_full_joint = tf.math.reduce_logsumexp(
      log_per_example_full_joint, axis=0) - tf.math.log(batch_size)

  # p_model(z) for source labels `z`, marginalizing out target labels `y`.
  log_full_target_marginal = tf.math.reduce_logsumexp(
      source_predictions, axis=0) - tf.math.log(batch_size)

  # p_model(y | z) for source labels `y`, conditioning on target labels `z`.
  log_full_conditional = (
      log_full_joint - tf.expand_dims(log_full_target_marginal, axis=0))

  # p_model(y = y_i | z) for datapoint `i`.
  log_predicted_conditional = (
      tf.einsum('iy,yz->iz', target_onehot_labels, log_full_conditional))

  # p_model(y = y_i | z) * p_model(z) for datapoint `i`.
  log_predicted_joint = (log_predicted_conditional + source_predictions)

  # p_model(y = y_i) for datapoint `i`.
  log_predicted_marginal = tf.math.reduce_logsumexp(log_predicted_joint, axis=1)

  return tf.reduce_sum(log_predicted_marginal) / batch_size
