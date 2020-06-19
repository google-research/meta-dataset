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
"""Abstract learners."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import logging
import gin.tf
from meta_dataset.models import functional_backbones
import tensorflow.compat.v1 as tf


@gin.configurable
class Learner(object):
  """A Learner."""

  def __init__(
      self,
      is_training,
      logit_dim,
      transductive_batch_norm,
      backprop_through_moments,
      embedding_fn,
      weight_decay,
      input_shape,
  ):
    """Initializes a Learner.

    Note that Gin configuration of subclasses of `Learner` will override any
    corresponding Gin configurations of `Learner`, since parameters are passed
    to the `Learner` base class's constructor (See
    https://github.com/google/gin-config/blob/master/README.md) for more
    details).

    Args:
      is_training: Whether the learning is in training mode.
      logit_dim: An integer; the maximum dimensionality of output predictions.
      transductive_batch_norm: Whether to batch-normalize in the transductive
        setting, where means and variances for normalization are computed from
        each of the support and query sets (rather than using the support set
        statistics for normalization of both the support and query set).
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      embedding_fn: A string; the name of the function that embeds images.
      weight_decay: coefficient for L2 regularization.
      input_shape: A Tensor corresponding to `[batch_size] + example_shape`.
    """
    self.is_training = is_training
    self.logit_dim = logit_dim
    self.input_shape = input_shape
    self.transductive_batch_norm = transductive_batch_norm
    self.backprop_through_moments = backprop_through_moments
    self.embedding_fn = functional_backbones.NAME_TO_EMBEDDING_NETWORK[
        embedding_fn]
    self.weight_decay = weight_decay

    if self.transductive_batch_norm:
      logging.info('Using transductive batch norm!')

  def compute_loss(self, onehot_labels, predictions):
    """Computes the CE loss of `predictions` with respect to `onehot_labels`.

    Args:
      onehot_labels: A `tf.Tensor` containing the the class labels; each vector
        along the class dimension should hold a valid probability distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` representing the average loss.
    """
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=predictions)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self, labels, predictions):
    """Computes the accuracy of `predictions` with respect to `labels`.

    Args:
      labels: A `tf.Tensor` containing the the class labels; each vector along
        the class dimension should hold a valid probability distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` of ones and zeros representing the individual accuracies.
         Use tf.reduce_mean(result) to obtain average accuracy.
    """
    correct = tf.equal(labels, tf.to_int32(tf.argmax(predictions, -1)))
    return tf.cast(correct, tf.float32)

  def forward_pass(self, data):
    """Returns the (query if episodic) logits."""
    raise NotImplementedError('Abstract method.')


class EpisodicLearner(Learner):
  """An episodic learner."""

  pass


class BatchLearner(Learner):
  """A batch learner."""

  pass
