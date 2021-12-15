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
"""Baseline learners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin.tf
from meta_dataset.learners import base as learner_base
from meta_dataset.models import functional_classifiers
import numpy as np
import tensorflow.compat.v1 as tf


# TODO(eringrant): Factor out all the different variants for episodic evaluation
# into different classes rather than relying on conditional statements.
@gin.configurable
class BaselineLearner(learner_base.BatchLearner):
  """A Baseline Network."""

  # TODO(eringrant): Remove this attribute when the `BaselineFinetuneLearner`
  # subclass is refactored to obey the interface of `Learner.compute_logits`.
  obeys_compute_logits_interface = True

  def __init__(self, knn_in_fc, knn_distance, cosine_classifier,
               cosine_logits_multiplier, use_weight_norm, **kwargs):
    """Initializes a baseline learner.

    Args:
      knn_in_fc: Whether kNN is performed in the space of fc activations or
        embeddings. If True, the logits from the last fc layer are used as the
        embedding on which kNN lookup is performed. Otherwise, the penultimate
        layer is what the kNN lookup is performed on.
      knn_distance: The distance measurement used by kNN lookup. 'l2', 'cosine'
      cosine_classifier: A bool. Whether to use a cosine classifier at training
        time when performing the all-way classification task to train the
        backbone.
      cosine_logits_multiplier: A float. A scalar that will multiply the logits
        computed by the cosine classifier (if applicable) before passing them
        into the softmax.
      use_weight_norm: A bool. Whether to apply weight normalization to the
        linear classifier layer.
      **kwargs: Keyword arguments common to all BatchLearners.
    """
    self.knn_in_fc = knn_in_fc
    self.distance = knn_distance
    self.cosine_classifier = cosine_classifier
    self.cosine_logits_multiplier = cosine_logits_multiplier
    self.use_weight_norm = use_weight_norm

    super(BaselineLearner, self).__init__(**kwargs)

  def forward_pass(self, data):
    if self.is_training:
      images = data.images
      embeddings_params_moments = self.embedding_fn(images, self.is_training)
      support_embeddings = embeddings_params_moments['embeddings']
      support_logits = self.forward_pass_fc(support_embeddings)
      return support_logits
    else:
      support_embeddings_params_moments = self.embedding_fn(
          data.support_images, self.is_training)
      support_embeddings = support_embeddings_params_moments['embeddings']
      support_set_moments = None
      if not self.transductive_batch_norm:
        support_set_moments = support_embeddings_params_moments['moments']
      query_embeddings = self.embedding_fn(
          data.query_images,
          self.is_training,
          moments=support_set_moments,
          backprop_through_moments=self.backprop_through_moments)
      query_embeddings = query_embeddings['embeddings']

      # TODO(eringrant): The `BaselineFinetuneLearner` subclass is not yet
      # refactored to obey the interface of `Learner.compute_logits`.
      if self.obeys_compute_logits_interface:
        query_logits = self.compute_logits(support_embeddings, query_embeddings,
                                           data.onehot_support_labels)
      else:
        query_logits = self.compute_logits(data)  # pylint: disable=no-value-for-parameter

      return query_logits

  def forward_pass_fc(self, embeddings):
    """Passes the provided embeddings through the fc layer to get the logits.

    Args:
      embeddings: A Tensor of the penultimate layer activations as computed by
        BaselineLearner.forward_pass.

    Returns:
      The fc layer activations.
    """
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      # Always maps to a space whose dimensionality is the number of classes
      # at meta-training time.
      logits = functional_classifiers.linear_classifier(
          embeddings, self.logit_dim, self.cosine_classifier,
          self.cosine_logits_multiplier, self.use_weight_norm)
      return logits

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    """Computes the class logits for the episode.

    Args:
      support_embeddings: A Tensor of size [num_support_images, embedding dim].
      query_embeddings: A Tensor of size [num_query_images, embedding dim].
      onehot_support_labels: A Tensor of size [batch size, way].

    Returns:
      The query set logits as a [num_query_images, way] matrix.

    Raises:
      ValueError: Distance must be one of l2 or cosine.
    """

    if self.knn_in_fc:
      # Recompute the support and query embeddings that were originally computed
      # in self.forward_pass() to be the fc layer activations.
      support_embeddings = self.forward_pass_fc(support_embeddings)
      query_embeddings = self.forward_pass_fc(query_embeddings)

    # ------------------------ K-NN look up -------------------------------
    # For each testing example in an episode, we use its embedding
    # vector to look for the closest neighbor in all the training examples'
    # embeddings from the same episode and then assign the training example's
    # class label to the testing example as the predicted class label for it.
    if self.distance == 'l2':
      #  [1, num_support, embed_dims]
      support_embeddings = tf.expand_dims(support_embeddings, axis=0)
      #  [num_query, 1, embed_dims]
      query_embeddings = tf.expand_dims(query_embeddings, axis=1)
      #  [num_query, num_support]
      distance = tf.norm(query_embeddings - support_embeddings, axis=2)
    elif self.distance == 'cosine':
      support_embeddings = tf.nn.l2_normalize(support_embeddings, axis=1)
      query_embeddings = tf.nn.l2_normalize(query_embeddings, axis=1)
      distance = -1 * tf.matmul(
          query_embeddings, support_embeddings, transpose_b=True)
    else:
      raise ValueError('Distance must be one of l2 or cosine.')
    #  [num_query]
    _, indices = tf.nn.top_k(-distance, k=1)
    indices = tf.squeeze(indices, axis=1)
    #  [num_query, num_classes]
    query_logits = tf.gather(onehot_support_labels, indices)
    return query_logits


@gin.configurable
class DatasetConditionalBaselineLearner(BaselineLearner):
  """A dataset-conditional Baseline Learner with separate readout heads."""

  def __init__(self, num_sets, *args, **kwargs):
    super(DatasetConditionalBaselineLearner, self).__init__(*args, **kwargs)
    self.num_sets = num_sets
    self._set_start_indices_for_sources()

  def forward_pass(self, data, source, *args, **kwargs):
    self.embedding_fn = functools.partial(
        self.embedding_fn, film_selector=source)
    self.forward_pass_fc = functools.partial(
        self.forward_pass_fc, source=source)

    return super(DatasetConditionalBaselineLearner,
                 self).forward_pass(data, *args, **kwargs)

  def forward_pass_fc(self, embeddings, source):
    start_idx = tf.gather(self._start_inds_for_sources, source)
    num_classes = self.logit_dim  # a list of the datasets' numbers of classes.
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      logits = functional_classifiers.separate_head_linear_classifier(
          embeddings, num_classes, source, start_idx, self.cosine_classifier,
          self.cosine_logits_multiplier)
      return logits

  def _set_start_indices_for_sources(self):
    self._start_inds_for_sources = [0] + list(np.cumsum(self.logit_dim))[:-1]

  def _restrict_to_source(self, one_hot_labels, source):
    """Returns the slice of one_hot_labels corresponding to source."""
    return tf.slice(
        one_hot_labels,
        begin=[0, tf.gather(self._start_inds_for_sources, source)],
        size=[tf.shape(one_hot_labels)[0],
              tf.gather(self.logit_dim, source)])

  def compute_loss(self, onehot_labels, predictions, source):
    """Computes the CE loss of `predictions` with respect to `onehot_labels`.

    Args:
      onehot_labels: A `tf.Tensor` containing the the class labels; each vector
        along the class dimension should hold a valid probability distribution
        over the entire set of training classes (of all datasets).
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities, for the classes of the
        relevant head.
      source: An int Tensor. The dataset source for this forward pass.

    Returns:
       A `tf.Tensor` representing the average loss.
    """
    # Restrict the one-hot labels to the range relevant for the given source.
    onehot_labels = self._restrict_to_source(onehot_labels, source)
    return super(DatasetConditionalBaselineLearner,
                 self).compute_loss(onehot_labels, predictions)

  def compute_accuracy(self, onehot_labels, predictions, source):
    """Computes the accuracy."""
    # Restrict the one-hot labels to the range relevant for the given source.
    onehot_labels = self._restrict_to_source(onehot_labels, source)
    return super(DatasetConditionalBaselineLearner,
                 self).compute_accuracy(onehot_labels, predictions)


@gin.configurable
class DatasetLearner(learner_base.BatchLearner):
  """A Learner for the dataset prediction task."""

  def forward_pass(self, data, source=None):
    del source  # Not required for forward pass.
    images = data.images
    logits = self.embedding_fn(images)
    return logits

  def compute_loss(self, onehot_labels, predictions, source=None):
    del source
    return super(DatasetLearner, self).compute_loss(onehot_labels, predictions)

  def compute_accuracy(self, onehot_labels, predictions, source=None):
    del source
    return super(DatasetLearner, self).compute_accuracy(onehot_labels,
                                                        predictions)

