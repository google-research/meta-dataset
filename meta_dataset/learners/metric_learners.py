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
"""Metric-based learners."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin.tf
from meta_dataset.learners import base as learner_base
from meta_dataset.models import functional_backbones
import tensorflow.compat.v1 as tf


def _compute_prototypes(embeddings, labels):
  """Computes class prototypes over the last dimension of embeddings.

  Args:
    embeddings: Tensor of examples of shape [num_examples, embedding_size].
    labels: Tensor of one-hot encoded labels of shape [num_examples,
      num_classes].

  Returns:
    prototypes: Tensor of class prototypes of shape [num_classes,
    embedding_size].
  """
  labels = tf.cast(labels, tf.float32)

  # [num examples, 1, embedding size].
  embeddings = tf.expand_dims(embeddings, 1)

  # [num examples, num classes, 1].
  labels = tf.expand_dims(labels, 2)

  # Sums each class' embeddings. [num classes, embedding size].
  class_sums = tf.reduce_sum(labels * embeddings, 0)

  # The prototype of each class is the averaged embedding of its examples.
  class_num_images = tf.reduce_sum(labels, 0)  # [way].
  prototypes = class_sums / class_num_images

  return prototypes


def compute_prototypes(embeddings, labels):
  """Computes class prototypes over features.

  Flattens and reshapes the features if they are not already flattened.
  Args:
    embeddings: Tensor of examples of shape [num_examples, embedding_size] or
      [num_examples, spatial_dim, spatial_dim n_features].
    labels: Tensor of one-hot encoded labels of shape [num_examples,
      num_classes].

  Returns:
    prototypes: Tensor of class prototypes of shape [num_classes,
      embedding_size].
  """
  if len(embeddings.shape) > 2:
    feature_shape = embeddings.shape.as_list()[1:]
    n_images = tf.shape(embeddings)[0]
    n_classes = tf.shape(labels)[-1]

    vectorized_embedding = tf.reshape(embeddings, [n_images, -1])
    vectorized_prototypes = _compute_prototypes(vectorized_embedding, labels)
    prototypes = tf.reshape(vectorized_prototypes, [n_classes] + feature_shape)
  else:
    prototypes = _compute_prototypes(embeddings, labels)

  return prototypes


class MetricLearner(learner_base.EpisodicLearner):
  """A learner that uses a learned distance metric to make predictions."""

  def __init__(self, **kwargs):
    super(MetricLearner, self).__init__(**kwargs)
    # `MetricLearner` subclasses don't require a pre-specified
    # output dimensionality.
    delattr(self, 'logit_dim')

  def forward_pass(self, data):
    """Embeds all (training and testing) images of the episode.

    Args:
      data: A `meta_dataset.providers.Episode` containing the data for the
        episode.

    Returns:
      The predictions for the query set within the episode.
    """
    # Compute the support set's mean and var and use these as the moments for
    # batch norm on the query set.
    support_embeddings_dict = self.embedding_fn(
        data.support_images,
        self.is_training,
        keep_spatial_dims=self.keep_spatial_dims)
    support_embeddings = support_embeddings_dict['embeddings']
    support_set_moments = None
    if not self.transductive_batch_norm:
      support_set_moments = support_embeddings_dict['moments']
    query_embeddings_dict = self.embedding_fn(
        data.query_images,
        self.is_training,
        moments=support_set_moments,
        keep_spatial_dims=self.keep_spatial_dims,
        backprop_through_moments=self.backprop_through_moments)
    query_embeddings = query_embeddings_dict['embeddings']

    query_logits = self.compute_logits(
        support_embeddings,
        query_embeddings,
        data.onehot_support_labels,
    )

    return query_logits

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    raise NotImplementedError('Abstract method.')


@gin.configurable
class PrototypicalNetworkLearner(MetricLearner):
  """A Prototypical Network."""
  keep_spatial_dims = False

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    """Computes the negative distances of each query point to each prototype."""

    # [num test images, 1, embedding size].
    query_embeddings = tf.expand_dims(query_embeddings, 1)

    prototypes = compute_prototypes(support_embeddings, onehot_support_labels)

    # [1, num_clases, embedding_size].
    prototypes = tf.expand_dims(prototypes, 0)

    # Squared euclidean distances between each test embedding / prototype pair.
    distances = tf.reduce_sum(tf.square(query_embeddings - prototypes), 2)
    return -distances


@gin.configurable
class MatchingNetworkLearner(MetricLearner):
  """A Matching Network."""
  keep_spatial_dims = False

  def __init__(self, exact_cosine_distance, **kwargs):
    """Initializes the Matching Networks instance.

    Args:
      exact_cosine_distance: If True then the cosine distance is used, otherwise
        the query set embeddings are left unnormalized when computing the dot
        product.
      **kwargs: Keyword arguments common to all `MetricLearner`s.
    """
    self.exact_cosine_distance = exact_cosine_distance
    super(MatchingNetworkLearner, self).__init__(**kwargs)

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    """Computes the class logits.

    Probabilities are computed as a weighted sum of one-hot encoded training
    labels. Weights for individual support/query pairs of examples are
    proportional to the (potentially semi-normalized) cosine distance between
    the embeddings of the two examples.

    Args:
      support_embeddings: A Tensor of size [num_support_images, embedding dim].
      query_embeddings: A Tensor of size [num_query_images, embedding dim].
      onehot_support_labels: A Tensor of size [batch size, way].

    Returns:
      The query set logits as a [num_query_images, way] matrix.
    """
    # Undocumented in the paper, but *very important*: *only* the support set
    # embeddings is L2-normalized, which means that the distance is not exactly
    # a cosine distance. For comparison we also allow for the actual cosine
    # distance to be computed, which is controlled with the
    # `exact_cosine_distance` instance attribute.
    support_embeddings = tf.nn.l2_normalize(support_embeddings, 1, epsilon=1e-3)
    if self.exact_cosine_distance:
      query_embeddings = tf.nn.l2_normalize(query_embeddings, 1, epsilon=1e-3)
    # [num_query_images, num_support_images]
    similarities = tf.matmul(
        query_embeddings, support_embeddings, transpose_b=True)
    attention = tf.nn.softmax(similarities)

    # [num_query_images, way]
    probs = tf.matmul(attention, tf.cast(onehot_support_labels, tf.float32))
    return tf.log(probs)


@gin.configurable
class RelationNetworkLearner(MetricLearner):
  """A Relation Network."""
  keep_spatial_dims = True

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    """Computes the relation score of each query example to each prototype."""
    # [n_test, 21, 21, n_features].
    query_embed_shape = query_embeddings.shape.as_list()
    n_feature = query_embed_shape[3]
    out_shape = query_embed_shape[1:3]
    n_test = tf.shape(query_embeddings)[0]

    # [n_test, num_clases, 21, 21, n_feature].
    # It is okay one of the elements in the list to be tensor.
    prototypes = compute_prototypes(support_embeddings, onehot_support_labels)

    prototype_extended = tf.tile(
        tf.expand_dims(prototypes, 0), [n_test, 1, 1, 1, 1])
    # [num_clases, n_test, 21, 21, n_feature].
    query_f_extended = tf.tile(
        tf.expand_dims(query_embeddings, 1),
        [1, tf.shape(onehot_support_labels)[-1], 1, 1, 1])
    relation_pairs = tf.concat((prototype_extended, query_f_extended), 4)
    # relation_pairs.shape.as_list()[-3:] == [-1] + out_shape + [n_feature*2]
    relation_pairs = tf.reshape(relation_pairs,
                                [-1] + out_shape + [n_feature * 2])
    relationnet_dict = functional_backbones.relation_module(relation_pairs)
    way = tf.shape(onehot_support_labels)[-1]
    relations = tf.reshape(relationnet_dict['output'], [-1, way])
    return relations

  def compute_loss(self, onehot_labels, predictions):
    """Computes the MSE loss of `predictions` with respect to `onehot_labels`.

    Args:
      onehot_labels: A `tf.Tensor` containing the the class labels; each vector
        along the class dimension should hold a valid probability distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` representing the average loss.
    """
    mse_loss = tf.losses.mean_squared_error(onehot_labels, predictions)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = mse_loss + regularization
    return loss
