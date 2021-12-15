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
"""Metric-based learners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from meta_dataset.learners.experimental import base as learner_base
from meta_dataset.models.experimental import reparameterizable_backbones
import tensorflow as tf


def compute_prototypes(embeddings, onehot_labels):
  """Compute class prototypes over the last dimension of embeddings.

  Args:
    embeddings: Tensor of examples of shape [num_examples] + embedding_shape
    onehot_labels: Tensor of one-hot encoded labels of shape [num_examples,
      num_classes].

  Returns:
    prototypes: Tensor of class prototypes of shape [num_classes,
    embedding_size].
  """
  # Sums each class' embeddings. [num classes] + embedding shape.
  embedding_indices = 'klm'[:len(embeddings.shape) - 1]
  class_sums = tf.einsum('ij,i{0}->j{0}'.format(embedding_indices),
                         onehot_labels, embeddings)

  # The prototype of each class is the averaged embedding of its examples.
  class_num_images = tf.reduce_sum(input_tensor=onehot_labels, axis=0)  # [way].
  prototypes = tf.math.divide_no_nan(
      class_sums,
      tf.reshape(class_num_images, [-1] + [1] * (len(embeddings.shape) - 1)))

  return prototypes


@gin.configurable
def euclidean_distance(x, y):
  """Computes the Euclidean distance."""
  x = tf.expand_dims(x, 1)
  y = tf.expand_dims(y, 0)
  return tf.reduce_sum(input_tensor=tf.square(x - y), axis=2)


@gin.configurable
def exact_cosine_distance(x, y):
  """Computes the exact cosine distance."""
  x = tf.nn.l2_normalize(x, 1, epsilon=1e-3)
  y = tf.nn.l2_normalize(y, 1, epsilon=1e-3)
  return 1 - tf.matmul(x, y, transpose_b=True)


@gin.configurable
def partial_cosine_distance(x, y):
  """Computes the cosine distance without L2 normalization of x."""
  y = tf.nn.l2_normalize(y, 1, epsilon=1e-3)
  return 1 - tf.matmul(x, y, transpose_b=True)


@gin.configurable
class MetricLearner(learner_base.ExperimentalEpisodicLearner):
  """A metric learner."""

  def __init__(self, distance_metric, **kwargs):
    """Initializes a MetricLearner instance.

    Args:
      distance_metric: A callable; The distance metric used to compute class
        distances.
      **kwargs: Arguments common to all Learners.
    """
    self.distance_metric = distance_metric
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
    # TODO(eringrant): Implement non-transductive batch normalization (i.e.,
    # pass the support set statistics through the query set forward pass.
    support_embeddings = self.embedding_fn(data.support_images, training=True)
    query_embeddings = self.embedding_fn(data.query_images, training=True)

    # The class logits are the negative class distances.
    logits = -self.compute_class_distances(
        support_embeddings,
        data.onehot_support_labels,
        query_embeddings,
    )

    return logits

  # TODO(eringrant): Better name this method to accomodate `MatchingNetwork`,
  # which computes an attention-weighted combination of query set items.
  def compute_class_distances(self, support_embeddings, onehot_support_labels,
                              query_embeddings):
    raise NotImplementedError


@gin.configurable
class PrototypicalNetwork(MetricLearner):
  """A Prototypical Network."""
  keep_spatial_dims = False

  def __init__(self, distance_metric=euclidean_distance, **kwargs):
    """Initializes a PrototypicalNetwork instance.

    Args:
      distance_metric: A callable; The distance metric used to compute class
        distances.
      **kwargs: Arguments common to all `MetricLearner`s.
    """
    super(PrototypicalNetwork, self).__init__(
        distance_metric=distance_metric, **kwargs)

  def compute_class_distances(self, support_embeddings, onehot_support_labels,
                              query_embeddings):
    """Returns the distances of each query example to each class prototype."""
    prototypes = compute_prototypes(support_embeddings, onehot_support_labels)
    return self.distance_metric(query_embeddings, prototypes)


@gin.configurable
class MatchingNetwork(MetricLearner):
  """A Matching Network."""
  keep_spatial_dims = False

  def __init__(self, distance_metric=partial_cosine_distance, **kwargs):
    """Initializes a MatchingNetwork instance.

    Undocumented in the MatchingNets paper but *very important* is that *only*
    the support set embeddings are L2-normalized when computing the cosine
    distance, which means that the distance is not exactly a cosine distance.
    For comparison, the actual cosine distance can be computed by setting the
    `distance_metric` to `exact_cosine_distance`.

    Args:
      distance_metric: A callable; The distance metric used to compute class
        distances.
      **kwargs: Arguments common to all `MetricLearner`s.
    """
    super(MatchingNetwork, self).__init__(
        distance_metric=distance_metric, **kwargs)

  def compute_class_distances(self, support_embeddings, onehot_support_labels,
                              query_embeddings):
    """Returns the weighted distance of each query to each support example.

    Args:
      support_embeddings: Tensor of examples of shape [num_examples,
        embedding_dim] or [num_examples, spatial_dim, spatial_dim, num_filters].
      onehot_support_labels: Tensor of targets of shape [num_examples,
        num_classes].
      query_embeddings: Tensor of examples of shape [num_examples,
        embedding_dim] or [num_examples, spatial_dim, spatial_dim, num_filters].

    Returns:
      Class log-probabilities computed as a weighted sum of one-hot encoded
      training labels. Weights for individual support-query pairs of examples
      are proportional to the distance between the embeddings of the two
      examples.
    """
    # [num_query_images, num_support_images]
    similarities = 1 - self.distance_metric(query_embeddings,
                                            support_embeddings)
    attention = tf.nn.softmax(similarities)

    # [num_query_images, way]
    probs = tf.matmul(attention,
                      tf.cast(onehot_support_labels, dtype=tf.float32))
    return tf.math.log(probs)


@gin.configurable
class RelationNetwork(MetricLearner):
  """A Relation Network."""
  keep_spatial_dims = True

  def __init__(self,
               relation_module_cls=reparameterizable_backbones.RelationModule,
               **kwargs):
    """Initializes a RelationNetwork instance.

    Args:
      relation_module_cls: The class of a Module that computes relations from
        pairs.
      **kwargs: Arguments common to all `MetricLearner`s.

    Raises:
      `ValueError` if `RelationNetwork.embedding_fn` is not an instance of
      `reparameterizable_backbones.RelationNetConvNet`.
    """
    super(RelationNetwork, self).__init__(**kwargs)
    if not isinstance(self.embedding_fn,
                      reparameterizable_backbones.RelationNetConvNet):
      raise ValueError('`embedding_fn` for `RelationNetwork` must be a '
                       '`reparameterizable_backbones.RelationNetConvNet`, as '
                       'this backbone omits dimensionality reductions.')

    # Instantiate the relation module and its parameters.
    self.relation_module_fn = relation_module_cls()

  def build(self):
    """Instantiate the parameters belonging to this `RelationNetwork`."""
    super(RelationNetwork, self).build()
    if not self.relation_module_fn.built:
      self.relation_module_fn.build(self.embedding_shape)
    self.output_shape = self.relation_module_fn.compute_output_shape(
        self.embedding_shape)

  def compute_class_distances(self, support_embeddings, onehot_support_labels,
                              query_embeddings):
    """Return the relation score of each query example to each prototype."""
    # `query_embeddings` is [num_examples, 21, 21, num_features].
    out_shape = query_embeddings.shape.as_list()[1:]
    num_features = out_shape[-1]
    num_query_examples = tf.shape(input=query_embeddings)[0]

    # [num_classes, 19, 19, num_features].
    prototypes = compute_prototypes(support_embeddings, onehot_support_labels)

    # [num_classes, 19, 19, num_features].
    prototype_extended = tf.tile(
        tf.expand_dims(prototypes, 0),
        [num_query_examples] + [1] * (1 + len(out_shape)))

    # [num_query_examples, 19, 19, num_features].
    way = onehot_support_labels.shape.as_list()[-1]
    query_extended = tf.tile(
        tf.expand_dims(query_embeddings, 1), [1, way] + [1] * len(out_shape))
    relation_pairs = tf.concat((prototype_extended, query_extended),
                               len(out_shape) + 1)

    # relation_pairs.shape.as_list()[-3:] == [-1] + out_shape + [num_features*2]
    relation_pairs = tf.reshape(relation_pairs,
                                [-1] + out_shape[:-1] + [num_features * 2])

    return tf.reshape(self.relation_module_fn(relation_pairs), [-1, way])
