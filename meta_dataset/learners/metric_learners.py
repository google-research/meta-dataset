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

import functools
import gin.tf
from meta_dataset import distribute_utils
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

    query_logits = self.compute_logits_for_episode(
        support_embeddings,
        query_embeddings,
        data,
    )

    return query_logits

  def compute_logits_for_episode(self, local_support_embeddings,
                                 local_query_embeddings, data):
    all_support_labels = distribute_utils.aggregate(data.onehot_support_labels)
    all_support_embeddings = distribute_utils.aggregate(
        local_support_embeddings)
    query_logits = self.compute_logits(
        all_support_embeddings,
        local_query_embeddings,
        all_support_labels,
    )

    return query_logits

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    raise NotImplementedError('Abstract method.')


@gin.configurable
class PrototypicalNetworkLearner(MetricLearner):
  """A Prototypical Network."""
  keep_spatial_dims = False

  def compute_logits(self,
                     support_embeddings,
                     query_embeddings,
                     onehot_support_labels,
                     cosine_distance=False):
    """Computes the negative distances of each query point to each prototype."""
    prototypes = compute_prototypes(support_embeddings, onehot_support_labels)

    if cosine_distance:
      query_embeddings = tf.nn.l2_normalize(query_embeddings, 1, epsilon=1e-3)
      prototypes = tf.nn.l2_normalize(prototypes, 1, epsilon=1e-3)
      logits = tf.matmul(query_embeddings, prototypes, transpose_b=True)
    else:
      # [num test images, 1, embedding size].
      query_embeddings = tf.expand_dims(query_embeddings, 1)

      # [1, num_clases, embedding_size].
      prototypes = tf.expand_dims(prototypes, 0)

      # Squared euclidean distance between each test embedding / prototype pair.
      distances = tf.reduce_sum(tf.square(query_embeddings - prototypes), 2)
      logits = -distances
    return logits


@gin.configurable
class CrossTransformerLearner(MetricLearner):
  """A CrossTransformer Network."""
  keep_spatial_dims = True

  def __init__(self,
               query_dim=128,
               val_dim=128,
               rematerialize=True,
               tformer_weight_decay=0.0,
               **kwargs):
    """Initializer including query and val dimensions."""
    super(CrossTransformerLearner, self).__init__(**kwargs)

    self.query_dim = query_dim
    self.val_dim = val_dim
    self.rematerialize = rematerialize
    self.tformer_weight_decay = tformer_weight_decay

  def compute_loss(self, onehot_labels, predictions):
    """Computes the CE loss of `predictions` with respect to `onehot_labels`.

    Args:
      onehot_labels: A `tf.Tensor` containing the the class labels; each vector
        along the (last) class dimension should represent a valid probability
        distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` representing the average loss.
    """
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=predictions,
        reduction=tf.losses.Reduction.NONE)

    return cross_entropy_loss

  def compute_logits_for_episode(self, support_embeddings, query_embeddings,
                                 data):
    """Compute CrossTransformer logits."""
    with tf.variable_scope('tformer_keys', reuse=tf.AUTO_REUSE):
      support_keys, key_params = functional_backbones.conv(
          support_embeddings, [1, 1],
          self.query_dim,
          1,
          weight_decay=self.tformer_weight_decay)
      query_queries, _ = functional_backbones.conv(
          query_embeddings, [1, 1],
          self.query_dim,
          1,
          params=key_params,
          weight_decay=self.tformer_weight_decay)

    with tf.variable_scope('tformer_values', reuse=tf.AUTO_REUSE):
      support_values, value_params = functional_backbones.conv(
          support_embeddings, [1, 1],
          self.val_dim,
          1,
          weight_decay=self.tformer_weight_decay)
      query_values, _ = functional_backbones.conv(
          query_embeddings, [1, 1],
          self.val_dim,
          1,
          params=value_params,
          weight_decay=self.tformer_weight_decay)

    onehot_support_labels = distribute_utils.aggregate(
        data.onehot_support_labels)
    support_keys = distribute_utils.aggregate(support_keys)
    support_values = distribute_utils.aggregate(support_values)

    labels = tf.argmax(onehot_support_labels, axis=1)
    if self.rematerialize:
      distances = self._get_dist_rematerialize(query_queries, query_values,
                                               support_keys, support_values,
                                               labels)
    else:
      distances = self._get_dist(query_queries, query_values, support_keys,
                                 support_values, labels)

    self.test_logits = -tf.transpose(distances)

    return self.test_logits

  @tf.custom_gradient
  def _get_dist_rematerialize(self, query_queries, query_values, support_keys,
                              support_values, labels):
    """Get distances between queries and query-aligned prototypes."""
    zero_dim = labels[0] - labels[0]

    def fwd_fn(query_queries_fwd, query_values_fwd, support_keys_fwd,
               support_values_fwd, labels_fwd):
      """CrossTransformer forward, using a while loop to save memory."""
      initial = (0,
                 tf.zeros([tf.reduce_max(labels) + 1, zero_dim],
                          dtype=query_queries_fwd.dtype))

      def loop_body(idx, dist):
        dist_new = self._get_dist(query_queries_fwd[idx:idx + 1],
                                  query_values_fwd[idx:idx + 1],
                                  support_keys_fwd, support_values_fwd,
                                  labels_fwd)
        dist = tf.concat([dist, dist_new], axis=1)
        return (idx + 1, dist)

      _, res = tf.while_loop(
          lambda x, _: x < tf.shape(query_queries_fwd)[0],
          loop_body,
          initial,
          parallel_iterations=1)
      return res

    def grad_fn(dy):
      """Compute gradients using a while loop to save memory."""
      support_keys_id = tf.identity(support_keys)
      support_values_id = tf.identity(support_values)
      initial = (0, tf.zeros(tf.shape(query_queries)[1:],
                             dtype=dy.dtype)[tf.newaxis, :][:zero_dim],
                 tf.zeros(tf.shape(query_values)[1:],
                          dtype=dy.dtype)[tf.newaxis, :][:zero_dim],
                 tf.zeros(tf.shape(support_keys_id), dtype=dy.dtype),
                 tf.zeros(tf.shape(support_values_id), dtype=dy.dtype))

      def loop_body(idx, qq_grad, qv_grad, sk_grad, sv_grad):
        """Compute gradients for a single query."""
        qq = query_queries[idx:idx + 1]
        qv = query_values[idx:idx + 1]
        x = self._get_dist(qq, qv, support_keys_id, support_values_id, labels)
        grads = tf.gradients(
            x, [qq, qv, support_keys_id, support_values_id],
            grad_ys=dy[:, idx:idx + 1])
        qq_grad = tf.concat([qq_grad, grads[0]], axis=0)
        qv_grad = tf.concat([qv_grad, grads[1]], axis=0)
        sk_grad += grads[2]
        sv_grad += grads[3]
        return (idx + 1, qq_grad, qv_grad, sk_grad, sv_grad)

      agg_grads = tf.while_loop(
          lambda *arg: arg[0] < tf.shape(query_queries)[0],
          loop_body,
          initial,
          parallel_iterations=1)
      return agg_grads[1:] + (None,)

    return fwd_fn(query_queries, query_values, support_keys, support_values,
                  labels), grad_fn

  def _get_dist(self, query_queries, query_values, support_keys, support_values,
                labels):
    """Get distances between queries and query-aligned prototypes."""
    # attended_values: [N_support, n_query, h_query, w_query, C]
    attended_values = self._attend(query_queries, support_keys, support_values,
                                   labels)
    # query_aligned_prototypes: [N_classes, n_query, h_query, w_query, C]
    query_aligned_prototypes = tf.math.unsorted_segment_sum(
        attended_values, labels,
        tf.reduce_max(labels) + 1)

    # (scaled) Euclidean distance
    shp = tf.shape(query_values)
    aligned_dist = tf.square(query_values[tf.newaxis, Ellipsis] -
                             query_aligned_prototypes)

    return tf.reduce_sum(aligned_dist, [2, 3, 4]) / tf.cast(
        shp[-3] * shp[-2], aligned_dist.dtype)

  def get_support_set_softmax(self, logits, class_ids):
    """Softmax normalize over the support set.

    Args:
      logits: [N_k, H*W, Q] dimensional tensor.
      class_ids: [N_k] tensor giving the support-set-id of each image.

    Returns:
      Softmax-ed x over the support set.

    softmax(x) = np.exp(x) / np.reduce_sum(np.exp(x), axis)
    """
    max_logit = tf.reduce_max(logits, axis=1, keepdims=True)
    max_logit = tf.math.unsorted_segment_max(max_logit, class_ids,
                                             tf.reduce_max(class_ids) + 1)
    max_logit = tf.gather(max_logit, class_ids)
    logits_reduc = logits - max_logit

    exp_x = tf.exp(logits_reduc)
    sum_exp_x = tf.reduce_sum(exp_x, axis=1, keepdims=True)
    sum_exp_x = tf.math.unsorted_segment_sum(sum_exp_x, class_ids,
                                             tf.reduce_max(class_ids) + 1)
    log_sum_exp_x = tf.log(sum_exp_x)
    log_sum_exp_x = tf.gather(log_sum_exp_x, class_ids)

    norm_logits = logits_reduc - log_sum_exp_x
    softmax = tf.exp(norm_logits)
    return softmax

  def _attend(self, query, key, value, key_class_id):
    """Transformer attention function."""
    with tf.name_scope('attend'):
      q_shape = tf.shape(query)
      v_shape = tf.shape(value)

      n_q = q_shape[0]
      h_q = q_shape[1]
      w_q = q_shape[2]
      d = q_shape[3]

      n_v = v_shape[0]
      h_v = v_shape[1]
      w_v = v_shape[2]
      c = v_shape[3]

      q = tf.reshape(query, [-1, d])  # [n_q*Hq*Wq, d]
      k = tf.reshape(key, [-1, d])

      # [n_v*Hv*Wv, d] x [Nq*Hq*Wq, d]  --> [n_v*Hv*Wv, Nq*Hq*Wq]
      logits = tf.matmul(k, q, transpose_b=True)
      d_scale = tf.rsqrt(tf.cast(d, logits.dtype))

      # logits: [n_v, Hv*Wv, n_q*Hq*Wq]
      logits = tf.reshape(d_scale * logits, [n_v, h_v * w_v, -1])

      # attn: [n_v, Hv*Wv, n_q*Hq*Wq]
      attn = self.get_support_set_softmax(logits, key_class_id)

      # aggregate:
      v = tf.reshape(value, [n_v, h_v * w_v, c])

      # [n_v, Hv*Wv, n_q*Hq*Wq] x [n_v, Hv*Wv, c]  --> [n_v, n_q*Hq*Wq, c]
      v_agg = tf.einsum('ijk,ijl->ikl', attn, v)
      v_agg = tf.reshape(v_agg, [n_v, n_q, h_q, w_q, c])
      v_agg.set_shape([None, None, None, None, value.shape[-1]])

      return v_agg  # [N_c, n_q, Hq, Wq, c]


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
    relationnet_dict = functional_backbones.relation_module(
        relation_pairs, self.is_training)
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


@gin.configurable
class DatasetConditionalPrototypicalNetworkLearner(PrototypicalNetworkLearner):
  """A Prototypical Network with Dataset Conditioning."""

  def __init__(self, num_sets, *args, **kwargs):
    del num_sets
    super(DatasetConditionalPrototypicalNetworkLearner,
          self).__init__(*args, **kwargs)

  def forward_pass(self, data, source, *args, **kwargs):
    if 'source_for_classifier' in kwargs:
      del kwargs['source_for_classifier']

    self.embedding_fn = functools.partial(
        self.embedding_fn, film_selector=source)

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
        params=support_embeddings_dict['params'],
        moments=support_set_moments,
        keep_spatial_dims=self.keep_spatial_dims)
    query_embeddings = query_embeddings_dict['embeddings']

    query_logits = self.compute_logits(
        support_embeddings,
        query_embeddings,
        data.onehot_support_labels,
        cosine_distance=True)

    return query_logits

