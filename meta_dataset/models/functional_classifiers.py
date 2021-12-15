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

# Lint as: python2,python3
"""Classifier-related code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from meta_dataset.models import functional_backbones
import tensorflow.compat.v1 as tf


def linear_classifier_forward_pass(embeddings, w_fc, b_fc, cosine_classifier,
                                   cosine_logits_multiplier, use_weight_norm):
  """Passes embeddings through the linear layer defined by w_fc and b_fc.

  Args:
    embeddings: A Tensor of size [batch size, embedding dim].
    w_fc: A Tensor of size [embedding dim, num outputs].
    b_fc: Either None, or a Tensor of size [num outputs] or []. If
      cosine_classifier is False, it can not be None.
    cosine_classifier: A bool. If true, a cosine classifier is used which does
      not require the bias b_fc.
    cosine_logits_multiplier: A float. Only used if cosine_classifier is True,
      and multiplies the resulting logits.
    use_weight_norm: A bool. Whether weight norm was used. If so, then if using
      cosine classifier, normalize only the embeddings but not the weights.

  Returns:
    logits: A Tensor of size [batch size, num outputs].
  """
  if cosine_classifier:
    # Each column of the weight matrix may be interpreted as a class
    # representation (of the same dimenionality as the embedding space). The
    # logit for an embedding vector belonging to that class is the cosine
    # similarity between that embedding and that class representation.
    embeddings = tf.nn.l2_normalize(embeddings, axis=1, epsilon=1e-3)
    if not use_weight_norm:
      # Only normalize the weights if weight norm was not used.
      w_fc = tf.nn.l2_normalize(w_fc, axis=0, epsilon=1e-3)
    logits = tf.matmul(embeddings, w_fc)
    # Scale the logits as passing numbers in [-1, 1] to softmax is not very
    # expressive.
    logits *= cosine_logits_multiplier
  else:
    assert b_fc is not None
    logits = tf.matmul(embeddings, w_fc) + b_fc
  return logits


@gin.configurable
def linear_classifier(embeddings, num_classes, cosine_classifier,
                      cosine_logits_multiplier, use_weight_norm, weight_decay):
  """Forward pass through a linear classifier, or possibly a cosine classifier.

  Args:
    embeddings: A Tensor of size [batch size, embedding dim].
    num_classes: An integer; the dimension of the classification.
    cosine_classifier: A bool. If true, a cosine classifier is used, which does
      not require a bias.
    cosine_logits_multiplier: A float. Only used if cosine_classifier is True,
      and multiplies the resulting logits.
    use_weight_norm: A bool. Whether weight norm was used. If so, then if using
      cosine classifier, normalize only the embeddings but not the weights.
    weight_decay: A float; the scalar multiple on the L2 regularization of the
      weight matrix.

  Returns:
    logits: A Tensor of size [batch size, num outputs].
  """

  embedding_dims = embeddings.get_shape().as_list()[-1]

  if use_weight_norm:
    # A variable to keep track of whether the initialization has already
    # happened.
    data_dependent_init_done = tf.get_variable(
        'data_dependent_init_done',
        initializer=0,
        dtype=tf.int32,
        trainable=False)

    w_fc = tf.get_variable(
        'w_fc', [embedding_dims, num_classes],
        initializer=tf.random_normal_initializer(0, 0.05),
        trainable=True)
    # This init is temporary as it needs to be done in a data-dependent way.
    # It will be overwritten during the first forward pass through this layer.
    g = tf.get_variable(
        'g',
        dtype=tf.float32,
        initializer=tf.ones([num_classes]),
        trainable=True)
    b_fc = None
    if not cosine_classifier:
      # Also initialize a bias.
      b_fc = tf.get_variable(
          'b_fc', initializer=tf.zeros([num_classes]), trainable=True)

    def _do_data_dependent_init():
      """Returns ops for the data-dependent init of g and maybe b_fc."""
      w_fc_normalized = tf.nn.l2_normalize(w_fc.read_value(), [0])
      output_init = tf.matmul(embeddings, w_fc_normalized)
      mean_init, var_init = tf.nn.moments(output_init, [0])
      # Data-dependent init values.
      g_init_value = 1. / tf.sqrt(var_init + 1e-10)
      ops = [tf.assign(g, g_init_value)]
      if not cosine_classifier:
        # Also initialize a bias in a data-dependent way.
        b_fc_init_value = -mean_init * g_init_value
        ops.append(tf.assign(b_fc, b_fc_init_value))
      # Mark that the data-dependent initialization is done to prevent it from
      # happening again in the future.
      ops.append(tf.assign(data_dependent_init_done, 1))
      return tf.group(*ops)

    # Possibly perform data-dependent init (if it hasn't been done already).
    init_op = tf.cond(
        tf.equal(data_dependent_init_done, 0), _do_data_dependent_init,
        tf.no_op)

    with tf.control_dependencies([init_op]):
      # Apply weight normalization.
      w_fc *= g / tf.sqrt(tf.reduce_sum(tf.square(w_fc), [0]))
      # Forward pass through the layer defined by w_fc and b_fc.
      logits = linear_classifier_forward_pass(embeddings, w_fc, b_fc,
                                              cosine_classifier,
                                              cosine_logits_multiplier, True)

  else:
    # No weight norm.
    w_fc = functional_backbones.weight_variable([embedding_dims, num_classes],
                                                weight_decay=weight_decay)
    b_fc = None
    if not cosine_classifier:
      # Also initialize a bias.
      b_fc = functional_backbones.bias_variable([num_classes])
    # Forward pass through the layer defined by w_fc and b_fc.
    logits = linear_classifier_forward_pass(embeddings, w_fc, b_fc,
                                            cosine_classifier,
                                            cosine_logits_multiplier, False)
  return logits


@gin.configurable
def separate_head_linear_classifier(embeddings, num_classes, dataset_idx,
                                    start_idx, cosine_classifier,
                                    cosine_logits_multiplier, learnable_scale,
                                    weight_decay):
  """A linear classifier with num_sets heads, for different datasets.

  Args:
    embeddings: A Tensor of size [batch size, embedding dim].
    num_classes: A list of integers; the dimension of the classifier layers of
      the different heads.
    dataset_idx: An int Tensor. The index of the dataset head to use.
    start_idx: An int Tensor. The index of the first class of the given dataset.
    cosine_classifier: A bool. If true, a cosine classifier is used, which does
      not require a bias.
    cosine_logits_multiplier: A float. Only used if cosine_classifier is True,
      and multiplies the resulting logits.
    learnable_scale: A bool. Whether to make the cosine_logits_multiplier a
      learnable parameter. Only applies if cosine_classifier is True.
    weight_decay: A float; the scalar multiple on the L2 regularization of the
      weight matrix.

  Returns:
    logits: A Tensor of size [batch size, num outputs].
  """
  if not cosine_classifier:
    raise NotImplementedError('`separate_head_linear_classifier` currently '
                              'only supports `cosine_classifier` True.')

  if learnable_scale:
    cosine_logits_multiplier = tf.get_variable(
        'cosine_scale',
        initializer=cosine_logits_multiplier,
        dtype=tf.float32,
        trainable=True)

  embedding_dims = embeddings.get_shape().as_list()[-1]
  w_fc = functional_backbones.weight_variable(
      [embedding_dims, sum(num_classes)], weight_decay=weight_decay)

  # Select the output "head" to use in the forward pass.
  dataset_num_classes = tf.gather(num_classes, dataset_idx)
  w_fc = w_fc[:, start_idx:start_idx + dataset_num_classes]

  logits = linear_classifier_forward_pass(embeddings, w_fc, None,
                                          cosine_classifier,
                                          cosine_logits_multiplier, False)
  return logits

