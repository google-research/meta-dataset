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
"""Learner related code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from absl import logging
import gin.tf
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

MAX_WAY = 50  # The maximum number of classes we will see in any batch.


def conv2d(x, w, stride=1, b=None, padding='SAME'):
  """conv2d returns a 2d convolution layer with full stride."""
  h = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
  if b is not None:
    h += b
  return h


def relu(x, use_bounded_activation=False):
  if use_bounded_activation:
    return tf.nn.relu6(x)
  else:
    return tf.nn.relu(x)


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


# TODO(tylerzhu): Accumulate batch norm statistics (moving {var, mean})
# during training and use them during testing. However need to be careful
# about leaking information across episodes.
# Note: we should use ema object to accumulate the statistics for compatibility
# with TF Eager.
def bn(x, params=None, moments=None, backprop_through_moments=True):
  """Batch normalization.

  The usage should be as follows: If x is the support images, moments should be
  None so that they are computed from the support set examples. On the other
  hand, if x is the query images, the moments argument should be used in order
  to pass in the mean and var that were computed from the support set.

  Args:
    x: inputs.
    params: None or a dict containing the values of the offset and scale params.
    moments: None or a dict containing the values of the mean and var to use for
      batch normalization.
    backprop_through_moments: Whether to allow gradients to flow through the
      given support set moments. Only applies to non-transductive batch norm.

  Returns:
    output: The result of applying batch normalization to the input.
    params: The updated params.
    moments: The updated moments.
  """
  params_keys, params_vars, moments_keys, moments_vars = [], [], [], []

  with tf.variable_scope('batch_norm'):
    scope_name = tf.get_variable_scope().name
    if moments is None:
      # If not provided, compute the mean and var of the current batch.
      mean, var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
    else:
      if backprop_through_moments:
        mean = moments[scope_name + '/mean']
        var = moments[scope_name + '/var']
      else:
        # This variant does not yield good resutls.
        mean = tf.stop_gradient(moments[scope_name + '/mean'])
        var = tf.stop_gradient(moments[scope_name + '/var'])

    moments_keys += [scope_name + '/mean']
    moments_vars += [mean]
    moments_keys += [scope_name + '/var']
    moments_vars += [var]

    if params is None:
      offset = tf.get_variable(
          'offset',
          shape=mean.get_shape().as_list(),
          initializer=tf.initializers.zeros())
      scale = tf.get_variable(
          'scale',
          shape=var.get_shape().as_list(),
          initializer=tf.initializers.ones())
    else:
      offset = params[scope_name + '/offset']
      scale = params[scope_name + '/scale']

    params_keys += [scope_name + '/offset']
    params_vars += [offset]
    params_keys += [scope_name + '/scale']
    params_vars += [scale]

    output = tf.nn.batch_normalization(x, mean, var, offset, scale, 0.00001)
    params = collections.OrderedDict(zip(params_keys, params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))
    return output, params, moments


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.initializers.truncated_normal(stddev=0.1)
  return tf.get_variable(
      'weight', shape=shape, initializer=initial, regularizer=tf.nn.l2_loss)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.initializers.constant(0.1)
  return tf.get_variable('bias', shape=shape, initializer=initial)


def dense(x, output_size, activation_fn=tf.nn.relu, params=None):
  """Fully connected layer implementation.

  Args:
    x: tf.Tensor, input.
    output_size: int, number features in  the fully connected layer.
    activation_fn: function, to process pre-activations, namely x*w+b.
    params: None or a dict containing the values of the wieght and bias params.
      If None, default variables are used.

  Returns:
    output: The result of applying batch normalization to the input.
    params: dict, that includes parameters used during the calculation.
  """
  with tf.variable_scope('dense'):
    scope_name = tf.get_variable_scope().name

    if len(x.shape) > 2:
      x = tf.layers.flatten(x),
    input_size = x.get_shape().as_list()[-1]

    w_name = scope_name + '/kernel'
    b_name = scope_name + '/bias'
    if params is None:
      w = weight_variable([input_size, output_size])
      b = bias_variable([output_size])
    else:
      w = params[w_name]
      b = params[b_name]

  x = tf.nn.xw_plus_b(x, w, b)
  params = collections.OrderedDict(zip([w_name, b_name], [w, b]))
  x = activation_fn(x)
  return x, params


def conv(x, conv_size, depth, stride, padding='SAME', params=None):
  """A block that performs convolution."""
  params_keys, params_vars = [], []
  scope_name = tf.get_variable_scope().name
  input_depth = x.get_shape().as_list()[-1]
  if params is None:
    w_conv = weight_variable([conv_size[0], conv_size[1], input_depth, depth])
  else:
    w_conv = params[scope_name + '/kernel']

  params_keys += [scope_name + '/kernel']
  params_vars += [w_conv]

  x = conv2d(x, w_conv, stride=stride, padding=padding)
  params = collections.OrderedDict(zip(params_keys, params_vars))

  return x, params


def conv_bn(x,
            conv_size,
            depth,
            stride,
            padding='SAME',
            params=None,
            moments=None,
            backprop_through_moments=True):
  """A block that performs convolution, followed by batch-norm."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  x, conv_params = conv(
      x, conv_size, depth, stride, padding=padding, params=params)
  params_keys.extend(conv_params.keys())
  params_vars.extend(conv_params.values())

  x, bn_params, bn_moments = bn(
      x,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments)
  params_keys.extend(bn_params.keys())
  params_vars.extend(bn_params.values())
  moments_keys.extend(bn_moments.keys())
  moments_vars.extend(bn_moments.values())

  params = collections.OrderedDict(zip(params_keys, params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))

  return x, params, moments


def bottleneck(x,
               depth,
               stride=1,
               params=None,
               moments=None,
               use_project=False,
               backprop_through_moments=True,
               use_bounded_activation=False):
  """ResNet18 residual block."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []  # means and vars of different layers.
  with tf.variable_scope('conv1'):
    h, conv_bn_params, conv_bn_moments = conv_bn(
        x, [3, 3],
        depth[0],
        stride,
        params=params,
        moments=moments,
        backprop_through_moments=backprop_through_moments)
    params_keys.extend(conv_bn_params.keys())
    params_vars.extend(conv_bn_params.values())
    moments_keys.extend(conv_bn_moments.keys())
    moments_vars.extend(conv_bn_moments.values())

    h = relu(h, use_bounded_activation=use_bounded_activation)

  with tf.variable_scope('conv2'):
    h, conv_bn_params, conv_bn_moments = conv_bn(
        h, [3, 3],
        depth[1],
        stride=1,
        params=params,
        moments=moments,
        backprop_through_moments=backprop_through_moments)
    if use_bounded_activation:
      h = tf.clip_by_value(h, -6.0, 6.0)

    params_keys.extend(conv_bn_params.keys())
    params_vars.extend(conv_bn_params.values())
    moments_keys.extend(conv_bn_moments.keys())
    moments_vars.extend(conv_bn_moments.values())

  with tf.variable_scope('identity'):
    if use_project:
      with tf.variable_scope('projection_conv'):
        x, conv_bn_params, conv_bn_moments = conv_bn(
            x, [1, 1],
            depth[1],
            stride,
            params=params,
            moments=moments,
            backprop_through_moments=backprop_through_moments)
        params_keys.extend(conv_bn_params.keys())
        params_vars.extend(conv_bn_params.values())
        moments_keys.extend(conv_bn_moments.keys())
        moments_vars.extend(conv_bn_moments.values())
    x = relu(x + h, use_bounded_activation=use_bounded_activation)

  params = collections.OrderedDict(zip(params_keys, params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))
  return x, params, moments


def _resnet(x,
            is_training,
            scope,
            reuse=tf.AUTO_REUSE,
            params=None,
            moments=None,
            backprop_through_moments=True,
            use_bounded_activation=False,
            keep_spatial_dims=False):
  """A ResNet18 network."""
  # `is_training` will be used when start to use moving {var, mean} in batch
  # normalization. This refers to 'meta-training'.
  del is_training
  x = tf.stop_gradient(x)
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  with tf.variable_scope(scope, reuse=reuse):
    # We use DeepLab feature alignment rule to determine the input size.
    # Since the image size in the meta-dataset pipeline is a multiplier of 42,
    # e.g., [42, 84, 168], we align them to the closest sizes that conform to
    # the alignment rule and at the same time are larger. They are [65, 97, 193]
    # respectively. The aligned image size for 224 used in the ResNet work is
    # 225.
    #
    # References:
    # 1. ResNet https://arxiv.org/abs/1512.03385
    # 2. DeepLab https://arxiv.org/abs/1606.00915
    size = tf.cast(tf.shape(x)[1], tf.float32)
    aligned_size = tf.cast(tf.ceil(size / 32.0), tf.int32) * 32 + 1
    x = tf.image.resize_bilinear(
        x, size=[aligned_size, aligned_size], align_corners=True)

    with tf.variable_scope('conv1'):
      x, conv_bn_params, conv_bn_moments = conv_bn(
          x, [7, 7],
          64,
          2,
          params=params,
          moments=moments,
          backprop_through_moments=backprop_through_moments)
      params_keys.extend(conv_bn_params.keys())
      params_vars.extend(conv_bn_params.values())
      moments_keys.extend(conv_bn_moments.keys())
      moments_vars.extend(conv_bn_moments.values())

      x = relu(x, use_bounded_activation=use_bounded_activation)

    def _bottleneck(x, i, depth, params, moments, stride=2):
      """Wrapper for bottleneck."""
      output_stride = stride if i == 0 else 1
      use_project = True if i == 0 else False
      x, bottleneck_params, bottleneck_moments = bottleneck(
          x, (depth, depth),
          output_stride,
          params=params,
          moments=moments,
          use_project=use_project,
          backprop_through_moments=backprop_through_moments)
      return x, bottleneck_params, bottleneck_moments

    with tf.variable_scope('conv2_x'):
      x = tf.nn.max_pool(
          x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 64, params, moments, stride=1)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv3_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 128, params, moments)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv4_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 256, params, moments)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv5_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 512, params, moments)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    # x.shape: [?, 1, 1, 512]
    if not keep_spatial_dims:
      x = tf.reshape(x, [-1, 512])
    params = collections.OrderedDict(zip(params_keys, params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))

    return_dict = {'embeddings': x, 'params': params, 'moments': moments}
    return return_dict


def resnet(x,
           is_training,
           params=None,
           moments=None,
           reuse=tf.AUTO_REUSE,
           scope='resnet18',
           backprop_through_moments=True,
           use_bounded_activation=False,
           keep_spatial_dims=False):
  return _resnet(
      x,
      is_training,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      keep_spatial_dims=keep_spatial_dims)


def wide_resnet_block(x,
                      depth,
                      stride=1,
                      params=None,
                      moments=None,
                      use_project=False,
                      backprop_through_moments=True,
                      use_bounded_activation=False):
  """Wide ResNet residual block."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  with tf.variable_scope('conv1'):
    bn_1, bn_params, bn_moments = bn(
        x,
        params=params,
        moments=moments,
        backprop_through_moments=backprop_through_moments)
    params_keys.extend(bn_params.keys())
    params_vars.extend(bn_params.values())
    moments_keys.extend(bn_moments.keys())
    moments_vars.extend(bn_moments.values())

    out_1 = relu(bn_1, use_bounded_activation=use_bounded_activation)

    h_1, conv_params = conv(out_1, [3, 3], depth, stride, params=params)
    params_keys.extend(conv_params.keys())
    params_vars.extend(conv_params.values())
  with tf.variable_scope('conv2'):
    bn_2, bn_params, bn_moments = bn(
        h_1,
        params=params,
        moments=moments,
        backprop_through_moments=backprop_through_moments)
    params_keys.extend(bn_params.keys())
    params_vars.extend(bn_params.values())
    moments_keys.extend(bn_moments.keys())
    moments_vars.extend(bn_moments.values())

    out_2 = relu(bn_2, use_bounded_activation=use_bounded_activation)

    h_2, conv_params = conv(out_2, [3, 3], depth, stride=1, params=params)
    params_keys.extend(conv_params.keys())
    params_vars.extend(conv_params.values())

  h = h_2
  if use_bounded_activation:
    h = tf.clip_by_value(h, -6, 6)

  with tf.variable_scope('identity'):
    if use_project:
      with tf.variable_scope('projection_conv'):
        x, conv_params = conv(out_1, [1, 1], depth, stride, params=params)
        params_keys.extend(conv_params.keys())
        params_vars.extend(conv_params.values())

  params = collections.OrderedDict(zip(params_keys, params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))

  if use_bounded_activation:
    out = tf.clip_by_value(x + h, -6, 6)
  else:
    out = x + h
  return out, params, moments


def _wide_resnet(x,
                 is_training,
                 scope,
                 n,
                 k,
                 reuse=tf.AUTO_REUSE,
                 params=None,
                 moments=None,
                 backprop_through_moments=True,
                 use_bounded_activation=False,
                 keep_spatial_dims=False):
  """A wide ResNet."""
  # `is_training` will be used when start to use moving {var, mean} in batch
  # normalization.
  del is_training
  widths = [i * k for i in (16, 32, 64)]
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []

  def _update_params_lists(params_dict, params_keys, params_vars):
    params_keys.extend(params_dict.keys())
    params_vars.extend(params_dict.values())

  def _update_moments_lists(moments_dict, moments_keys, moments_vars):
    moments_keys.extend(moments_dict.keys())
    moments_vars.extend(moments_dict.values())

  with tf.variable_scope(scope, reuse=reuse):
    with tf.variable_scope('conv1'):
      x, conv_params = conv(x, [3, 3], 16, 1, params=params)
      _update_params_lists(conv_params, params_keys, params_vars)

    def _wide_resnet_block(x, depths, stride, use_project, moments):
      """Wrapper for a wide resnet block."""
      x, block_params, block_moments = wide_resnet_block(
          x,
          depths,
          stride=stride,
          params=params,
          moments=moments,
          use_project=use_project,
          backprop_through_moments=backprop_through_moments,
          use_bounded_activation=use_bounded_activation)
      return x, block_params, block_moments

    with tf.variable_scope('conv2_x'):
      with tf.variable_scope('wide_block_0'):
        if widths[0] == 16:
          use_project = False
        else:
          use_project = True
        x, block_params, block_moments = _wide_resnet_block(
            x, widths[0], 1, use_project, moments=moments)
        _update_params_lists(block_params, params_keys, params_vars)
        _update_moments_lists(block_moments, moments_keys, moments_vars)
      for i in range(1, n):
        with tf.variable_scope('wide_block_%d' % i):
          x, block_params, block_moments = _wide_resnet_block(
              x, widths[0], 1, use_project, moments=moments)
          _update_params_lists(block_params, params_keys, params_vars)
          _update_moments_lists(block_moments, moments_keys, moments_vars)

    with tf.variable_scope('conv3_x'):
      with tf.variable_scope('wide_block_0'):
        x, block_params, block_moments = _wide_resnet_block(
            x, widths[1], 2, True, moments=moments)
        _update_params_lists(block_params, params_keys, params_vars)
        _update_moments_lists(block_moments, moments_keys, moments_vars)
      for i in range(1, n):
        with tf.variable_scope('wide_block_%d' % i):
          x, block_params, block_moments = _wide_resnet_block(
              x, widths[1], 1, use_project, moments=moments)
          _update_params_lists(block_params, params_keys, params_vars)
          _update_moments_lists(block_moments, moments_keys, moments_vars)

    with tf.variable_scope('conv4_x'):
      with tf.variable_scope('wide_block_0'):
        x, block_params, block_moments = _wide_resnet_block(
            x, widths[2], 2, True, moments=moments)
        _update_params_lists(block_params, params_keys, params_vars)
        _update_moments_lists(block_moments, moments_keys, moments_vars)
      for i in range(1, n):
        with tf.variable_scope('wide_block_%d' % i):
          x, block_params, block_moments = _wide_resnet_block(
              x, widths[2], 1, use_project, moments=moments)
          _update_params_lists(block_params, params_keys, params_vars)
          _update_moments_lists(block_moments, moments_keys, moments_vars)

    with tf.variable_scope('embedding_layer'):
      x, bn_params, bn_moments = bn(
          x,
          params=params,
          moments=moments,
          backprop_through_moments=backprop_through_moments)
      _update_params_lists(bn_params, params_keys, params_vars)
      _update_moments_lists(bn_moments, moments_keys, moments_vars)

      x = relu(x, use_bounded_activation=use_bounded_activation)
    img_w, img_h = x.get_shape().as_list()[1:3]
    x = tf.nn.avg_pool(
        x, ksize=[1, img_w, img_h, 1], strides=[1, 1, 1, 1], padding='VALID')
    # x.shape: [X, 1, 1, 128]
    if not keep_spatial_dims:
      x = tf.reshape(x, [-1, widths[2]])
    params = collections.OrderedDict(zip(params_keys, params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))

    return_dict = {'embeddings': x, 'params': params, 'moments': moments}
    return return_dict


def wide_resnet(x,
                is_training,
                params=None,
                moments=None,
                reuse=tf.AUTO_REUSE,
                scope='wide_resnet',
                backprop_through_moments=True,
                use_bounded_activation=False,
                keep_spatial_dims=False):
  return _wide_resnet(
      x,
      is_training,
      scope,
      2,
      2,
      reuse=reuse,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      keep_spatial_dims=keep_spatial_dims)


def _four_layer_convnet(inputs,
                        scope,
                        reuse=tf.AUTO_REUSE,
                        params=None,
                        moments=None,
                        depth_multiplier=1.0,
                        backprop_through_moments=True,
                        use_bounded_activation=False,
                        keep_spatial_dims=False):
  """A four-layer-convnet architecture."""
  layer = tf.stop_gradient(inputs)
  model_params_keys, model_params_vars = [], []
  moments_keys, moments_vars = [], []

  with tf.variable_scope(scope, reuse=reuse):
    for i in range(4):
      with tf.variable_scope('layer_{}'.format(i), reuse=reuse):
        depth = int(64 * depth_multiplier)
        layer, conv_bn_params, conv_bn_moments = conv_bn(
            layer, [3, 3],
            depth,
            stride=1,
            params=params,
            moments=moments,
            backprop_through_moments=backprop_through_moments)
        model_params_keys.extend(conv_bn_params.keys())
        model_params_vars.extend(conv_bn_params.values())
        moments_keys.extend(conv_bn_moments.keys())
        moments_vars.extend(conv_bn_moments.values())

      if use_bounded_activation:
        layer = tf.nn.relu6(layer)
      else:
        layer = tf.nn.relu(layer)
      layer = tf.layers.max_pooling2d(layer, [2, 2], 2)
      logging.info('Output of block %d: %s', i, layer.shape)

    model_params = collections.OrderedDict(
        zip(model_params_keys, model_params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))
    if not keep_spatial_dims:
      layer = tf.layers.flatten(layer)
    return_dict = {
        'embeddings': layer,
        'params': model_params,
        'moments': moments
    }

    return return_dict


def _relation_net(inputs,
                  scope,
                  reuse=tf.AUTO_REUSE,
                  params=None,
                  moments=None,
                  depth_multiplier=1.0,
                  backprop_through_moments=True,
                  use_bounded_activation=False):
  """A 2-layer-convnet architecture with fully connected layers."""
  model_params_keys, model_params_vars = [], []
  moments_keys, moments_vars = [], []
  layer = inputs
  with tf.variable_scope(scope, reuse=reuse):
    for i in range(2):
      with tf.variable_scope('layer_{}'.format(i), reuse=reuse):
        depth = int(64 * depth_multiplier)
        # Note that original has `valid` padding where we use `same`.
        layer, conv_bn_params, conv_bn_moments = conv_bn(
            layer, [3, 3],
            depth,
            stride=1,
            params=params,
            moments=moments,
            backprop_through_moments=backprop_through_moments)
        model_params_keys.extend(conv_bn_params.keys())
        model_params_vars.extend(conv_bn_params.values())
        moments_keys.extend(conv_bn_moments.keys())
        moments_vars.extend(conv_bn_moments.values())

      layer = relu(layer, use_bounded_activation=use_bounded_activation)
      # This is a hacky way preventing max pooling if the spatial dimensions
      # are already reduced.
      if layer.shape[1] > 1:
        layer = tf.layers.max_pooling2d(layer, [2, 2], 2)
      tf.logging.info('Output of block %d: %s' % (i, layer.shape))

    layer = tf.layers.flatten(layer)
    relu_activation_fn = functools.partial(
        relu, use_bounded_activation=use_bounded_activation)
    with tf.variable_scope('layer_2_fc', reuse=reuse):
      layer, dense_params = dense(layer, 8, activation_fn=relu_activation_fn)
      tf.logging.info('Output layer_2_fc: %s' % layer.shape)
      model_params_keys.extend(dense_params.keys())
      model_params_vars.extend(dense_params.values())
    with tf.variable_scope('layer_3_fc', reuse=reuse):
      output, dense_params = dense(layer, 1, activation_fn=tf.nn.sigmoid)
      tf.logging.info('Output layer_3_fc: %s' % output.shape)
      model_params_keys.extend(dense_params.keys())
      model_params_vars.extend(dense_params.values())

    model_params = collections.OrderedDict(
        zip(model_params_keys, model_params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))
    return_dict = {'output': output, 'params': model_params, 'moments': moments}

    return return_dict


def relationnet_embedding(inputs,
                          is_training,
                          params=None,
                          moments=None,
                          depth_multiplier=1.0,
                          reuse=tf.AUTO_REUSE,
                          scope='relationnet_convnet',
                          backprop_through_moments=True,
                          use_bounded_activation=False,
                          keep_spatial_dims=False):
  """A 4-layer-convnet architecture for relationnet embedding.

  This is almost like the `learner.four_layer_convnet` embedding function except
  for the following differences: (1) no padding for the first 3 layers, (2) no
  maxpool on the last (4th) layer, and (3) no flatten.

  Paper: https://arxiv.org/abs/1711.06025
  Code:
  https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py

  Args:
    inputs: Tensors of shape [None, ] + image shape, e.g. [15, 84, 84, 3]
    is_training: Whether we are in the training phase.
    params: None will create new params (or reuse from scope), otherwise an
      ordered dict of convolutional kernels and biases such that
      params['kernel_0'] stores the kernel of the first convolutional layer,
      etc.
    moments: A dict of the means and vars of the different layers to use for
      batch normalization. If not provided, the mean and var are computed based
      on the given inputs.
    depth_multiplier: The depth multiplier for the convnet channels.
    reuse: Whether to reuse the network's weights.
    scope: An optional scope for the tf operations.
    backprop_through_moments: Whether to allow gradients to flow through the
      given support set moments. Only applies to non-transductive batch norm.
    use_bounded_activation: Whether to enable bounded activation. This is useful
      for post-training quantization.
    keep_spatial_dims: bool, if True the spatial dimensions are kept.

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
  """
  del is_training

  layer = tf.stop_gradient(inputs)
  model_params_keys, model_params_vars = [], []
  moments_keys, moments_vars = [], []

  with tf.variable_scope(scope, reuse=reuse):
    for i in range(4):
      with tf.variable_scope('layer_{}'.format(i), reuse=reuse):
        depth = int(64 * depth_multiplier)
        # The original implementation had VALID padding for the first two layers
        # that are followed by pooling. The rest (last two) had `SAME` padding.
        # In our setting, to avoid OOM, we pool (and apply VALID padding) to
        # the first three layers, and use SAME padding only in the last one.
        layer, conv_bn_params, conv_bn_moments = conv_bn(
            layer, [3, 3],
            depth,
            stride=1,
            padding='VALID' if i < 3 else 'SAME',
            params=params,
            moments=moments,
            backprop_through_moments=backprop_through_moments)
        model_params_keys.extend(conv_bn_params.keys())
        model_params_vars.extend(conv_bn_params.values())
        moments_keys.extend(conv_bn_moments.keys())
        moments_vars.extend(conv_bn_moments.values())

      layer = relu(layer, use_bounded_activation=use_bounded_activation)
      if i < 3:
        layer = tf.layers.max_pooling2d(layer, [2, 2], 2)
      tf.logging.info('Output of block %d: %s' % (i, layer.shape))

    model_params = collections.OrderedDict(
        zip(model_params_keys, model_params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))
    if not keep_spatial_dims:
      layer = tf.layers.flatten(layer)
    return_dict = {
        'embeddings': layer,
        'params': model_params,
        'moments': moments
    }

    return return_dict


def four_layer_convnet(inputs,
                       is_training,
                       params=None,
                       moments=None,
                       depth_multiplier=1.0,
                       reuse=tf.AUTO_REUSE,
                       scope='four_layer_convnet',
                       backprop_through_moments=True,
                       use_bounded_activation=False,
                       keep_spatial_dims=False):
  """Embeds inputs using a standard four-layer convnet.

  Args:
    inputs: Tensors of shape [None, ] + image shape, e.g. [15, 84, 84, 3]
    is_training: Whether we are in the training phase.
    params: None will create new params (or reuse from scope), otherwise an
      ordered dict of convolutional kernels and biases such that
      params['kernel_0'] stores the kernel of the first convolutional layer,
      etc.
    moments: A dict of the means and vars of the different layers to use for
      batch normalization. If not provided, the mean and var are computed based
      on the given inputs.
    depth_multiplier: The depth multiplier for the convnet channels.
    reuse: Whether to reuse the network's weights.
    scope: An optional scope for the tf operations.
    backprop_through_moments: Whether to allow gradients to flow through the
      given support set moments. Only applies to non-transductive batch norm.
    use_bounded_activation: Whether to enable bounded activation. This is useful
      for post-training quantization.
    keep_spatial_dims: bool, if True the spatial dimensions are kept.

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
  """
  del is_training
  return _four_layer_convnet(
      inputs,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      depth_multiplier=depth_multiplier,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      keep_spatial_dims=keep_spatial_dims)


NAME_TO_EMBEDDING_NETWORK = {
    'resnet': resnet,
    'relationnet_embedding': relationnet_embedding,
    'four_layer_convnet': four_layer_convnet,
    'wide_resnet': wide_resnet,
}


# TODO(lamblinp): Make `way` part of the EpisodeDataset itself, to avoid
# recomputing it in the graph.
def compute_way(episode):
  """Compute the way of the episode.

  Args:
    episode: An EpisodeDataset.

  Returns:
    way: An int constant tensor. The number of classes in the episode.
  """
  episode_classes, _ = tf.unique(episode.train_labels)
  way = tf.size(episode_classes)
  return way


def get_embeddings_vars_copy_ops(embedding_vars_dict, make_copies):
  """Gets copies of the embedding variables or returns those variables.

  This is useful at meta-test time for MAML and the finetuning baseline. In
  particular, at meta-test time, we don't want to make permanent updates to
  the model's variables, but only modifications that persist in the given
  episode. This can be achieved by creating copies of each variable and
  modifying and using these copies instead of the variables themselves.

  Args:
    embedding_vars_dict: A dict mapping each variable name to the corresponding
      Variable.
    make_copies: A bool. Whether to copy the given variables. If not, those
      variables themselves will be returned. Typically, this is True at meta-
      test time and False at meta-training time.

  Returns:
    embedding_vars_keys: A list of variable names.
    embeddings_vars: A corresponding list of Variables.
    embedding_vars_copy_ops: A (possibly empty) list of operations, each of
      which assigns the value of one of the provided Variables to a new
      Variable which is its copy.
  """
  embedding_vars_keys = []
  embedding_vars = []
  embedding_vars_copy_ops = []
  for name, var in six.iteritems(embedding_vars_dict):
    embedding_vars_keys.append(name)
    if make_copies:
      with tf.variable_scope('weight_copy'):
        shape = var.shape.as_list()
        var_copy = tf.Variable(
            tf.zeros(shape), collections=[tf.GraphKeys.LOCAL_VARIABLES])
        var_copy_op = tf.assign(var_copy, var)
        embedding_vars_copy_ops.append(var_copy_op)
      embedding_vars.append(var_copy)
    else:
      embedding_vars.append(var)
  return embedding_vars_keys, embedding_vars, embedding_vars_copy_ops


def get_fc_vars_copy_ops(fc_weights, fc_bias, make_copies):
  """Gets copies of the classifier layer variables or returns those variables.

  At meta-test time, a copy is created for the given Variables, and these copies
  copies will be used in place of the original ones.

  Args:
    fc_weights: A Variable for the weights of the fc layer.
    fc_bias: A Variable for the bias of the fc layer.
    make_copies: A bool. Whether to copy the given variables. If not, those
      variables themselves are returned.

  Returns:
    fc_weights: A Variable for the weights of the fc layer. Might be the same as
      the input fc_weights or a copy of it.
    fc_bias: Analogously, a Variable for the bias of the fc layer.
    fc_vars_copy_ops: A (possibly empty) list of operations for assigning the
      value of each of fc_weights and fc_bias to a respective copy variable.
  """
  fc_vars_copy_ops = []
  if make_copies:
    with tf.variable_scope('weight_copy'):
      # fc_weights copy
      fc_weights_copy = tf.Variable(
          tf.zeros(fc_weights.shape.as_list()),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      fc_weights_copy_op = tf.assign(fc_weights_copy, fc_weights)
      fc_vars_copy_ops.append(fc_weights_copy_op)

      # fc_bias copy
      fc_bias_copy = tf.Variable(
          tf.zeros(fc_bias.shape.as_list()),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      fc_bias_copy_op = tf.assign(fc_bias_copy, fc_bias)
      fc_vars_copy_ops.append(fc_bias_copy_op)

      fc_weights = fc_weights_copy
      fc_bias = fc_bias_copy
  return fc_weights, fc_bias, fc_vars_copy_ops


def gradient_descent_step(loss,
                          variables,
                          stop_grads,
                          allow_grads_to_batch_norm_vars,
                          learning_rate,
                          get_update_ops=True):
  """Returns the updated vars after one step of gradient descent."""
  grads = tf.gradients(loss, variables)

  if stop_grads:
    grads = [tf.stop_gradient(dv) for dv in grads]

  def _apply_grads(variables, grads):
    """Applies gradients using SGD on a list of variables."""
    v_new, update_ops = [], []
    for (v, dv) in zip(variables, grads):
      if (not allow_grads_to_batch_norm_vars and
          ('offset' in v.name or 'scale' in v.name)):
        updated_value = v  # no update.
      else:
        updated_value = v - learning_rate * dv  # gradient descent update.
        if get_update_ops:
          update_ops.append(tf.assign(v, updated_value))
      v_new.append(updated_value)
    return v_new, update_ops

  updated_vars, update_ops = _apply_grads(variables, grads)
  return {'updated_vars': updated_vars, 'update_ops': update_ops}


class Learner(object):
  """A Learner."""

  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, data):
    """Initializes a Learner.

    Args:
      is_training: Whether the learning is in training mode.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      data: An EpisodeDataset or Batch.

    Returns:
      A loss (potentially dependent on ops, e.g. for updating EMA), predictions.
    """
    self.is_training = is_training
    self.transductive_batch_norm = transductive_batch_norm
    self.backprop_through_moments = backprop_through_moments
    self.ema_object = ema_object
    self.embedding_fn = embedding_fn
    self.data = data

  def update_ema(self):
    """Apply the update operation."""

  def compute_loss(self):
    """Returns a Tensor representing the loss."""

  def forward_pass(self):
    """Returns the features of the given batch or episode."""


@gin.configurable
class RelationNetworkLearner(Learner):
  """A Relation Network."""

  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, reader,
               weight_decay):
    """Initializes a RelationNetworkLearner.

    Args:
      is_training: Whether the learning is in training mode.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      weight_decay: coefficient for L2 regularization.
    """
    super(RelationNetworkLearner,
          self).__init__(is_training, transductive_batch_norm,
                         backprop_through_moments, ema_object, embedding_fn,
                         reader)

    # The data for the next episode.
    self.episode = self.data
    self.test_targets = self.episode.test_labels
    self.way = compute_way(self.data)

    # Hyperparameters.
    self.weight_decay = weight_decay
    tf.logging.info(
        'RelationNetworkLearner: weight_decay {}'.format(weight_decay))

    # Parameters for embedding function depending on meta-training or not.
    self.forward_pass()

  def forward_pass(self):
    """Embeds all (training and testing) images of the episode."""
    # Compute the support set's mean and var and use these as the moments for
    # batch norm on the query set.
    train_embeddings = self.embedding_fn(
        self.episode.train_images, self.is_training, keep_spatial_dims=True)
    self.train_embeddings = train_embeddings['embeddings']
    support_set_moments = None
    if not self.transductive_batch_norm:
      support_set_moments = train_embeddings['moments']
    test_embeddings = self.embedding_fn(
        self.episode.test_images,
        self.is_training,
        moments=support_set_moments,
        backprop_through_moments=self.backprop_through_moments,
        keep_spatial_dims=True)
    self.test_embeddings = test_embeddings['embeddings']

  def compute_relations(self):
    """Computes the relation score of each test example to each prototype."""
    # [n_test, 21, 21, n_features].
    test_embed_shape = self.test_embeddings.shape.as_list()
    n_feature = test_embed_shape[3]
    out_shape = test_embed_shape[1:3]
    n_test = tf.shape(self.test_embeddings)[0]

    # [n_test, num_clases, 21, 21, n_feature].
    # It is okay one of the elements in the list to be tensor.
    prototype_extended = tf.tile(
        tf.expand_dims(self.prototypes, 0), [n_test, 1, 1, 1, 1])
    # [num_clases, n_test, 21, 21, n_feature].
    test_f_extended = tf.tile(
        tf.expand_dims(self.test_embeddings, 1), [1, self.way, 1, 1, 1])
    relation_pairs = tf.concat((prototype_extended, test_f_extended), 4)
    # relation_pairs.shape.as_list()[-3:] == [-1] + out_shape + [n_feature*2]
    relation_pairs = tf.reshape(relation_pairs,
                                [-1] + out_shape + [n_feature * 2])
    self.relationnet_dict = _relation_net(relation_pairs, 'relationnet')
    relations = tf.reshape(self.relationnet_dict['output'], [-1, self.way])
    return relations

  def compute_loss(self):
    """Returns the loss of the Prototypical Network."""

    onehot_train_labels = tf.one_hot(self.episode.train_labels, self.way)
    self.prototypes = compute_prototypes(self.train_embeddings,
                                         onehot_train_labels)
    self.test_logits = self.compute_relations()
    onehot_test_labels = tf.one_hot(self.episode.test_labels, self.way)
    mse_loss = tf.losses.mean_squared_error(onehot_test_labels,
                                            self.test_logits)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = mse_loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self):
    """Computes the accuracy on the given episode."""
    self.test_predictions = tf.cast(tf.argmax(self.test_logits, 1), tf.int32)
    correct = tf.equal(self.episode.test_labels, self.test_predictions)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


@gin.configurable
class PrototypicalNetworkLearner(Learner):
  """A Prototypical Network."""

  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, reader,
               weight_decay):
    """Initializes a PrototypicalNetworkLearner.

    Args:
      is_training: Whether the learning is in training mode.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      weight_decay: coefficient for L2 regularization.
    """
    super(PrototypicalNetworkLearner,
          self).__init__(is_training, transductive_batch_norm,
                         backprop_through_moments, ema_object, embedding_fn,
                         reader)

    # The data for the next episode.
    self.episode = self.data
    self.test_targets = self.episode.test_labels
    self.way = compute_way(self.data)

    # Hyperparameters.
    self.weight_decay = weight_decay
    logging.info('PrototypicalLearner: weight_decay %s', weight_decay)

    # Parameters for embedding function depending on meta-training or not.
    self.forward_pass()

  def forward_pass(self):
    """Embeds all (training and testing) images of the episode."""
    # Compute the support set's mean and var and use these as the moments for
    # batch norm on the query set.
    train_embeddings = self.embedding_fn(self.episode.train_images,
                                         self.is_training)
    self.train_embeddings = train_embeddings['embeddings']
    support_set_moments = None
    if not self.transductive_batch_norm:
      support_set_moments = train_embeddings['moments']
    test_embeddings = self.embedding_fn(
        self.episode.test_images,
        self.is_training,
        moments=support_set_moments,
        backprop_through_moments=self.backprop_through_moments)
    self.test_embeddings = test_embeddings['embeddings']

  def compute_logits(self):
    """Computes the negative distances of each test point to each prototype."""
    # [num test images, 1, embedding size].
    test_embeddings = tf.expand_dims(self.test_embeddings, 1)

    # [1, num_clases, embedding_size].
    prototypes = tf.expand_dims(self.prototypes, 0)

    # Squared euclidean distances between each test embedding / prototype pair.
    distances = tf.reduce_sum(tf.square(test_embeddings - prototypes), 2)
    self.test_logits = -distances
    return self.test_logits

  def compute_loss(self):
    """Returns the loss of the Prototypical Network."""
    onehot_train_labels = tf.one_hot(self.episode.train_labels, self.way)
    self.prototypes = compute_prototypes(self.train_embeddings,
                                         onehot_train_labels)
    self.test_logits = self.compute_logits()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.episode.test_labels, logits=self.test_logits)
    cross_entropy_loss = tf.reduce_mean(loss)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self):
    """Computes the accuracy on the given episode."""
    self.test_predictions = tf.cast(tf.argmax(self.test_logits, 1), tf.int32)
    correct = tf.equal(self.episode.test_labels, self.test_predictions)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


@gin.configurable
class MatchingNetworkLearner(PrototypicalNetworkLearner):
  """A Matching Network."""

  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, reader,
               weight_decay, exact_cosine_distance):
    """Initializes the Matching Networks instance.

    Args:
      is_training: Whether the learning is in training mode.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      weight_decay: coefficient for L2 regularization.
      exact_cosine_distance: If True then the cosine distance is used, otherwise
        the query set embeddings are left unnormalized when computing the dot
        product.
    """
    super(MatchingNetworkLearner,
          self).__init__(is_training, transductive_batch_norm,
                         backprop_through_moments, ema_object, embedding_fn,
                         reader, weight_decay)

    self.exact_cosine_distance = exact_cosine_distance
    self.weight_decay = weight_decay
    logging.info('MatchingNetworkLearner: weight_decay %s', weight_decay)

  def compute_logits(self):
    """Computes the class logits.

    Probabilities are computed as a weighted sum of one-hot encoded training
    labels. Weights for individual support/query pairs of examples are
    proportional to the (potentially semi-normalized) cosine distance between
    the embeddings of the two examples.

    Returns:
      The class logits as a [num_test_images, way] matrix.
    """
    # [num train labels, num classes] where each row is a one-hot-encoded label.
    onehot_train_labels = tf.one_hot(self.data.train_labels, self.way)

    # Undocumented in the paper, but *very important*: *only* the support set
    # embeddings is L2-normalized, which means that the distance is not exactly
    # a cosine distance. For comparison we also allow for the actual cosine
    # distance to be computed, which is controlled with the
    # `exact_cosine_distance` instance attribute.
    train_embeddings = tf.nn.l2_normalize(
        self.train_embeddings, 1, epsilon=1e-3)
    test_embeddings = self.test_embeddings
    if self.exact_cosine_distance:
      test_embeddings = tf.nn.l2_normalize(test_embeddings, 1, epsilon=1e-3)
    # [num_test_images, num_train_images]
    similarities = tf.matmul(
        test_embeddings, train_embeddings, transpose_b=True)
    attention = tf.nn.softmax(similarities)

    # [num_test_images, way]
    probs = tf.matmul(attention, onehot_train_labels)
    self.test_logits = tf.log(probs)
    return self.test_logits

  def compute_loss(self):
    """Returns the loss of the Matching Network."""
    self.test_logits = self.compute_logits()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.data.test_labels, logits=self.test_logits)
    cross_entropy_loss = tf.reduce_mean(loss)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + self.weight_decay * regularization
    return loss


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


def linear_classifier_logits(embeddings, num_classes, cosine_classifier,
                             cosine_logits_multiplier, use_weight_norm):
  """Forward pass through a linear classifier, possibly a cosine classifier."""

  # A variable to keep track of whether the initialization has already happened.
  data_dependent_init_done = tf.get_variable(
      'data_dependent_init_done',
      initializer=0,
      dtype=tf.int32,
      trainable=False)

  embedding_dims = embeddings.get_shape().as_list()[-1]

  if use_weight_norm:
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
    w_fc = weight_variable([embedding_dims, num_classes])
    b_fc = None
    if not cosine_classifier:
      # Also initialize a bias.
      b_fc = bias_variable([num_classes])
    # Forward pass through the layer defined by w_fc and b_fc.
    logits = linear_classifier_forward_pass(embeddings, w_fc, b_fc,
                                            cosine_classifier,
                                            cosine_logits_multiplier, False)
  return logits


# TODO(tylerzhu): Consider adding an episodic kNN learner as well so we can
# create a baseline leaner by composing a batch learner and the evaluation
# process of an episodic kNN learner.
@gin.configurable
class BaselineLearner(Learner):
  """A Baseline Network."""

  # TODO(kswersky): get rid of these arguments.
  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, reader,
               num_train_classes, num_test_classes, weight_decay, knn_in_fc,
               knn_distance, cosine_classifier, cosine_logits_multiplier,
               use_weight_norm):
    """Initializes a baseline learner.

    Args:
      is_training: If we are training or not.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      num_train_classes: The total number of classes of the dataset.
      num_test_classes: The number of classes in each episode.
      weight_decay: coefficient for L2 regularization.
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
    """
    self.cosine_classifier = cosine_classifier
    self.cosine_logits_multiplier = cosine_logits_multiplier
    self.use_weight_norm = use_weight_norm

    super(BaselineLearner, self).__init__(is_training, transductive_batch_norm,
                                          backprop_through_moments, ema_object,
                                          embedding_fn, reader)

    self.num_train_classes = num_train_classes
    self.num_test_classes = num_test_classes

    # Hyperparameters.
    self.weight_decay = weight_decay
    self.distance = knn_distance
    logging.info(
        'BaselineLearner: distance %s, weight_decay %s, cosine_classifier: %s',
        knn_distance, weight_decay, cosine_classifier)

    self.forward_pass()

    if not self.is_training:
      # For aggregating statistics later.
      self.test_targets = self.data.test_labels
      self.way = compute_way(self.data)
      self.knn_in_fc = knn_in_fc
      logging.info('BaselineLearner: knn_in_fc %s', knn_in_fc)

  def forward_pass(self):
    if self.is_training:
      images = self.data.images
      embeddings_params_moments = self.embedding_fn(images, self.is_training)
      self.train_embeddings = embeddings_params_moments['embeddings']
      self.train_embeddings_var_dict = embeddings_params_moments['params']
    else:
      train_embeddings_params_moments = self.embedding_fn(
          self.data.train_images, self.is_training)
      self.train_embeddings = train_embeddings_params_moments['embeddings']
      self.train_embeddings_var_dict = train_embeddings_params_moments['params']
      support_set_moments = None
      if not self.transductive_batch_norm:
        support_set_moments = train_embeddings_params_moments['moments']
      test_embeddings = self.embedding_fn(
          self.data.test_images,
          self.is_training,
          moments=support_set_moments,
          backprop_through_moments=self.backprop_through_moments)
      self.test_embeddings = test_embeddings['embeddings']

  def forward_pass_fc(self, embeddings):
    """Passes the provided embeddings through the fc layer to get the logits.

    Args:
      embeddings: A Tensor of the penultimate layer activations as computed by
        self.forward_pass().

    Returns:
      The fc layer activations.
    """
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      # Always maps to a space whose dimensionality is the number of classes
      # at meta-training time.
      logits = linear_classifier_logits(embeddings, self.num_train_classes,
                                        self.cosine_classifier,
                                        self.cosine_logits_multiplier,
                                        self.use_weight_norm)
      return logits

  def compute_logits(self):
    """Returns the logits.

    Raises:
      ValueError: Distance must be one of l2 or cosine.
    """
    if self.is_training:
      logits = self.forward_pass_fc(self.train_embeddings)
    else:
      if self.knn_in_fc:
        # Overwrite the train and test embeddings that were originally computed
        # in self.forward_pass() to be the fc layer activations.
        all_logits = self.forward_pass_fc(self.all_embeddings)
        self.train_embeddings = all_logits[:self.num_train_images]
        self.test_embeddings = all_logits[self.num_train_images:]

      num_classes = self.way
      # ------------------------ K-NN look up -------------------------------
      # For each testing example in an episode, we use its embedding
      # vector to look for the closest neighbor in all the training examples'
      # embeddings from the same episode and then assign the training example's
      # class label to the testing example as the predicted class label for it.
      #  [num_train]
      train_labels = self.data.train_labels
      #  [num_train, num_classes]
      onehot_train_labels = tf.one_hot(train_labels, num_classes)
      if self.distance == 'l2':
        #  [1, num_train, embed_dims]
        train_embeddings = tf.expand_dims(self.train_embeddings, axis=0)
        #  [num_test, 1, embed_dims]
        test_embeddings = tf.expand_dims(self.test_embeddings, axis=1)
        #  [num_test, num_train]
        distance = tf.norm(test_embeddings - train_embeddings, axis=2)
      elif self.distance == 'cosine':
        train_embeddings = tf.nn.l2_normalize(self.train_embeddings, axis=1)
        test_embeddings = tf.nn.l2_normalize(self.test_embeddings, axis=1)
        distance = -1 * tf.matmul(
            test_embeddings, train_embeddings, transpose_b=True)
      else:
        raise ValueError('Distance must be one of l2 or cosine.')
      #  [num_test]
      _, indices = tf.nn.top_k(-distance, k=1)
      indices = tf.squeeze(indices, axis=1)
      #  [num_test, num_classes]
      self.test_logits = tf.gather(onehot_train_labels, indices)
      logits = self.test_logits
    return logits

  def compute_loss(self):
    """Computes the loss."""
    if self.is_training:
      self.train_logits = self.compute_logits()
      labels = tf.cast(self.data.labels, tf.int64)
      onehot_labels = tf.one_hot(labels, self.num_train_classes)
      with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=self.train_logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
        regularization = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = cross_entropy + self.weight_decay * regularization
        return loss
    else:
      self.test_logits = self.compute_logits()
      return tf.constant(0.0)

  def compute_accuracy(self):
    """Computes the accuracy on the given episode."""
    if not self.is_training:
      logits = self.test_logits
      labels = self.data.test_labels
    else:
      logits = self.train_logits
      labels = self.data.labels

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    correct = tf.equal(labels, predictions)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


@gin.configurable
class BaselineFinetuneLearner(BaselineLearner):
  """A Baseline Network with test-time finetuning."""

  def __init__(self,
               is_training,
               transductive_batch_norm,
               backprop_through_moments,
               ema_object,
               embedding_fn,
               reader,
               num_train_classes,
               num_test_classes,
               weight_decay,
               num_finetune_steps,
               finetune_lr,
               debug_log=False,
               finetune_all_layers=False,
               finetune_with_adam=False):
    """Initializes a baseline learner.

    Args:
      is_training: If we are training or not.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      num_train_classes: The total number of classes of the dataset.
      num_test_classes: The number of classes in each episode.
      weight_decay: coefficient for L2 regularization.
      num_finetune_steps: number of finetune steps.
      finetune_lr: the learning rate used for finetuning.
      debug_log: If True, print out debug logs.
      finetune_all_layers: Whether to finetune all embedding variables. If
        False, only trains a linear classifier on top of the embedding.
      finetune_with_adam: Whether to use Adam for the within-episode finetuning.
        If False, gradient descent is used instead.
    """
    self.num_finetune_steps = num_finetune_steps
    self.finetune_lr = finetune_lr
    self.debug_log = debug_log
    self.finetune_all_layers = finetune_all_layers
    self.finetune_with_adam = finetune_with_adam

    if finetune_with_adam:
      self.finetune_opt = tf.train.AdamOptimizer(self.finetune_lr)

    # Note: the weight_decay value provided here overrides the value gin might
    # have for BaselineLearner's own weight_decay.
    super(BaselineFinetuneLearner,
          self).__init__(is_training, transductive_batch_norm,
                         backprop_through_moments, ema_object, embedding_fn,
                         reader, num_train_classes, num_test_classes,
                         weight_decay)

  def compute_logits(self):
    """Computes the logits."""
    logits = None
    if self.is_training:
      logits = self.forward_pass_fc(self.train_embeddings)
    else:
      # ------------------------ Finetuning -------------------------------
      # Possibly make copies of embedding variables, if they will get modified.
      # This is for making temporary-only updates to the embedding network
      # which will not persist after the end of the episode.
      make_copies = self.finetune_all_layers
      (self.embedding_vars_keys, self.embedding_vars,
       embedding_vars_copy_ops) = get_embeddings_vars_copy_ops(
           self.train_embeddings_var_dict, make_copies)
      embedding_vars_copy_op = tf.group(*embedding_vars_copy_ops)

      # Compute the initial training loss (only for printing purposes). This
      # line is also needed for adding the fc variables to the graph so that the
      # tf.all_variables() line below detects them.
      logits = self._fc_layer(self.train_embeddings)[:, 0:self.way]
      finetune_loss = self._classification_loss(logits, self.data.train_labels,
                                                self.way)

      # Decide which variables to finetune.
      fc_vars, vars_to_finetune = [], []
      for var in tf.trainable_variables():
        if 'fc_finetune' in var.name:
          fc_vars.append(var)
          vars_to_finetune.append(var)
      if self.finetune_all_layers:
        vars_to_finetune.extend(self.embedding_vars)
      self.vars_to_finetune = vars_to_finetune
      logging.info('Finetuning will optimize variables: %s', vars_to_finetune)

      for i in range(self.num_finetune_steps):
        if i == 0:
          # Randomly initialize the fc layer.
          fc_reset = tf.variables_initializer(var_list=fc_vars)
          # Adam related variables are created when minimize() is called.
          # We create an unused op here to put all adam varariables under
          # the 'adam_opt' namescope and create a reset op to reinitialize
          # these variables before the first finetune step.
          adam_reset = tf.no_op()
          if self.finetune_with_adam:
            with tf.variable_scope('adam_opt'):
              unused_op = self.finetune_opt.minimize(
                  finetune_loss, var_list=vars_to_finetune)
            adam_reset = tf.variables_initializer(self.finetune_opt.variables())
          with tf.control_dependencies(
              [fc_reset, adam_reset, finetune_loss, embedding_vars_copy_op] +
              vars_to_finetune):
            print_op = tf.no_op()
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                  finetune_loss
              ])

            with tf.control_dependencies([print_op]):
              # Get the operation for finetuning.
              # (The logits and loss are returned just for printing).
              logits, finetune_loss, finetune_op = self._get_finetune_op()

              if self.debug_log:
                # Test logits are computed only for printing logs.
                test_embeddings = self.embedding_fn(
                    self.data.test_images,
                    self.is_training,
                    params=collections.OrderedDict(
                        zip(self.embedding_vars_keys, self.embedding_vars)),
                    reuse=True)['embeddings']
                test_logits = self._fc_layer(test_embeddings)[:, 0:self.way]

        else:
          with tf.control_dependencies([finetune_op, finetune_loss] +
                                       vars_to_finetune):
            print_op = tf.no_op()
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                  finetune_loss, 'accuracy:',
                  self._compute_accuracy(logits, self.data.train_labels),
                  'test accuracy:',
                  self._compute_accuracy(test_logits, self.data.test_labels)
              ])

            with tf.control_dependencies([print_op]):
              # Get the operation for finetuning.
              # (The logits and loss are returned just for printing).
              logits, finetune_loss, finetune_op = self._get_finetune_op()

              if self.debug_log:
                # Test logits are computed only for printing logs.
                test_embeddings = self.embedding_fn(
                    self.data.test_images,
                    self.is_training,
                    params=collections.OrderedDict(
                        zip(self.embedding_vars_keys, self.embedding_vars)),
                    reuse=True)['embeddings']
                test_logits = self._fc_layer(test_embeddings)[:, 0:self.way]

      # Finetuning is now over, compute the test performance using the updated
      # fc layer, and possibly the updated embedding network.
      with tf.control_dependencies([finetune_op] + vars_to_finetune):
        test_embeddings = self.embedding_fn(
            self.data.test_images,
            self.is_training,
            params=collections.OrderedDict(
                zip(self.embedding_vars_keys, self.embedding_vars)),
            reuse=True)['embeddings']
        test_logits = self._fc_layer(test_embeddings)[:, 0:self.way]

        if self.debug_log:
          # The train logits are computed only for printing.
          train_embeddings = self.embedding_fn(
              self.data.train_images,
              self.is_training,
              params=collections.OrderedDict(
                  zip(self.embedding_vars_keys, self.embedding_vars)),
              reuse=True)['embeddings']
          logits = self._fc_layer(train_embeddings)[:, 0:self.way]

        print_op = tf.no_op()
        if self.debug_log:
          print_op = tf.print([
              'accuracy: ',
              self._compute_accuracy(logits, self.data.train_labels),
              'test accuracy: ',
              self._compute_accuracy(test_logits, self.data.test_labels)
          ])
        with tf.control_dependencies([print_op]):
          self.test_logits = self._fc_layer(test_embeddings)[:, 0:self.way]
          logits = self.test_logits
    return logits

  def _get_finetune_op(self):
    """Returns the operation for performing a finetuning step."""
    if self.finetune_all_layers:
      # Must re-do the forward pass because the embedding has changed.
      train_embeddings = self.embedding_fn(
          self.data.train_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(self.embedding_vars_keys, self.embedding_vars)),
          reuse=True)['embeddings']
    else:
      train_embeddings = self.train_embeddings
    logits = self._fc_layer(train_embeddings)[:, 0:self.way]
    finetune_loss = self._classification_loss(logits, self.data.train_labels,
                                              self.way)
    # Perform one step of finetuning.
    if self.finetune_with_adam:
      finetune_op = self.finetune_opt.minimize(
          finetune_loss, var_list=self.vars_to_finetune)
    else:
      # Apply vanilla gradient descent instead of Adam.
      update_ops = gradient_descent_step(finetune_loss, self.vars_to_finetune,
                                         True, False,
                                         self.finetune_lr)['update_ops']
      finetune_op = tf.group(*update_ops)
    return logits, finetune_loss, finetune_op

  def _fc_layer(self, embedding):
    """The fully connected layer to be finetuned."""
    with tf.variable_scope('fc_finetune', reuse=tf.AUTO_REUSE):
      logits = linear_classifier_logits(embedding, MAX_WAY,
                                        self.cosine_classifier,
                                        self.cosine_logits_multiplier,
                                        self.use_weight_norm)
    return logits

  def _classification_loss(self, logits, labels, num_classes):
    """Computes softmax cross entropy loss."""
    labels = tf.cast(labels, tf.int64)
    onehot_labels = tf.one_hot(labels, num_classes)
    with tf.name_scope('finetuning_loss'):
      cross_entropy = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)
      cross_entropy = tf.reduce_mean(cross_entropy)
      regularization = tf.reduce_sum(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      loss = cross_entropy + self.weight_decay * regularization
    return loss

  def _compute_accuracy(self, logits, labels):
    """Computes the accuracy on the given episode."""
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    correct = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


@gin.configurable
class MAMLLearner(Learner):
  """Model-Agnostic Meta Learner."""

  def __init__(self, is_training, transductive_batch_norm,
               backprop_through_moments, ema_object, embedding_fn, reader,
               weight_decay, num_update_steps, additional_test_update_steps,
               first_order, alpha, train_batch_norm, debug, zero_fc_layer,
               proto_maml_fc_layer_init):
    """Initializes a baseline learner.

    Args:
      is_training: Whether the learning is in training mode.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      weight_decay: coefficient for L2 regularization.
      num_update_steps: The number of inner-loop steps to take.
      additional_test_update_steps: The number of additional inner-loop steps to
        take on meta test and meta validation set.
      first_order: If True, ignore second-order gradients (faster).
      alpha: The inner-loop learning rate.
      train_batch_norm: If True, train batch norm during meta training.
      debug: If True, print out debug logs.
      zero_fc_layer: Whether to use zero fc layer initialization.
      proto_maml_fc_layer_init: Whether to use ProtoNets equivalent fc layer
        initialization.

    Raises:
      ValueError: The embedding function must be MAML-compatible.
      RuntimeError: Requested to meta-learn the initialization of the linear
        layer weights but they are unexpectedly omitted from saving/restoring.
    """
    if not zero_fc_layer and not proto_maml_fc_layer_init:
      # So the linear classifier weights initialization is meta-learned.
      if 'linear_classifier' in FLAGS.omit_from_saving_and_reloading:
        raise RuntimeError('The linear layer is requested to be meta-learned '
                           'since both zero_fc_layer and '
                           'proto_maml_fc_layer_init are False, but the '
                           'linear_classifier weights are found in '
                           'FLAGS.omit_from_saving_and_reloading so they will '
                           'not be properly restored. Please exclude these '
                           'weights from omit_from_saving_and_reloading for '
                           'this setting to work as expected.')

    super(MAMLLearner, self).__init__(is_training, transductive_batch_norm,
                                      backprop_through_moments, ema_object,
                                      embedding_fn, reader)

    # For aggregating statistics later.
    self.test_targets = self.data.test_labels
    self.way = compute_way(self.data)

    self.weight_decay = weight_decay
    self.alpha = alpha
    self.num_update_steps = num_update_steps
    self.additional_test_update_steps = additional_test_update_steps
    self.first_order = first_order
    self.train_batch_norm = train_batch_norm
    self.debug_log = debug
    self.zero_fc_layer = zero_fc_layer
    self.proto_maml_fc_layer_init = proto_maml_fc_layer_init

    logging.info('alpha: %s, num_update_steps: %d', self.alpha,
                 self.num_update_steps)
    self.forward_pass()

  def proto_maml_fc_weights(self, prototypes, zero_pad_to_max_way=False):
    """Computes the Prototypical MAML fc layer's weights.

    Args:
      prototypes: Tensor of shape [num_classes, embedding_size]
      zero_pad_to_max_way: Whether to zero padd to max num way.

    Returns:
      fc_weights: Tensor of shape [embedding_size, num_classes] or
        [embedding_size, MAX_WAY] when zero_pad_to_max_way is True.
    """
    fc_weights = 2 * prototypes
    fc_weights = tf.transpose(fc_weights)
    if zero_pad_to_max_way:
      paddings = [[0, 0], [0, MAX_WAY - tf.shape(fc_weights)[1]]]
      fc_weights = tf.pad(fc_weights, paddings, 'CONSTANT', constant_values=0)
    return fc_weights

  def proto_maml_fc_bias(self, prototypes, zero_pad_to_max_way=False):
    """Computes the Prototypical MAML fc layer's bias.

    Args:
      prototypes: Tensor of shape [num_classes, embedding_size]
      zero_pad_to_max_way: Whether to zero padd to max num way.

    Returns:
      fc_bias: Tensor of shape [num_classes] or [MAX_WAY]
        when zero_pad_to_max_way is True.
    """
    fc_bias = -tf.square(tf.norm(prototypes, axis=1))
    if zero_pad_to_max_way:
      paddings = [[0, MAX_WAY - tf.shape(fc_bias)[0]]]
      fc_bias = tf.pad(fc_bias, paddings, 'CONSTANT', constant_values=0)
    return fc_bias

  def forward_pass(self):
    """Computes the test logits of MAML.

    Computes the test logits of MAML on the query (test) set after running
    meta update steps on the support (train) set.
    """
    # Have to use one-hot labels since sparse softmax doesn't allow
    # second derivatives.
    onehot_train_labels = tf.one_hot(self.data.train_labels, self.way)
    train_embeddings_ = self.embedding_fn(
        self.data.train_images, self.is_training, reuse=tf.AUTO_REUSE)
    train_embeddings = train_embeddings_['embeddings']
    embedding_vars_dict = train_embeddings_['params']

    with tf.variable_scope('linear_classifier', reuse=tf.AUTO_REUSE):
      embedding_depth = train_embeddings.shape.as_list()[-1]
      fc_weights = weight_variable([embedding_depth, MAX_WAY])
      fc_bias = bias_variable([MAX_WAY])

    # A list of variable names, a list of corresponding Variables, and a list
    # of operations (possibly empty) that creates a copy of each Variable.
    (embedding_vars_keys, embedding_vars,
     embedding_vars_copy_ops) = get_embeddings_vars_copy_ops(
         embedding_vars_dict, make_copies=not self.is_training)

    # A Variable for the weights of the fc layer, a Variable for the bias of the
    # fc layer, and a list of operations (possibly empty) that copies them.
    (fc_weights, fc_bias, fc_vars_copy_ops) = get_fc_vars_copy_ops(
        fc_weights, fc_bias, make_copies=not self.is_training)

    fc_vars = [fc_weights, fc_bias]
    num_embedding_vars = len(embedding_vars)
    num_fc_vars = len(fc_vars)

    def _cond(step, *args):
      del args
      num_steps = self.num_update_steps
      if not self.is_training:
        num_steps += self.additional_test_update_steps
      return step < num_steps

    def _body(step, *args):
      """The inner update loop body."""
      updated_embedding_vars = args[0:num_embedding_vars]
      updated_fc_vars = args[num_embedding_vars:num_embedding_vars +
                             num_fc_vars]
      train_embeddings = self.embedding_fn(
          self.data.train_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          reuse=True)['embeddings']

      updated_fc_weights, updated_fc_bias = updated_fc_vars
      train_logits = tf.matmul(train_embeddings,
                               updated_fc_weights) + updated_fc_bias

      train_logits = train_logits[:, 0:self.way]
      loss = tf.losses.softmax_cross_entropy(onehot_train_labels, train_logits)

      print_op = tf.no_op()
      if self.debug_log:
        print_op = tf.print(['step: ', step, updated_fc_bias[0], 'loss:', loss])

      with tf.control_dependencies([print_op]):
        updated_embedding_vars = gradient_descent_step(
            loss, updated_embedding_vars, self.first_order,
            self.train_batch_norm, self.alpha, False)['updated_vars']
        updated_fc_vars = gradient_descent_step(loss, updated_fc_vars,
                                                self.first_order,
                                                self.train_batch_norm,
                                                self.alpha,
                                                False)['updated_vars']

        step = step + 1
      return tuple([step] + list(updated_embedding_vars) +
                   list(updated_fc_vars))

    # MAML meta updates using query set examples from an episode.
    if self.zero_fc_layer:
      # To account for variable class sizes, we initialize the output
      # weights to zero. See if truncated normal initialization will help.
      zero_weights_op = tf.assign(fc_weights, tf.zeros_like(fc_weights))
      zero_bias_op = tf.assign(fc_bias, tf.zeros_like(fc_bias))
      fc_vars_init_ops = [zero_weights_op, zero_bias_op]
    else:
      fc_vars_init_ops = fc_vars_copy_ops

    if self.proto_maml_fc_layer_init:
      train_embeddings = self.embedding_fn(
          self.data.train_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, embedding_vars)),
          reuse=True)['embeddings']

      prototypes = compute_prototypes(train_embeddings, onehot_train_labels)
      pmaml_fc_weights = self.proto_maml_fc_weights(
          prototypes, zero_pad_to_max_way=True)
      pmaml_fc_bias = self.proto_maml_fc_bias(
          prototypes, zero_pad_to_max_way=True)
      fc_vars = [pmaml_fc_weights, pmaml_fc_bias]

    # These control dependencies assign the value of each variable to a new copy
    # variable that corresponds to it. This is required at test time for
    # initilizing the copies as they are used in place of the original vars.
    with tf.control_dependencies(fc_vars_init_ops + embedding_vars_copy_ops):
      # Make step a local variable as we don't want to save and restore it.
      step = tf.Variable(
          0,
          trainable=False,
          name='inner_step_counter',
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      loop_vars = [step] + embedding_vars + fc_vars
      step_and_all_updated_vars = tf.while_loop(
          _cond, _body, loop_vars, swap_memory=True)
      step = step_and_all_updated_vars[0]
      all_updated_vars = step_and_all_updated_vars[1:]
      updated_embedding_vars = all_updated_vars[0:num_embedding_vars]
      updated_fc_weights, updated_fc_bias = all_updated_vars[
          num_embedding_vars:num_embedding_vars + num_fc_vars]

    # Forward pass the training images with the updated weights in order to
    # compute the means and variances, to use for the query's batch norm.
    support_set_moments = None
    if not self.transductive_batch_norm:
      support_set_moments = self.embedding_fn(
          self.data.train_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          reuse=True)['moments']

    test_embeddings = self.embedding_fn(
        self.data.test_images,
        self.is_training,
        params=collections.OrderedDict(
            zip(embedding_vars_keys, updated_embedding_vars)),
        moments=support_set_moments,  # Use support set stats for batch norm.
        reuse=True,
        backprop_through_moments=self.backprop_through_moments)['embeddings']

    self.test_logits = (tf.matmul(test_embeddings, updated_fc_weights) +
                        updated_fc_bias)[:, 0:self.way]

  def compute_loss(self):
    onehot_test_labels = tf.one_hot(self.data.test_labels, self.way)
    loss = tf.losses.softmax_cross_entropy(onehot_test_labels, self.test_logits)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self):
    """Computes the accuracy on the given episode."""
    self.test_predictions = tf.cast(tf.argmax(self.test_logits, 1), tf.int32)
    correct = tf.equal(self.data.test_labels, self.test_predictions)
    return tf.reduce_mean(tf.cast(correct, tf.float32))
