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
"""Backbone-related code."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import functools
from absl import logging
import gin.tf
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


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
      mean, var = tf.nn.moments(
          x, axes=list(range(len(x.shape) - 1)), keep_dims=True)
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


def relation_net(inputs,
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

  This is almost like the `four_layer_convnet` embedding function except
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


@gin.configurable(
    'fully_connected_network', whitelist=[
        'n_hidden_units',
        'use_batchnorm',
    ])
def fully_connected_network(inputs,
                            is_training,
                            params=None,
                            moments=None,
                            n_hidden_units=(64,),
                            use_batchnorm=False,
                            reuse=tf.AUTO_REUSE,
                            scope='fully_connected',
                            use_bounded_activation=False,
                            backprop_through_moments=None,
                            keep_spatial_dims=None):
  """A fully connected linear network.

  Since there is no batch-norm, `moments` and `backprop_through_moments` flags
  are not used.

  Args:
    inputs: Tensor of shape [None, num_features], where `num_features` is the
      number of input features.
    is_training: not used.
    params: None will create new params (or reuse from scope), otherwise an
      ordered dict of fully connected weights and biases such that
      params['weight_0'] stores the kernel of the first fully-connected layer,
      etc.
    moments: not used.
    n_hidden_units: tuple, Number of hidden units for each layer. If empty, it
      is the identity mapping.
    use_batchnorm: bool, Whether to use batchnorm after layers, except last.
    reuse: Whether to reuse the network's weights.
    scope: An optional scope for the tf operations.
    use_bounded_activation: Whether to enable bounded activation. This is useful
      for post-training quantization.
    backprop_through_moments: not used.
    keep_spatial_dims: not used.

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
  """
  del is_training, keep_spatial_dims
  layer = inputs
  model_params_keys, model_params_vars = [], []
  moments_keys, moments_vars = [], []
  activation_fn = functools.partial(
      relu, use_bounded_activation=use_bounded_activation)
  with tf.variable_scope(scope, reuse=reuse):
    for i, n_unit in enumerate(n_hidden_units):
      with tf.variable_scope('layer_%d' % i, reuse=reuse):
        layer, dense_params = dense(
            layer, n_unit, activation_fn=activation_fn, params=params)
        model_params_keys.extend(dense_params.keys())
        model_params_vars.extend(dense_params.values())
        if use_batchnorm:
          layer, bn_params, bn_moments = bn(
              layer,
              params=params,
              moments=moments,
              backprop_through_moments=backprop_through_moments)
          model_params_keys.extend(bn_params.keys())
          model_params_keys.extend(bn_params.values())
          moments_keys.extend(bn_moments.keys())
          moments_vars.extend(bn_moments.values())

  model_params = collections.OrderedDict(
      zip(model_params_keys, model_params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))
  return_dict = {
      'embeddings': layer,
      'params': model_params,
      'moments': moments
  }
  return return_dict


NAME_TO_EMBEDDING_NETWORK = {
    'resnet': resnet,
    'relationnet_embedding': relationnet_embedding,
    'four_layer_convnet': four_layer_convnet,
    'fully_connected_network': fully_connected_network,
    'wide_resnet': wide_resnet,
}
