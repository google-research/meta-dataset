# coding=utf-8
# Copyright 2021 The Meta-Dataset Authors.
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

OPTIMIZER_KEYWORDS = ('adam', 'power', 'global_step',
                      'data_dependent_init_done')
EMBEDDING_KEYWORDS = ('conv', 'resnet', 'fully_connected')
HEAD_CLASSIFIER_KEYWORDS = ('fc',)


def is_backbone_variable(variable_name, only_if=lambda x: True):
  """Returns True if `variable_name` refers to a backbone variable.

  Args:
    variable_name: A string; the name of a `tf.Variable` that will be checked to
      determine whether the variable belongs to the backbone (embedding
      function) of a `Learner`.
    only_if: A callable that returns `True` when the name of a `tf.Variable`
      satisfies some condition; by default `only_if` returns `True` for any
      argument.

  Returns:
    `True` if the `tf.Variable` to which `variable_name` refers belongs to a
    backbone (embedding function) and `only_if(variable_name)` is also
    satisfied.
  """
  variable_name = variable_name.lower()

  # We restore all embedding variables.
  is_embedding_var = any(
      keyword in variable_name for keyword in EMBEDDING_KEYWORDS)

  # We exclude all head classifier variables.
  is_head_classifier_var = any(
      keyword in variable_name for keyword in HEAD_CLASSIFIER_KEYWORDS)

  # We exclude 'relation*' variables as they are not present in a pretrained
  # checkpoint.
  is_relationnet_var = variable_name.startswith('relation')

  # We exclude optimizer variables, as the episodic finetuning procedure is a
  # different optimization problem than the original training objective.
  is_optimizer_var = any(
      keyword in variable_name for keyword in OPTIMIZER_KEYWORDS)

  return (only_if(variable_name) and is_embedding_var and
          not is_head_classifier_var and not is_relationnet_var and
          not is_optimizer_var)


def conv2d(x, w, stride=1, b=None, padding='SAME', rate=1):
  """conv2d returns a 2d convolution layer with full stride."""
  h = tf.nn.conv2d(
      x,
      w,
      strides=[1, stride, stride, 1],
      padding=padding,
      dilations=[1, rate, rate, 1])
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
@gin.configurable('bn', allowlist=['use_ema', 'ema_epsilon'])
def bn(x,
       params=None,
       moments=None,
       backprop_through_moments=True,
       use_ema=False,
       is_training=True,
       ema_epsilon=.9):
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
    use_ema: apply moving averages of batch norm statistics, or update them,
      depending on whether we are training or testing.  Note that passing
      moments will override this setting, and result in neither updating or
      using ema statistics.  This is important to make sure that episodic
      learners don't update ema statistics a second time when processing
      queries.
    is_training: if use_ema=True, this determines whether to apply the moving
      averages, or update them.
    ema_epsilon: if updating moving averages, use this value for the
      exponential moving averages.

  Returns:
    output: The result of applying batch normalization to the input.
    params: The updated params.
    moments: The updated moments.
  """
  params_keys, params_vars, moments_keys, moments_vars = [], [], [], []

  with tf.variable_scope('batch_norm'):
    scope_name = tf.get_variable_scope().name

    if use_ema:
      ema_shape = [1, 1, 1, x.get_shape().as_list()[-1]]
      mean_ema = tf.get_variable(
          'mean_ema',
          shape=ema_shape,
          initializer=tf.initializers.zeros(),
          trainable=False)
      var_ema = tf.get_variable(
          'var_ema',
          shape=ema_shape,
          initializer=tf.initializers.ones(),
          trainable=False)

    if moments is not None:
      if backprop_through_moments:
        mean = moments[scope_name + '/mean']
        var = moments[scope_name + '/var']
      else:
        # This variant does not yield good resutls.
        mean = tf.stop_gradient(moments[scope_name + '/mean'])
        var = tf.stop_gradient(moments[scope_name + '/var'])
    elif use_ema and not is_training:
      mean = mean_ema
      var = var_ema
    else:
      # If not provided, compute the mean and var of the current batch.

      replica_ctx = tf.distribute.get_replica_context()
      if replica_ctx:
        # from third_party/tensorflow/python/keras/layers/normalization_v2.py
        axes = list(range(len(x.shape) - 1))
        local_sum = tf.reduce_sum(x, axis=axes, keepdims=True)
        local_squared_sum = tf.reduce_sum(
            tf.square(x), axis=axes, keepdims=True)
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        x_sum, x_squared_sum, global_batch_size = (
            replica_ctx.all_reduce('sum',
                                   [local_sum, local_squared_sum, batch_size]))

        axes_vals = [(tf.shape(x))[i] for i in range(1, len(axes))]
        multiplier = tf.cast(tf.reduce_prod(axes_vals), tf.float32)
        multiplier = multiplier * global_batch_size

        mean = x_sum / multiplier
        x_squared_mean = x_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        var = x_squared_mean - tf.square(mean)
      else:
        mean, var = tf.nn.moments(
            x, axes=list(range(len(x.shape) - 1)), keep_dims=True)

    # Only update ema's if training and we computed the moments in the current
    # call.  Note: at test time for episodic learners, ema's may be passed
    # from the support set to the query set, even if it's not really needed.
    if use_ema and is_training and moments is None:
      replica_ctx = tf.distribute.get_replica_context()
      mean_upd = tf.assign(mean_ema,
                           mean_ema * ema_epsilon + mean * (1.0 - ema_epsilon))
      var_upd = tf.assign(var_ema,
                          var_ema * ema_epsilon + var * (1.0 - ema_epsilon))
      updates = tf.group([mean_upd, var_upd])
      if replica_ctx:
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.cond(
                tf.equal(replica_ctx.replica_id_in_sync_group, 0),
                lambda: updates, tf.no_op))
      else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updates)

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


def weight_variable(shape, weight_decay):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.initializers.truncated_normal(stddev=0.1)
  return tf.get_variable(
      'weight',
      shape=shape,
      initializer=initial,
      regularizer=tf.keras.regularizers.L2(weight_decay))


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.initializers.constant(0.1)
  return tf.get_variable('bias', shape=shape, initializer=initial)


def dense(x, output_size, weight_decay, activation_fn=tf.nn.relu, params=None):
  """Fully connected layer implementation.

  Args:
    x: tf.Tensor, input.
    output_size: int, number features in  the fully connected layer.
    weight_decay: float, scaling constant for L2 weight decay on weight
      variables.
    activation_fn: function, to process pre-activations, namely x*w+b.
    params: None or a dict containing the values of the weight and bias params.
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
      w = weight_variable([input_size, output_size], weight_decay)
      b = bias_variable([output_size])
    else:
      w = params[w_name]
      b = params[b_name]

  x = tf.nn.xw_plus_b(x, w, b)
  params = collections.OrderedDict(zip([w_name, b_name], [w, b]))
  x = activation_fn(x)
  return x, params


def conv(x,
         conv_size,
         depth,
         stride,
         weight_decay,
         padding='SAME',
         rate=1,
         params=None):
  """A block that performs convolution."""
  params_keys, params_vars = [], []
  scope_name = tf.get_variable_scope().name
  input_depth = x.get_shape().as_list()[-1]
  if params is None:
    w_conv = weight_variable([conv_size[0], conv_size[1], input_depth, depth],
                             weight_decay)
  else:
    w_conv = params[scope_name + '/kernel']

  params_keys += [scope_name + '/kernel']
  params_vars += [w_conv]

  x = conv2d(x, w_conv, stride=stride, padding=padding, rate=rate)
  params = collections.OrderedDict(zip(params_keys, params_vars))

  return x, params


def conv_bn(x,
            conv_size,
            depth,
            stride,
            weight_decay,
            padding='SAME',
            params=None,
            moments=None,
            is_training=True,
            rate=1,
            backprop_through_moments=True):
  """A block that performs convolution, followed by batch-norm."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  x, conv_params = conv(
      x,
      conv_size,
      depth,
      stride,
      weight_decay,
      padding=padding,
      params=params,
      rate=rate)
  params_keys.extend(conv_params.keys())
  params_vars.extend(conv_params.values())

  x, bn_params, bn_moments = bn(
      x,
      params=params,
      moments=moments,
      is_training=is_training,
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
               stride,
               weight_decay,
               params=None,
               moments=None,
               use_project=False,
               backprop_through_moments=True,
               is_training=True,
               input_rate=1,
               output_rate=1,
               use_bounded_activation=False):
  """ResNet18 residual block."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []  # means and vars of different layers.
  with tf.variable_scope('conv1'):
    h, conv_bn_params, conv_bn_moments = conv_bn(
        x, [3, 3],
        depth[0],
        stride,
        weight_decay,
        params=params,
        moments=moments,
        is_training=is_training,
        rate=input_rate,
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
        weight_decay=weight_decay,
        params=params,
        moments=moments,
        is_training=is_training,
        rate=output_rate,
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
            weight_decay,
            params=params,
            moments=moments,
            is_training=is_training,
            rate=1,
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
            weight_decay,
            scope,
            reuse=tf.AUTO_REUSE,
            params=None,
            moments=None,
            backprop_through_moments=True,
            use_bounded_activation=False,
            blocks=(2, 2, 2, 2),
            max_stride=None,
            deeplab_alignment=True,
            keep_spatial_dims=False):
  """A ResNet network; ResNet18 by default."""
  x = tf.stop_gradient(x)
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  assert max_stride in [None, 4, 8, 16,
                        32], 'max_stride must be 4, 8, 16, 32, or None'
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
    if deeplab_alignment:
      size = tf.cast(tf.shape(x)[1], tf.float32)
      aligned_size = tf.cast(tf.ceil(size / 32.0), tf.int32) * 32 + 1
      x = tf.image.resize_bilinear(
          x, size=[aligned_size, aligned_size], align_corners=True)

    with tf.variable_scope('conv1'):
      x, conv_bn_params, conv_bn_moments = conv_bn(
          x, [7, 7],
          64,
          2,
          weight_decay,
          params=params,
          moments=moments,
          is_training=is_training,
          backprop_through_moments=backprop_through_moments)
      params_keys.extend(conv_bn_params.keys())
      params_vars.extend(conv_bn_params.values())
      moments_keys.extend(conv_bn_moments.keys())
      moments_vars.extend(conv_bn_moments.values())

      x = relu(x, use_bounded_activation=use_bounded_activation)

    def _bottleneck(x,
                    i,
                    depth,
                    stride,
                    params,
                    moments,
                    net_stride=1,
                    net_rate=1):
      """Wrapper for bottleneck."""
      input_rate = net_rate
      output_rate = input_rate
      if i == 0:
        if max_stride and stride * net_stride > max_stride:
          output_stride = 1
          output_rate *= stride
        else:
          output_stride = stride
      else:
        output_stride = 1
      use_project = True if i == 0 else False

      x, bottleneck_params, bottleneck_moments = bottleneck(
          x, (depth, depth),
          output_stride,
          weight_decay,
          params=params,
          moments=moments,
          input_rate=input_rate,
          output_rate=output_rate,
          use_project=use_project,
          is_training=is_training,
          backprop_through_moments=backprop_through_moments)
      net_stride *= output_stride
      return x, bottleneck_params, bottleneck_moments, net_stride, output_rate

    net_stride = 4
    net_rate = 1

    with tf.variable_scope('conv2_x'):
      x = tf.nn.max_pool(
          x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      for i in range(blocks[0]):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments, net_stride, net_rate = _bottleneck(
              x, i, 64, 1, params, moments, net_stride, net_rate)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv3_x'):
      for i in range(blocks[1]):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments, net_stride, net_rate = _bottleneck(
              x, i, 128, 2, params, moments, net_stride, net_rate)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv4_x'):
      for i in range(blocks[2]):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments, net_stride, net_rate = _bottleneck(
              x, i, 256, 2, params, moments, net_stride, net_rate)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())

    with tf.variable_scope('conv5_x'):
      for i in range(blocks[3]):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments, net_stride, net_rate = _bottleneck(
              x, i, 512, 2, params, moments, net_stride, net_rate)
          params_keys.extend(bottleneck_params.keys())
          params_vars.extend(bottleneck_params.values())
          moments_keys.extend(bottleneck_moments.keys())
          moments_vars.extend(bottleneck_moments.values())
    if not keep_spatial_dims:
      # x.shape: [?, 1, 1, 512]
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      x = tf.reshape(x, [-1, 512])
    params = collections.OrderedDict(zip(params_keys, params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))

    return_dict = {'embeddings': x, 'params': params, 'moments': moments}
    return return_dict


@gin.configurable(
    'resnet', allowlist=['weight_decay', 'max_stride', 'deeplab_alignment'])
def resnet(x,
           is_training,
           weight_decay,
           params=None,
           moments=None,
           reuse=tf.AUTO_REUSE,
           scope='resnet18',
           backprop_through_moments=True,
           use_bounded_activation=False,
           max_stride=None,
           deeplab_alignment=True,
           keep_spatial_dims=False):
  """ResNet18 embedding function."""
  return _resnet(
      x,
      is_training,
      weight_decay,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      blocks=(2, 2, 2, 2),
      max_stride=max_stride,
      deeplab_alignment=deeplab_alignment,
      keep_spatial_dims=keep_spatial_dims)


@gin.configurable(
    'resnet34', allowlist=['weight_decay', 'max_stride', 'deeplab_alignment'])
def resnet34(x,
             is_training,
             weight_decay,
             params=None,
             moments=None,
             reuse=tf.AUTO_REUSE,
             scope='resnet34',
             backprop_through_moments=True,
             use_bounded_activation=False,
             max_stride=None,
             deeplab_alignment=True,
             keep_spatial_dims=False):
  """ResNet34 embedding function."""
  return _resnet(
      x,
      is_training,
      weight_decay,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      blocks=(3, 4, 6, 3),
      max_stride=max_stride,
      deeplab_alignment=deeplab_alignment,
      keep_spatial_dims=keep_spatial_dims)


def wide_resnet_block(x,
                      depth,
                      stride,
                      weight_decay,
                      params=None,
                      moments=None,
                      use_project=False,
                      backprop_through_moments=True,
                      is_training=True,
                      use_bounded_activation=False):
  """Wide ResNet residual block."""
  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  with tf.variable_scope('conv1'):
    bn_1, bn_params, bn_moments = bn(
        x,
        params=params,
        moments=moments,
        is_training=is_training,
        backprop_through_moments=backprop_through_moments)
    params_keys.extend(bn_params.keys())
    params_vars.extend(bn_params.values())
    moments_keys.extend(bn_moments.keys())
    moments_vars.extend(bn_moments.values())

    out_1 = relu(bn_1, use_bounded_activation=use_bounded_activation)

    h_1, conv_params = conv(
        out_1, [3, 3], depth, stride, weight_decay, params=params)
    params_keys.extend(conv_params.keys())
    params_vars.extend(conv_params.values())
  with tf.variable_scope('conv2'):
    bn_2, bn_params, bn_moments = bn(
        h_1,
        params=params,
        moments=moments,
        is_training=is_training,
        backprop_through_moments=backprop_through_moments)
    params_keys.extend(bn_params.keys())
    params_vars.extend(bn_params.values())
    moments_keys.extend(bn_moments.keys())
    moments_vars.extend(bn_moments.values())

    out_2 = relu(bn_2, use_bounded_activation=use_bounded_activation)

    h_2, conv_params = conv(
        out_2, [3, 3],
        depth,
        stride=1,
        weight_decay=weight_decay,
        params=params)
    params_keys.extend(conv_params.keys())
    params_vars.extend(conv_params.values())

  h = h_2
  if use_bounded_activation:
    h = tf.clip_by_value(h, -6, 6)

  with tf.variable_scope('identity'):
    if use_project:
      with tf.variable_scope('projection_conv'):
        x, conv_params = conv(
            out_1, [1, 1], depth, stride, weight_decay, params=params)
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
                 weight_decay,
                 reuse=tf.AUTO_REUSE,
                 params=None,
                 moments=None,
                 backprop_through_moments=True,
                 use_bounded_activation=False,
                 keep_spatial_dims=False):
  """A wide ResNet."""
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
      x, conv_params = conv(x, [3, 3], 16, 1, weight_decay, params=params)
      _update_params_lists(conv_params, params_keys, params_vars)

    def _wide_resnet_block(x, depths, stride, use_project, moments):
      """Wrapper for a wide resnet block."""
      x, block_params, block_moments = wide_resnet_block(
          x,
          depths,
          stride,
          weight_decay,
          params=params,
          moments=moments,
          use_project=use_project,
          is_training=is_training,
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
          is_training=is_training,
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


@gin.configurable('wide_resnet', allowlist=['weight_decay'])
def wide_resnet(x,
                is_training,
                weight_decay,
                params=None,
                moments=None,
                reuse=tf.AUTO_REUSE,
                scope='wide_resnet',
                backprop_through_moments=True,
                use_bounded_activation=False,
                keep_spatial_dims=False):
  """A WideResNet embedding function."""
  return _wide_resnet(
      x,
      is_training,
      scope,
      2,
      2,
      weight_decay,
      reuse=reuse,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      keep_spatial_dims=keep_spatial_dims)


def _four_layer_convnet(inputs,
                        is_training,
                        scope,
                        weight_decay,
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
            weight_decay=weight_decay,
            params=params,
            moments=moments,
            is_training=is_training,
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


@gin.configurable('four_layer_convnet', allowlist=['weight_decay'])
def four_layer_convnet(inputs,
                       is_training,
                       weight_decay,
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
    weight_decay: float, scaling constant for L2 weight decay on weight
      variables.
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
  return _four_layer_convnet(
      inputs,
      is_training,
      scope,
      weight_decay=weight_decay,
      reuse=reuse,
      params=params,
      moments=moments,
      depth_multiplier=depth_multiplier,
      backprop_through_moments=backprop_through_moments,
      use_bounded_activation=use_bounded_activation,
      keep_spatial_dims=keep_spatial_dims)


@gin.configurable('relation_module', allowlist=['weight_decay'])
def relation_module(inputs,
                    is_training,
                    weight_decay,
                    scope='relation_module',
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
            1,
            weight_decay,
            params=params,
            moments=moments,
            is_training=is_training,
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
      layer, dense_params = dense(
          layer, 8, weight_decay, activation_fn=relu_activation_fn)
      tf.logging.info('Output layer_2_fc: %s' % layer.shape)
      model_params_keys.extend(dense_params.keys())
      model_params_vars.extend(dense_params.values())
    with tf.variable_scope('layer_3_fc', reuse=reuse):
      output, dense_params = dense(
          layer, 1, weight_decay, activation_fn=tf.nn.sigmoid)
      tf.logging.info('Output layer_3_fc: %s' % output.shape)
      model_params_keys.extend(dense_params.keys())
      model_params_vars.extend(dense_params.values())

    model_params = collections.OrderedDict(
        zip(model_params_keys, model_params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))
    return_dict = {'output': output, 'params': model_params, 'moments': moments}

    return return_dict


@gin.configurable('relationnet_convnet', allowlist=['weight_decay'])
def relationnet_convnet(inputs,
                        is_training,
                        weight_decay,
                        params=None,
                        moments=None,
                        depth_multiplier=1.0,
                        reuse=tf.AUTO_REUSE,
                        scope='relationnet_convnet',
                        backprop_through_moments=True,
                        use_bounded_activation=False,
                        keep_spatial_dims=False):
  """A 4-layer-convnet architecture for RelationNet embedding.

  This is almost like the `four_layer_convnet` embedding function except
  for the following differences: (1) no padding for the first 3 layers, (2) no
  maxpool on the last (4th) layer, and (3) no flatten.

  Paper: https://arxiv.org/abs/1711.06025
  Code:
  https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py

  Args:
    inputs: Tensors of shape [None, ] + image shape, e.g. [15, 84, 84, 3]
    is_training: Whether we are in the training phase.
    weight_decay: float, scaling constant for L2 weight decay on weight
      variables.
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
            weight_decay=weight_decay,
            padding='VALID' if i < 3 else 'SAME',
            params=params,
            moments=moments,
            is_training=is_training,
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


@gin.configurable(
    'fully_connected_network',
    allowlist=[
        'n_hidden_units',
        'use_batchnorm',
        'weight_decay',
    ])
def fully_connected_network(inputs,
                            is_training,
                            weight_decay,
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

  Args:
    inputs: Tensor of shape [None, num_features], where `num_features` is the
      number of input features.
    is_training: whether it's train or test mode (for batch norm).
    weight_decay: float, scaling constant for L2 weight decay on weight
      variables.
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
    backprop_through_moments: Whether to allow gradients to flow through the
      given support set moments. Only applies to non-transductive batch norm.
    keep_spatial_dims: is there only to match the interface.  This backbone
      cannot keep spatial dimensions, so it will fail if it's True.

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
  """
  assert not keep_spatial_dims
  layer = inputs
  model_params_keys, model_params_vars = [], []
  moments_keys, moments_vars = [], []
  activation_fn = functools.partial(
      relu, use_bounded_activation=use_bounded_activation)
  with tf.variable_scope(scope, reuse=reuse):
    for i, n_unit in enumerate(n_hidden_units):
      with tf.variable_scope('layer_%d' % i, reuse=reuse):
        layer, dense_params = dense(
            layer,
            n_unit,
            weight_decay,
            activation_fn=activation_fn,
            params=params)
        model_params_keys.extend(dense_params.keys())
        model_params_vars.extend(dense_params.values())
        if use_batchnorm:
          layer, bn_params, bn_moments = bn(
              layer,
              params=params,
              moments=moments,
              is_training=is_training,
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
