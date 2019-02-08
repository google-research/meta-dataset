# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
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

"""Learner related code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin.tf
import tensorflow as tf

MAX_WAY = 50  # The maximum number of classes we will see in any batch.


def conv2d(x, w, stride=1, b=None):
  """conv2d returns a 2d convolution layer with full stride."""
  h = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
  if b is not None:
    h += b
  return h


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
      offset = tf.get_variable('offset', initializer=tf.zeros_like(mean))
      scale = tf.get_variable('scale', initializer=tf.ones_like(var))
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
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(
      'weight', initializer=initial, regularizer=tf.nn.l2_loss)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable('bias', initializer=initial)


def conv(x, conv_size, depth, stride, params=None, maml_arch=False):
  """A block that performs convolution."""
  params_keys, params_vars = [], []
  scope_name = tf.get_variable_scope().name
  input_depth = x.get_shape().as_list()[-1]
  if params is None:
    w_conv = weight_variable([conv_size[0], conv_size[1], input_depth, depth])
    b_conv = bias_variable([depth]) if maml_arch else None
  else:
    w_conv = params[scope_name + '/kernel']
    b_conv = params[scope_name + '/bias']

  params_keys += [scope_name + '/kernel']
  params_vars += [w_conv]
  params_keys += [scope_name + '/bias']
  params_vars += [b_conv]

  x = conv2d(x, w_conv, stride=stride, b=b_conv)
  params = collections.OrderedDict(zip(params_keys, params_vars))

  return x, params


def conv_bn(x,
            conv_size,
            depth,
            stride,
            params=None,
            moments=None,
            maml_arch=False,
            backprop_through_moments=True):
  """A block that performs convolution, followed by batch-norm."""

  params_keys, params_vars = [], []
  moments_keys, moments_vars = [], []
  x, conv_params = conv(
      x, conv_size, depth, stride, params=params, maml_arch=maml_arch)
  params_keys += conv_params.keys()
  params_vars += conv_params.values()

  x, bn_params, bn_moments = bn(
      x,
      params=params,
      moments=moments,
      backprop_through_moments=backprop_through_moments)
  params_keys += bn_params.keys()
  params_vars += bn_params.values()
  moments_keys += bn_moments.keys()
  moments_vars += bn_moments.values()

  params = collections.OrderedDict(zip(params_keys, params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))

  return x, params, moments


def bottleneck(x,
               depth,
               stride=1,
               params=None,
               moments=None,
               maml_arch=False,
               use_project=False,
               backprop_through_moments=True):
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
        maml_arch=maml_arch,
        backprop_through_moments=backprop_through_moments)
    params_keys += conv_bn_params.keys()
    params_vars += conv_bn_params.values()
    moments_keys += conv_bn_moments.keys()
    moments_vars += conv_bn_moments.values()
    h = tf.nn.relu(h)

  with tf.variable_scope('conv2'):
    h, conv_bn_params, conv_bn_moments = conv_bn(
        h, [3, 3],
        depth[1],
        stride=1,
        params=params,
        moments=moments,
        maml_arch=maml_arch,
        backprop_through_moments=backprop_through_moments)
    params_keys += conv_bn_params.keys()
    params_vars += conv_bn_params.values()
    moments_keys += conv_bn_moments.keys()
    moments_vars += conv_bn_moments.values()

  with tf.variable_scope('identity'):
    if use_project:
      with tf.variable_scope('projection_conv'):
        x, conv_bn_params, conv_bn_moments = conv_bn(
            x, [1, 1],
            depth[1],
            stride,
            params=params,
            moments=moments,
            maml_arch=maml_arch,
            backprop_through_moments=backprop_through_moments)
        params_keys += conv_bn_params.keys()
        params_vars += conv_bn_params.values()
        moments_keys += conv_bn_moments.keys()
        moments_vars += conv_bn_moments.values()
    x = tf.nn.relu(x + h)

  params = collections.OrderedDict(zip(params_keys, params_vars))
  moments = collections.OrderedDict(zip(moments_keys, moments_vars))
  return x, params, moments


def _resnet(x,
            is_training,
            scope,
            reuse=tf.AUTO_REUSE,
            params=None,
            moments=None,
            maml_arch=False,
            backprop_through_moments=True):
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
    size = tf.to_float(tf.shape(x)[1])
    aligned_size = tf.to_int32(tf.ceil(size / 32.0)) * 32 + 1
    x = tf.image.resize_bilinear(
        x, size=[aligned_size, aligned_size], align_corners=True)

    with tf.variable_scope('conv1'):
      x, conv_bn_params, conv_bn_moments = conv_bn(
          x, [7, 7],
          64,
          2,
          params=params,
          moments=moments,
          maml_arch=maml_arch,
          backprop_through_moments=backprop_through_moments)
      params_keys += conv_bn_params.keys()
      params_vars += conv_bn_params.values()
      moments_keys += conv_bn_moments.keys()
      moments_vars += conv_bn_moments.values()
      x = tf.nn.relu(x)

    def _bottleneck(x, i, depth, params, moments, stride=2):
      """Wrapper for bottleneck."""
      output_stride = stride if i == 0 else 1
      use_project = True if i == 0 else False
      x, bottleneck_params, bottleneck_moments = bottleneck(
          x, (depth, depth),
          output_stride,
          params=params,
          moments=moments,
          maml_arch=maml_arch,
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
          params_keys += bottleneck_params.keys()
          params_vars += bottleneck_params.values()
          moments_keys += bottleneck_moments.keys()
          moments_vars += bottleneck_moments.values()

    with tf.variable_scope('conv3_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 128, params, moments)
          params_keys += bottleneck_params.keys()
          params_vars += bottleneck_params.values()
          moments_keys += bottleneck_moments.keys()
          moments_vars += bottleneck_moments.values()

    with tf.variable_scope('conv4_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 256, params, moments)
          params_keys += bottleneck_params.keys()
          params_vars += bottleneck_params.values()
          moments_keys += bottleneck_moments.keys()
          moments_vars += bottleneck_moments.values()

    with tf.variable_scope('conv5_x'):
      for i in range(2):
        with tf.variable_scope('bottleneck_%d' % i):
          x, bottleneck_params, bottleneck_moments = _bottleneck(
              x, i, 512, params, moments)
          params_keys += bottleneck_params.keys()
          params_vars += bottleneck_params.values()
          moments_keys += bottleneck_moments.keys()
          moments_vars += bottleneck_moments.values()

    x = tf.reduce_mean(x, axis=[1, 2])
    x = tf.reshape(x, [-1, 512])
    params = collections.OrderedDict(zip(params_keys, params_vars))
    moments = collections.OrderedDict(zip(moments_keys, moments_vars))

    return_dict = {'embeddings': x, 'params': params, 'moments': moments}
    return return_dict


def resnet(x,
           is_training,
           moments=None,
           reuse=tf.AUTO_REUSE,
           scope='resnet18',
           backprop_through_moments=True):
  return _resnet(
      x,
      is_training,
      scope,
      reuse=reuse,
      params=None,
      moments=moments,
      maml_arch=False,
      backprop_through_moments=backprop_through_moments)


def resnet_maml(x,
                params=None,
                moments=None,
                depth_multiplier=1.0,
                reuse=tf.AUTO_REUSE,
                scope='resnet_maml',
                backprop_through_moments=True):
  del depth_multiplier
  return _resnet(
      x,
      True,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      maml_arch=True,
      backprop_through_moments=backprop_through_moments)


def four_layer_convnet(inputs,
                       is_training,
                       moments=None,
                       depth_multiplier=1.0,
                       reuse=tf.AUTO_REUSE,
                       scope='four_layer_convnet',
                       backprop_through_moments=True):
  """Embeds inputs using a standard four-layer convnet.

  Args:
    inputs: Tensors of shape [None, ] + image shape, e.g. [15, 84, 84, 3]
    is_training: Whether we are in the training phase.
    moments: A dict of the means and vars of the different layers to use for
      batch normalization. If not provided, the mean and var are computed based
      on the given inputs.
    depth_multiplier: The depth multiplier for the convnet channels.
    reuse: Whether to reuse the network's weights.
    scope: An optional scope for the tf operations.
    backprop_through_moments: Whether to allow gradients to flow through the
      given support set moments. Only applies to non-transductive batch norm.

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
  """
  del is_training
  return _four_layer_convnet(
      inputs,
      scope,
      reuse=reuse,
      params=None,
      moments=moments,
      maml_arch=False,
      depth_multiplier=depth_multiplier,
      backprop_through_moments=backprop_through_moments)


def four_layer_convnet_maml(inputs,
                            params=None,
                            moments=None,
                            depth_multiplier=1.0,
                            reuse=tf.AUTO_REUSE,
                            scope='four_layer_convnet_maml',
                            backprop_through_moments=True):
  """Embeds inputs using a standard four-layer convnet for the MAML model.

  Args:
    inputs: Tensors of shape [None, ] + image shape, e.g. [15, 84, 84, 3]
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

  Returns:
    A 2D Tensor, where each row is the embedding of an input in inputs.
    A dictionary that maps model parameter name to the TF variable.
  """
  return _four_layer_convnet(
      inputs,
      scope,
      reuse=reuse,
      params=params,
      moments=moments,
      maml_arch=True,
      depth_multiplier=depth_multiplier,
      backprop_through_moments=backprop_through_moments)


NAME_TO_EMBEDDING_NETWORK = {
    'resnet': resnet,
    'resnet_maml': resnet_maml,
    'four_layer_convnet': four_layer_convnet,
    'four_layer_convnet_maml': four_layer_convnet_maml,
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
    super(PrototypicalNetworkLearner, self).__init__(
        is_training, transductive_batch_norm, backprop_through_moments,
        ema_object, embedding_fn, reader)

    # The data for the next episode.
    self.episode = self.data
    self.test_targets = self.episode.test_labels
    self.way = compute_way(self.data)

    # Hyperparameters.
    self.weight_decay = weight_decay
    tf.logging.info('PrototypicalLearner: weight_decay {}'.format(weight_decay))

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

  def compute_prototypes(self):
    """Computes the class prototypes."""
    # [num train images, 1, embedding size].
    train_embeddings = tf.expand_dims(self.train_embeddings, 1)

    # [num train labels, num classes] where each row is a one-hot-encoded label.
    one_hot_train_labels = tf.one_hot(self.episode.train_labels, self.way)
    # [num train labels, num classes, 1].
    one_hot_train_labels = tf.expand_dims(one_hot_train_labels, 2)

    # Sums each class' embeddings. [num classes, embedding size].
    class_sums = tf.reduce_sum(one_hot_train_labels * train_embeddings, 0)

    # The prototype of each class is the average embedding of its train points.
    class_num_images = tf.reduce_sum(one_hot_train_labels, 0)  # [way].
    prototypes = class_sums / class_num_images
    return prototypes

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
    self.prototypes = self.compute_prototypes()
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
    super(MatchingNetworkLearner, self).__init__(
        is_training, transductive_batch_norm, backprop_through_moments,
        ema_object, embedding_fn, reader, weight_decay)

    self.exact_cosine_distance = exact_cosine_distance
    self.weight_decay = weight_decay
    tf.logging.info(
        'MatchingNetworkLearner: weight_decay {}'.format(weight_decay))

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
    one_hot_train_labels = tf.one_hot(self.data.train_labels, self.way)

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
    probs = tf.matmul(attention, one_hot_train_labels)
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
               knn_distance):
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
    """
    super(BaselineLearner, self).__init__(is_training, transductive_batch_norm,
                                          backprop_through_moments, ema_object,
                                          embedding_fn, reader)
    if self.embedding_fn is four_layer_convnet_maml:

      def wrapped_four_layer_convnet_maml(inputs,
                                          is_training,
                                          moments=None,
                                          backprop_through_moments=True,
                                          reuse=tf.AUTO_REUSE,
                                          scope='four_layer_convnet_maml'):
        del is_training
        return four_layer_convnet_maml(
            inputs,
            moments=moments,
            reuse=reuse,
            scope=scope,
            backprop_through_moments=backprop_through_moments)

      self.embedding_fn = wrapped_four_layer_convnet_maml

    if self.embedding_fn is resnet_maml:

      def wrapped_resnet_maml(inputs,
                              is_training,
                              moments=None,
                              backprop_through_moments=True,
                              reuse=tf.AUTO_REUSE,
                              scope='resnet_maml'):
        del is_training
        return resnet_maml(
            inputs,
            moments=moments,
            reuse=reuse,
            scope=scope,
            backprop_through_moments=backprop_through_moments)

      self.embedding_fn = wrapped_resnet_maml

    self.num_train_classes = num_train_classes
    self.num_test_classes = num_test_classes

    # Hyperparameters.
    self.weight_decay = weight_decay
    self.distance = knn_distance
    tf.logging.info('BaselineLearner: distance {}, weight_decay {}'.format(
        knn_distance, weight_decay))

    self.forward_pass()

    if not self.is_training:
      # For aggregating statistics later.
      self.test_targets = self.data.test_labels
      self.way = compute_way(self.data)
      self.knn_in_fc = knn_in_fc
      tf.logging.info('BaselineLearner: knn_in_fc {}'.format(knn_in_fc))

  def forward_pass(self):
    if self.is_training:
      images = self.data.images
      embeddings = self.embedding_fn(images, self.is_training)['embeddings']
      self.train_embeddings = embeddings
    else:
      train_embeddings = self.embedding_fn(self.data.train_images,
                                           self.is_training)
      self.train_embeddings = train_embeddings['embeddings']
      support_set_moments = None
      if not self.transductive_batch_norm:
        support_set_moments = train_embeddings['moments']
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
      embedding_dims = embeddings.get_shape().as_list()[-1]
      # Always maps to a space whose dimensionality is the number of classes
      # at meta-training time.
      w_fc = weight_variable([embedding_dims, self.num_train_classes])
      b_fc = bias_variable([self.num_train_classes])
      return tf.matmul(embeddings, w_fc) + b_fc

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
      labels = tf.to_int64(self.data.labels)
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
               debug_log=False):
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
    """
    self.num_finetune_steps = num_finetune_steps
    self.finetune_lr = finetune_lr
    self.debug_log = debug_log
    self.finetune_opt = tf.train.AdamOptimizer(self.finetune_lr)
    # Note: the weight_decay value provided here overrides the value gin might
    # have for BaselineLearner's own weight_decay.
    super(BaselineFinetuneLearner, self).__init__(
        is_training, transductive_batch_norm, backprop_through_moments,
        ema_object, embedding_fn, reader, num_train_classes, num_test_classes,
        weight_decay)

  def compute_logits(self):
    """Computes the logits."""
    logits = None
    if self.is_training:
      logits = self.forward_pass_fc(self.train_embeddings)
    else:
      self.train_logits = self._fc_layer(self.train_embeddings)[:, 0:self.way]
      # ------------------------ Finetuning -------------------------------
      finetune_loss = self._classification_loss(
          self.train_logits, self.data.train_labels, self.way)
      vars_to_finetune = []
      for var in tf.all_variables():
        if 'fc_finetune' in var.name:
          vars_to_finetune.append(var)

      for i in range(self.num_finetune_steps):
        if i == 0:
          fc_reset = tf.variables_initializer(var_list=vars_to_finetune)
          # Adam related variables are created when minimize() is called.
          # We create an unused op here to put all adam varariables under
          # the 'adam_opt' namescope and create a reset op to reinitialize
          # these variables before the first finetune step.
          with tf.variable_scope('adam_opt'):
            unused_op = self.finetune_opt.minimize(
                finetune_loss, var_list=vars_to_finetune)
          adam_reset = tf.variables_initializer(self.finetune_opt.variables())
          with tf.control_dependencies([fc_reset, adam_reset, finetune_loss] +
                                       vars_to_finetune):
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                  finetune_loss
              ])
            else:
              print_op = tf.no_op()
            with tf.control_dependencies([print_op]):
              logits = self._fc_layer(self.train_embeddings)[:, 0:self.way]
              test_logits = self._fc_layer(self.test_embeddings)[:, 0:self.way]
              finetune_loss = self._classification_loss(
                  logits, self.data.train_labels, self.way)
              finetune_op = self.finetune_opt.minimize(
                  finetune_loss, var_list=vars_to_finetune)
        else:
          with tf.control_dependencies([finetune_op, finetune_loss] +
                                       vars_to_finetune):
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                  finetune_loss, 'accuracy:',
                  self._compute_accuracy(logits, self.data.train_labels),
                  'test accuracy:',
                  self._compute_accuracy(test_logits, self.data.test_labels)
              ])
            else:
              print_op = tf.no_op()
            with tf.control_dependencies([print_op]):
              logits = self._fc_layer(self.train_embeddings)[:, 0:self.way]
              test_logits = self._fc_layer(self.test_embeddings)[:, 0:self.way]
              finetune_loss = self._classification_loss(
                  logits, self.data.train_labels, self.way)
              finetune_op = self.finetune_opt.minimize(
                  finetune_loss, var_list=vars_to_finetune)

      with tf.control_dependencies([finetune_op] + vars_to_finetune):
        logits = self._fc_layer(self.train_embeddings)[:, 0:self.way]
        test_logits = self._fc_layer(self.test_embeddings)[:, 0:self.way]
        if self.debug_log:
          print_op = tf.print([
              'accuracy: ',
              self._compute_accuracy(logits, self.data.train_labels),
              'test accuracy: ',
              self._compute_accuracy(test_logits, self.data.test_labels)
          ])
        else:
          print_op = tf.no_op()
        with tf.control_dependencies([print_op]):
          self.test_logits = self._fc_layer(self.test_embeddings)[:, 0:self.way]
          logits = self.test_logits
    return logits

  def _fc_layer(self, embedding):
    """The fully connected layer to be finetuned."""
    with tf.variable_scope('fc_finetune', reuse=tf.AUTO_REUSE):
      embedding_dims = embedding.get_shape().as_list()[-1]
      w_fc = weight_variable([embedding_dims, MAX_WAY])
      b_fc = bias_variable([MAX_WAY])
      y = tf.matmul(embedding, w_fc) + b_fc
      return y

  def _classification_loss(self, logits, labels, num_classes):
    """Computes softmax cross entropy loss."""
    labels = tf.to_int64(labels)
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
               first_order, alpha, train_batch_norm, depth_multiplier, debug,
               zero_fc_layer, proto_maml_fc_layer_on_query_set,
               proto_maml_fc_layer_on_support_set, proto_maml_fc_layer_init):
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
      depth_multiplier: The depth multiplier to use for convnet filters.
      debug: If True, print out debug logs.
      zero_fc_layer: Whether to use zero fc layer initialization.
      proto_maml_fc_layer_on_query_set: Whether to use ProtoNets equivalent fc
        layer on query set.
      proto_maml_fc_layer_on_support_set: Whether to use ProtoNets equivalent fc
        layer on support set.
      proto_maml_fc_layer_init: Whether to use ProtoNets equivalent fc layer
        initialization.

    Raises:
      ValueError: The embedding function must be MAML-compatible.
    """
    super(MAMLLearner, self).__init__(is_training, transductive_batch_norm,
                                      backprop_through_moments, ema_object,
                                      embedding_fn, reader)

    maml_embedding_fns = [
        four_layer_convnet_maml,
        resnet_maml,
    ]
    if not any(self.embedding_fn is maml_fn for maml_fn in maml_embedding_fns):
      raise ValueError('MAML requires a specific architecture to work.')

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
    self.depth_multiplier = depth_multiplier
    self.zero_fc_layer = zero_fc_layer
    self.proto_maml_fc_layer_on_query_set = proto_maml_fc_layer_on_query_set
    self.proto_maml_fc_layer_on_support_set = proto_maml_fc_layer_on_support_set
    self.proto_maml_fc_layer_init = proto_maml_fc_layer_init

    tf.logging.info('alpha: {}, num_update_steps: {}'.format(
        self.alpha, self.num_update_steps))
    self.forward_pass()

  # TODO(tylerzhu): Refactor this method. An extremely similar function is
  # implemented as a method of the PrototypicalNetworkLearner.
  def proto_maml_prototypes(self, train_embeddings):
    """Computes the support-set prototypes.

    Args:
      train_embeddings: Tensor of shape [num_train_images, embedding_size]

    Returns:
      prototypes: Tensor of shape [num_classes, embedding_size]
    """
    # [num train images, 1, embedding size].
    train_embeddings = tf.expand_dims(train_embeddings, 1)
    # [num train labels, num classes] where each row is a one-hot-encoded label.
    one_hot_train_labels = tf.one_hot(self.data.train_labels, self.way)
    # [num train labels, num classes, 1].
    one_hot_train_labels = tf.expand_dims(one_hot_train_labels, 2)
    # Sums each class' embeddings. [num classes, embedding size].
    class_sums = tf.reduce_sum(one_hot_train_labels * train_embeddings, 0)
    # The prototype of each class is the average embedding of its train points.
    class_num_images = tf.reduce_sum(one_hot_train_labels, 0)  # [way].
    prototypes = class_sums / class_num_images
    return prototypes

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
    # Have to use one_hot labels since sparse softmax doesn't allow
    # second derivatives.
    one_hot_train_labels = tf.one_hot(self.data.train_labels, self.way)
    train_embeddings_ = self.embedding_fn(
        self.data.train_images,
        depth_multiplier=self.depth_multiplier,
        reuse=tf.AUTO_REUSE)
    train_embeddings = train_embeddings_['embeddings']
    embedding_vars_dict = train_embeddings_['params']

    with tf.variable_scope('linear_classifier', reuse=tf.AUTO_REUSE):
      embedding_depth = train_embeddings.shape.as_list()[-1]
      fc_weights = weight_variable([embedding_depth, MAX_WAY])
      fc_bias = bias_variable([MAX_WAY])

    embedding_vars_keys = []
    embedding_vars = []
    embedding_vars_copy_ops = []
    for name, var in embedding_vars_dict.iteritems():
      embedding_vars_keys.append(name)
      if not self.is_training:
        with tf.variable_scope('weight_copy'):
          shape = var.shape.as_list()
          var_copy = tf.Variable(
              tf.zeros(shape), collections=[tf.GraphKeys.LOCAL_VARIABLES])
          var_copy_op = tf.assign(var_copy, var)
          embedding_vars_copy_ops.append(var_copy_op)
        embedding_vars.append(var_copy)
      else:
        embedding_vars.append(var)

    fc_vars_copy_ops = []
    if not self.is_training:
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
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          depth_multiplier=self.depth_multiplier,
          reuse=True)['embeddings']

      if self.proto_maml_fc_layer_on_support_set:
        # Set fc layer weights with prototypical equivalent values.
        prototypes = self.proto_maml_prototypes(train_embeddings)
        pmaml_fc_weights = self.proto_maml_fc_weights(
            prototypes, zero_pad_to_max_way=True)
        pmaml_fc_bias = self.proto_maml_fc_bias(
            prototypes, zero_pad_to_max_way=True)
        train_logits = tf.matmul(train_embeddings,
                                 pmaml_fc_weights) + pmaml_fc_bias
      else:
        updated_fc_weights, updated_fc_bias = updated_fc_vars
        train_logits = tf.matmul(train_embeddings,
                                 updated_fc_weights) + updated_fc_bias

      train_logits = train_logits[:, 0:self.way]
      loss = tf.losses.softmax_cross_entropy(one_hot_train_labels, train_logits)

      if self.debug_log:
        print_op = tf.print(['step: ', step, updated_fc_bias[0], 'loss:', loss])
      else:
        print_op = tf.no_op()

      embedding_grads = tf.gradients(loss, updated_embedding_vars)
      # Only computes fc grad when it's not created from prototypes.
      if not self.proto_maml_fc_layer_on_support_set:
        fc_grads = tf.gradients(loss, updated_fc_vars)

      if self.first_order:

        def _stop_grads(grads):
          return [tf.stop_gradient(dv) for dv in grads]

        embedding_grads = _stop_grads(embedding_grads)
        if not self.proto_maml_fc_layer_on_support_set:
          fc_grads = _stop_grads(fc_grads)

      # Apply gradients
      def _apply_grads(variables, grads):
        """Applies gradients using SGD on a list of variables."""
        v_new = []
        for (v, dv) in zip(variables, grads):
          if (not self.train_batch_norm and
              ('offset' in v.name or 'scale' in v.name)):
            v_new.append(v)
          else:
            v_new.append(v - self.alpha * dv)
        return v_new

      with tf.control_dependencies([print_op]):
        updated_embedding_vars = _apply_grads(updated_embedding_vars,
                                              embedding_grads)
        # Only apply fc grad when it's not created from prototypes.
        if not self.proto_maml_fc_layer_on_support_set:
          updated_fc_vars = _apply_grads(updated_fc_vars, fc_grads)
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
          params=collections.OrderedDict(
              zip(embedding_vars_keys, embedding_vars)),
          depth_multiplier=self.depth_multiplier,
          reuse=True)['embeddings']
      prototypes = self.proto_maml_prototypes(train_embeddings)
      pmaml_fc_weights = self.proto_maml_fc_weights(
          prototypes, zero_pad_to_max_way=True)
      pmaml_fc_bias = self.proto_maml_fc_bias(
          prototypes, zero_pad_to_max_way=True)
      fc_vars = [pmaml_fc_weights, pmaml_fc_bias]

    with tf.control_dependencies(fc_vars_init_ops + embedding_vars_copy_ops):
      # We will first compute gradients using the initial weights
      # Don't want to restore it during eval.
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
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          depth_multiplier=self.depth_multiplier,
          reuse=True)['moments']

    test_embeddings = self.embedding_fn(
        self.data.test_images,
        params=collections.OrderedDict(
            zip(embedding_vars_keys, updated_embedding_vars)),
        moments=support_set_moments,  # Use support set stats for batch norm.
        depth_multiplier=self.depth_multiplier,
        reuse=True,
        backprop_through_moments=self.backprop_through_moments)['embeddings']

    if not self.proto_maml_fc_layer_on_query_set:
      self.test_logits = (tf.matmul(test_embeddings, updated_fc_weights) +
                          updated_fc_bias)[:, 0:self.way]
    else:
      train_embeddings = self.embedding_fn(
          self.data.train_images,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          depth_multiplier=self.depth_multiplier,
          reuse=True)['embeddings']
      prototypes = self.proto_maml_prototypes(train_embeddings)
      pmaml_fc_weights = self.proto_maml_fc_weights(prototypes)
      pmaml_fc_bias = self.proto_maml_fc_bias(prototypes)
      self.test_logits = (
          tf.matmul(test_embeddings, pmaml_fc_weights) + pmaml_fc_bias)

  def compute_loss(self):
    one_hot_test_labels = tf.one_hot(self.data.test_labels, self.way)
    loss = tf.losses.softmax_cross_entropy(one_hot_test_labels,
                                           self.test_logits)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self):
    """Computes the accuracy on the given episode."""
    self.test_predictions = tf.cast(tf.argmax(self.test_logits, 1), tf.int32)
    correct = tf.equal(self.data.test_labels, self.test_predictions)
    return tf.reduce_mean(tf.cast(correct, tf.float32))
