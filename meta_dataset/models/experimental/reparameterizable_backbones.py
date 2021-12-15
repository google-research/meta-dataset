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

# Lint as: python3
"""Mappings from images to embeddings or embeddings to predictions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin.tf
from meta_dataset.models.experimental import reparameterizable_base
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa


@gin.configurable
class ReparameterizableBackbone(tf.keras.Sequential,
                                reparameterizable_base.ReparameterizableModule):
  """A reparameterizable module consisting of a linear stack of layers."""

  def __init__(self,
               output_dim=None,
               kernel_initializer=tf.keras.initializers.he_normal(),
               kernel_regularizer=tf.keras.regularizers.l2(),
               parameter_adapter=None,
               name=None):
    """Creates a `ReparameterizableBackbone`.

    Args:
      output_dim: If not None, the output dimensionality of the projection layer
        that is appended to this `ReparameterizableBackbone`'s stack.
      kernel_initializer: A subclass of `tf.keras.regularizer.Initializer`, used
        as a default initializer for kernel (weight) matrices.
      kernel_regularizer: A subclass of `tf.keras.regularizer.Regularizer`, used
        as a default regularizer for kernel (weight) matrices.
      parameter_adapter: A callable taking a single `tf.Module` argument that
        restores values for the variables belonging to that `tf.Module`; see
        `meta_dataset.models.experimental.parameter_adapter.py`.
      name: The name for this `ReparameterizableBackbone`.
    """
    self.output_dim = output_dim
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.parameter_adapter = parameter_adapter

    super(ReparameterizableBackbone, self).__init__(
        layers=self.stack + self.output_layers, name=name)

  def build(self, input_shape):
    """Wraps the `build` method of `tf.keras.Sequential` to load parameters."""
    super(ReparameterizableBackbone, self).build(input_shape=input_shape)
    if self.parameter_adapter is not None:
      self.parameter_adapter(self)

  @property
  def stack(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  @property
  def output_layers(self):
    """Returns a projection layer if `self.output_dim` is specified."""
    if self.output_dim:
      return [
          tf.keras.layers.Dense(
              self.output_dim,
              activation=None,
              kernel_regularizer=self.kernel_regularizer,
              # Initialize as a an output logit layer.
              kernel_initializer=tf.keras.initializers.normal(stddev=0.01),
              name='output_layer',
          )
      ]
    else:
      return []


@gin.configurable
class ReparameterizableSpatialBackbone(ReparameterizableBackbone):
  """A reparameterizable backbone that preserves or flattens spatial axes."""

  def __init__(self, keep_spatial_dims=False, **kwargs):
    """Creates a `ReparameterizableSpatialBackbone`.

    Args:
      keep_spatial_dims: If True, the spatial (non-batch) axes of the output of
        this `ReparameterizableSpatialBackbone`'s stack is preserved; if False,
        the spatial axes are flattened into a one-dimensional Tensor.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.

    Raises:
      ValueError: if `keep_spatial_dims` is True and a non-None value of
      `output_dim` was provided via `kwargs` to the parent constructor.
    """
    self.keep_spatial_dims = keep_spatial_dims
    super(ReparameterizableSpatialBackbone, self).__init__(**kwargs)
    if keep_spatial_dims and self.output_dim is not None:
      raise ValueError(
          'Cannot simultaneously apply a linear projection and preserve '
          'spatial dimensions.')

  @property
  def output_layers(self):
    """Prefix a flatten operation to `ReparameterizableModule.output_layers`."""
    if not self.keep_spatial_dims:
      return [tf.keras.layers.Flatten()] + super(
          ReparameterizableSpatialBackbone, self).output_layers
    else:
      return super(ReparameterizableSpatialBackbone, self).output_layers


@gin.configurable
class LinearModel(ReparameterizableBackbone):
  """Linear model with weights and biases."""

  def __init__(self, output_dim, **kwargs):
    """Creates a `LinearModel`.

    Args:
      output_dim: The output dimensionality of this `LinearModel`.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.

    Raises:
      ValueError: if `output_dim` is `None`.
    """
    if output_dim is None:
      raise ValueError(
          'Output dimensionality must be specified for a `LinearModel`.')
    super(LinearModel, self).__init__(output_dim=output_dim, **kwargs)

  @property
  def stack(self):
    """Returns no transformation in addition to `LinearModel.output_layers`."""
    return []


@gin.configurable
class FullyConnectedNet(ReparameterizableBackbone):
  """Fully-connected neural network with one or more hidden layers.

  #### Examples

  The two-layer, 40-dimensional fully-connected network used in the sinusoidal
  regression experiments in [Finn et al. (2017)][1] can be created as follows:

  ```python
  net = FullyConnectedNet(num_hiddens_per_layer=(40, 40,), output_dim=5)
  ```

  #### References

  [1]: Finn, Chelsea, Pieter Abbeel, and Sergey Levine. Model-agnostic
       meta-learning for fast adaptation of deep networks. In _Proceedings of
       the 34th International Conference on Machine Learning-Volume 70_, 2017.
       https://arxiv.org/abs/1703.03400
  """

  def __init__(self, num_hiddens_per_layer, **kwargs):
    """Creates a `FullyConnectedNet`.

    Args:
      num_hiddens_per_layer: An iterable of scalars; the dimensionality of each
        hidden layer, with the length of `num_hidden` equal to the number of
        hidden layers in this `FullyConnectedNet`.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.
    """
    self.num_hiddens_per_layer = num_hiddens_per_layer
    super(FullyConnectedNet, self).__init__(**kwargs)

  @property
  def stack(self):
    """Returns a stack of hidden layers with ReLU activation."""
    stack = []
    for num_hiddens in self.num_hiddens_per_layer:
      stack += [
          tf.keras.layers.Dense(
              units=num_hiddens,
              activation=tf.nn.relu,
              kernel_regularizer=self.kernel_regularizer,
              kernel_initializer=self.kernel_initializer,
          )
      ]
    return stack


@gin.configurable
class ConvNet(ReparameterizableSpatialBackbone):
  """Convolutional neural network with batch norm and max-pooling.

  #### Examples

  The four-layer, 32-dimensional convolutional network used in the miniImageNet
  few-shot classification experiments in [Finn et al. (2017)][1] can be created
  as follows:

  ```python
  net = ConvNet(num_filters_per_layer=(32, 32, 32, 32), output_dim=5)
  ```

  The four-layer, 64-dimensional convolutional network used in the meta-dataset
  few-shot classification experiments in [Triantafillou et al. (2020)][2] can be
  created as follows:

  ```python
  net = ConvNet(num_filters_per_layer=(64, 64, 64, 64), output_dim=5)
  ```

  #### References

  [1]: Finn, Chelsea, Pieter Abbeel, and Sergey Levine. Model-agnostic
       meta-learning for fast adaptation of deep networks. In _Proceedings of
       the 34th International Conference on Machine Learning-Volume 70_, 2017.
       https://arxiv.org/abs/1703.03400

  [2]: Triantafillou, Eleni, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku
       Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky,
       Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of
       datasets for learning to learn from few examples. In _Proceedings of 8th
       International Conference on Learning Representations_, 2020.
       https://arxiv.org/abs/1903.03096
  """

  def __init__(self, num_filters_per_layer, **kwargs):
    """Creates a `ConvNet`.

    Args:
      num_filters_per_layer: An iterable of scalars; the number  of filters in
        each convolutional layer, with the length of `num_filters_per_layer`
        equal to the number of convolutional layers in this `ConvNet`.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.
    """
    self.num_filters_per_layer = num_filters_per_layer
    super(ConvNet, self).__init__(**kwargs)

  @property
  def stack(self):
    """Returns a stack of convolutional layers with batch norm and ReLU."""
    stack = []
    for num_filters in self.num_filters_per_layer:
      stack += [
          tf.keras.layers.Conv2D(
              filters=num_filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              use_bias=False,  # shift handled by batchnorm
              padding='same',
              kernel_initializer=self.kernel_initializer,
              kernel_regularizer=self.kernel_regularizer,
          ),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      ]
    return stack


@gin.configurable
class RelationNetConvNet(ReparameterizableSpatialBackbone):
  """Architecture for embedding module in [Sung et al.

  (2018)][1].

  This implementation is adapted from
  https://github.com/floodsung/LearningToCompare_FSL/blob/b74e6366c7e7208a9d0fc82f7bedb9e1000199e4/miniimagenet/miniimagenet_train_one_shot.py#L55.

  #### References

  [1]: Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and
       Timothy M.  Hospedales. Learning to compare: Relation network for
       few-shot learning. In _Proceedings of the IEEE Conference on Computer
       Vision and Pattern Recognition_, 2018. https://arxiv.org/abs/1711.06025
  """

  @property
  def stack(self):
    """Returns the RelationNet-specific convolutional embedding stack."""
    return [
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ]


@gin.configurable
class RelationModule(ReparameterizableBackbone):
  """Architecture for the relation module in [Sung et al.

  (2018)][1].

  This implementation is adapted from
  https://github.com/floodsung/LearningToCompare_FSL/blob/b74e6366c7e7208a9d0fc82f7bedb9e1000199e4/miniimagenet/miniimagenet_train_one_shot.py#L86.

  #### References

  [1]: Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and
       Timothy M.  Hospedales. Learning to compare: Relation network for
       few-shot learning. In _Proceedings of the IEEE Conference on Computer
       Vision and Pattern Recognition_, 2018. https://arxiv.org/abs/1711.06025
  """

  @property
  def stack(self):
    """Returns the RelationNet relation module stack."""
    return [
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=False,  # shift handled by batchnorm
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]


class BasicBlock(tf.keras.Model):
  """Abstract basic residual block that instantiates parameters for subclasses.

  Version 1 [He et al. (2016)][1] and version 2 [He et al. (2016)][2] of the
  basic residual block are implemented as subclasses of this abstract class.

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385

  [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity mappings in
      deep residual networks. In _Proceedings of the European Conference on
      Computer Vision_, 2016. https://arxiv.org/abs/1603.05027

  """
  expansion_factor = 1
  normalization_layer = tf.keras.layers.BatchNormalization

  def __init__(self,
               num_channels,
               width_multiplier,
               strides,
               use_projection,
               kernel_initializer=tf.keras.initializers.he_normal(),
               kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
               name='basic_residual_block'):
    """Constructs a `BasicBlock`.

    Args:
      num_channels: An integer specifying the base number of filters in each
        convolutional layer in this `BasicBlock`.
      width_multiplier: An integer specifying the factor by which to increase
        the number of filters in the 3x3 convolution in this `BasicBlock`.
      strides: An integer or an iterable of integers, specifying the stride(s)
        of the 3x3 convolution in this `BasicBlock`.
      use_projection: If True, the additive residual component is projected with
        a 1x1 convolution with stride `strides`; otherwise, it is simply the
        input.
      kernel_initializer: A subclass of `tf.keras.regularizer.Initializer`, used
        as a default initializer for kernel (weight) matrices.
      kernel_regularizer: A subclass of `tf.keras.regularizer.Regularizer`, used
        as a default regularizer for kernel (weight) matrices.
      name: The name for this `BasicBlock`.
    """
    super(BasicBlock, self).__init__(name=name)
    self.use_projection = use_projection

    self.padding_0 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
    self.conv_0 = tf.keras.layers.Conv2D(
        filters=num_channels * width_multiplier,
        kernel_size=3,
        strides=strides,
        use_bias=False,
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    self.bn_0 = self.normalization_layer(epsilon=1e-5)
    self.act_0 = tf.keras.layers.ReLU()

    self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
    self.conv_1 = tf.keras.layers.Conv2D(
        filters=num_channels * width_multiplier,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    # Zero-initialize the weights of the last batch-normalization layer in each
    # residual block, so that the (non-projected) residual branch starts with a
    # zero output and so each residual block starts as an identity mapping.
    # This improves model performance of ResNetV2 by 0.2~0.3% according to
    # https://arxiv.org/abs/1706.02677.
    self.bn_1 = self.normalization_layer(
        gamma_initializer=tf.keras.initializers.zeros(), epsilon=1e-5)

    if self.resnet_v2:
      self.act_1 = tf.keras.layers.ReLU()
    else:
      self.output_act = tf.keras.layers.ReLU()

    if self.use_projection:
      self.proj_conv = tf.keras.layers.Conv2D(
          filters=num_channels * width_multiplier * self.expansion_factor,
          kernel_size=1,
          strides=strides,
          use_bias=False,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          name='shortcut_conv')

      if not self.resnet_v2:
        self.proj_norm = self.normalization_layer(
            name='shortcut_batch_normalization', epsilon=1e-5)

  def compute_output_shape(self, input_shape):
    """Computes the output shape of this `BasicBlock`."""
    if self.use_projection:
      return self.proj_conv.compute_output_shape(input_shape)
    else:
      return input_shape


class BasicBlockV1(BasicBlock):
  """Version 1 of the basic residual block from [He et al.

  (2016)][1].

  In code, v1 refers to the ResNet defined in [1] but where a stride 2 is used
  on the 3x3 conv rather than the first 1x1 in the residual block.

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385
  """
  resnet_v2 = False
  expansion_factor = 1

  def call(self, inputs, training):

    x = shortcut = inputs
    for i, [pad, conv, norm, act] in enumerate((
        (self.padding_0, self.conv_0, self.bn_0, self.act_0),
        (self.padding_1, self.conv_1, self.bn_1, None),
    )):
      x = pad(x)
      x = conv(x)
      x = norm(x, training=training)
      x = act(x) if i == 0 else x

    if self.use_projection:
      shortcut = self.proj_conv(inputs)
      shortcut = self.proj_norm(shortcut, training=training)

    return self.output_act(x + shortcut)


class BasicBlockV2(BasicBlock):
  # pyformat: disable
  """Version 2 of the basic residual block from [He et al. (2016)][1].

  The principle difference from version 1 of the basic residual block
  from [He et al. (2016)][2] is that v1 applies batch normalization and
  activation after convolution, while v2 applies batch normalization, then the
  activation function, and finally convolution; a schematic comparison is
  presented in Figure 1 (left) of [1].

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity mappings in
      deep residual networks. In _Proceedings of the European Conference on
      Computer Vision_, 2016. https://arxiv.org/abs/1603.05027

  [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385
  """
  # pyformat: enable
  resnet_v2 = True
  expansion_factor = 1

  def call(self, inputs, training):

    x = shortcut = inputs
    for i, [pad, conv, norm, act] in enumerate((
        (self.padding_0, self.conv_0, self.bn_0, self.act_0),
        (self.padding_1, self.conv_1, self.bn_1, self.act_1),
    )):
      x = norm(x, training=training)
      x = act(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = pad(x)
      x = conv(x)

    return x + shortcut


class BottleneckBlock(tf.keras.Model):
  """Abstract bottleneck residual block that instantiates parameters.

  Version 1 [He et al. (2016)][1] and version 2 [He et al. (2016)][2] of the
  bottleneck residual block are implemented as subclasses of this abstract
  class.

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385

  [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity mappings in
      deep residual networks. In _Proceedings of the European Conference on
      Computer Vision_, 2016. https://arxiv.org/abs/1603.05027

  """
  normalization_layer = tf.keras.layers.BatchNormalization

  def __init__(self,
               num_channels,
               width_multiplier,
               strides,
               use_projection,
               kernel_initializer=tf.keras.initializers.he_normal(),
               kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
               name='bottleneck_residual_block'):
    """Constructs a `BottleneckBlock`.

    Args:
      num_channels: An integer specifying the base number of filters in each
        convolutional layer in this `BottleneckBlock`.
      width_multiplier: An integer specifying the factor by which to increase
        the number of filters in the 3x3 convolution in this `BottleneckBlock`.
      strides: An integer or an iterable of integers, specifying the stride(s)
        of the 3x3 convolution in this `BottleneckBlock`.
      use_projection: If True, the additive residual component is projected with
        a 1x1 convolution with stride `strides`; otherwise, it is simply the
        input.
      kernel_initializer: A subclass of `tf.keras.regularizer.Initializer`, used
        as a default initializer for kernel (weight) matrices.
      kernel_regularizer: A subclass of `tf.keras.regularizer.Regularizer`, used
        as a default regularizer for kernel (weight) matrices.
      name: The name for this `BottleneckBlock`.
    """
    super(BottleneckBlock, self).__init__(name=name)
    self.use_projection = use_projection

    self.conv_0 = tf.keras.layers.Conv2D(
        filters=num_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    self.bn_0 = self.normalization_layer(epsilon=1e-5)
    self.act_0 = tf.keras.layers.ReLU()

    self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
    self.conv_1 = tf.keras.layers.Conv2D(
        filters=num_channels * width_multiplier,
        kernel_size=3,
        strides=strides,
        use_bias=False,
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    self.bn_1 = self.normalization_layer(epsilon=1e-5)
    self.act_1 = tf.keras.layers.ReLU()

    self.conv_2 = tf.keras.layers.Conv2D(
        filters=num_channels * self.expansion_factor,
        kernel_size=1,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    # Zero-initialize the weights of the last batch-normalization layer in each
    # residual block, so that the (non-projected) residual branch starts with a
    # zero output and so each residual block starts as an identity mapping.
    # This improves model performance of ResNetV2 by 0.2~0.3% according to
    # https://arxiv.org/abs/1706.02677.
    self.bn_2 = self.normalization_layer(
        gamma_initializer=tf.keras.initializers.zeros(), epsilon=1e-5)

    if self.resnet_v2:
      self.act_2 = tf.keras.layers.ReLU()
    else:
      self.output_act = tf.keras.layers.ReLU()

    if self.use_projection:
      self.proj_conv = tf.keras.layers.Conv2D(
          filters=num_channels * self.expansion_factor,
          kernel_size=1,
          strides=strides,
          use_bias=False,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer)

      if not self.resnet_v2:
        self.proj_norm = self.normalization_layer(epsilon=1e-5)

  def compute_output_shape(self, input_shape):
    """Computes the output shape of this `BottleneckBlock`."""
    if self.use_projection:
      return self.proj_conv.compute_output_shape(input_shape)
    else:
      return input_shape


class BottleneckBlockV1(BottleneckBlock):
  """Version 1 of the bottleneck residual block from [He et al.

  (2016)][1].

  In code, v1 refers to the ResNet defined in [1] but where a stride 2 is used
  on the 3x3 conv rather than the first 1x1 in the bottleneck.

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385
  """
  expansion_factor = 4
  resnet_v2 = False

  def call(self, inputs, training):

    x = shortcut = inputs
    for i, [conv, norm, act] in enumerate((
        (self.conv_0, self.bn_0, self.act_0),
        (self.conv_1, self.bn_1, self.act_1),
        (self.conv_2, self.bn_2, None),
    )):
      if i == 1:
        x = self.padding_1(x)
      x = conv(x)
      x = norm(x, training=training)
      x = act(x) if i < 2 else x

    if self.use_projection:
      shortcut = self.proj_conv(inputs)
      shortcut = self.proj_norm(shortcut, training=training)

    return self.output_act(x + shortcut)


class BottleneckBlockV2(BottleneckBlock):
  """Version 2 of the bottleneck residual block from [He et al.

  (2016)][1].

  The principle difference from version 1 of the bottleneck residual block
  from [He et al. (2016)][2] is that v1 applies batch normalization and
  activation after convolution, while v2 applies batch normalization, then the
  activation function, and finally convolution; a schematic comparison is
  presented in Figure 1 (left) of [1].

  #### References

  [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity mappings in
      deep residual networks. In _Proceedings of the European Conference on
      Computer Vision_, 2016. https://arxiv.org/abs/1603.05027

  [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
      learning for image recognition. In _Proceedings of the IEEE Conference on
      Computer Vision and Pattern Recognition_, 2016.
      https://arxiv.org/abs/1512.03385
  """
  expansion_factor = 4
  resnet_v2 = True

  def call(self, inputs, training):

    x = shortcut = inputs
    for i, [conv, norm, act] in enumerate((
        (self.conv_0, self.bn_0, self.act_0),
        (self.conv_1, self.bn_1, self.act_1),
        (self.conv_2, self.bn_2, self.act_2),
    )):
      x = norm(x, training=training)
      x = act(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      if i == 1:
        x = self.padding_1(x)
      x = conv(x)

    return x + shortcut


@gin.configurable
class ResNet(ReparameterizableSpatialBackbone):
  """Abstract base class for residual network implementations.

  #### Examples

  The 18-layer residual network ([He et al., 2016][1]; [He et al., 2016][2])
  used in the meta-dataset few-shot classification experiments in [Triantafillou
  et al. (2020)][3] can be created as follows:

  ```python
  net = ResNet18V1()
  ```

  The 16-layer, 2-width wide residual network [(Zagoruyko & Komodakis)][4] used
  in the meta-dataset few-shot classification experiments in [Triantafillou et
  al. (2020)][3] can be created as follows:

  ```python
  net = WideResNet16V2(width_multiplier=2, large_input_kernel=False)
  ```

  #### References

  [1]: He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual
       learning for image recognition. In _Proceedings of the IEEE Conference on
       Computer Vision and Pattern Recognition_, 2016.
       https://arxiv.org/abs/1512.03385

  [2]: He, Kaiming, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity mappings in
       deep residual networks. In _Proceedings of the European Conference on
       Computer Vision_, 2016. https://arxiv.org/abs/1603.05027

  [3]: Triantafillou, Eleni, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku
       Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky,
       Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of
       datasets for learning to learn from few examples. In _Proceedings of 8th
       International Conference on Learning Representations_, 2020.
       https://arxiv.org/abs/1903.03096

  [4]: Zagoruyko, Sergey, and Nikos Komodakis. Wide residual networks. In
       _Proceedings of the British Machine Vision Conference_, 2016.
       https://arxiv.org/abs/1605.07146
  """
  width_multiplier = 1
  filter_multiplier = 4
  normalization_layer = tf.keras.layers.BatchNormalization

  def __init__(self,
               kernel_initializer=tf.keras.initializers.he_normal(),
               kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
               large_input_kernel=True,
               name='ResNet',
               **kwargs):
    """Creates a `ResNet`.

    Args:
      kernel_initializer: A subclass of `tf.keras.regularizer.Initializer`, used
        as a default initializer for kernel (weight) matrices; default is He
        initialization as per the original paper.
      kernel_regularizer: A subclass of `tf.keras.regularizer.Regularizer`, used
        as a default regularizer for kernel (weight) matrices; default is L2
        weight decay with parameter 1e-4 as per the original paper.
      large_input_kernel: If True, the input kernel will be of size 7 and will
        be applied at a stride of 2; if False, the input kernel will be of size
        3 and will be applied at a stride of 1. The output size of the first
        convolutional layer is the same in both cases.
      name: The name for this `ResNet`.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.
    """
    self.large_input_kernel = large_input_kernel
    super(ResNet, self).__init__(
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name,
        **kwargs)

  # TODO(eringrant): Implement DeepLab image alignment.
  # https://github.com/google-research/meta-dataset/blob/main/meta_dataset/models/functional_backbones.py#L363
  @property
  def stack(self):

    # WideResNet parameterization.
    base_num_filters = 16 * self.filter_multiplier

    if self.large_input_kernel:
      # Large kernel size as in the original ResNet paper; apply strided.
      input_kernel_size = 7
      input_kernel_stride = 2
    else:
      # Small kernel size as in the WideResNet paper; do not apply strided.
      input_kernel_size = 3
      input_kernel_stride = 1

    # Input convolution.
    input_stack = [
        # Use a zero padding layer prior to the convolution because the
        # `tf.keras.layers.Conv2D` layer does not allow explicit padding.
        tf.keras.layers.ZeroPadding2D(padding=(3, 3)),
        tf.keras.layers.Conv2D(
            filters=base_num_filters,
            kernel_size=input_kernel_size,
            strides=input_kernel_stride,
            use_bias=False,
            padding='valid',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        ),
    ]
    if not self.residual_block.resnet_v2:
      input_stack += [
          self.normalization_layer(epsilon=1e-5),
          tf.keras.layers.ReLU(),
      ]
    input_stack += [
        tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
    ]

    resnet_stack = []
    for i, num_blocks in enumerate(self.blocks_per_group):
      num_channels = base_num_filters * 2**i  # Double filters each block.
      for j in range(num_blocks):
        resnet_stack += [
            self.residual_block(
                num_channels=num_channels,
                width_multiplier=self.width_multiplier,
                # Stride 2 in the first block of later stages.
                strides=2 if i != 0 and j == 0 else 1,
                # Project in the first block of later stages that have stride 2,
                # and possibly the first stage if downsampling is required due
                # to the number of filters.
                use_projection=(j == 0 and
                                (i != 0 or base_num_filters !=
                                 (num_channels * self.width_multiplier *
                                  self.residual_block.expansion_factor))),
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='stage_%d/block_%d' % (i, j))
        ]

    output_stack = []
    if self.residual_block.resnet_v2:
      output_stack += [
          self.normalization_layer(epsilon=1e-5),
          tf.keras.layers.ReLU(),
      ]
    output_stack += [tf.keras.layers.GlobalAveragePooling2D()]

    return [
        tf.keras.Sequential(input_stack, name='input_stage'),
    ] + resnet_stack + [
        tf.keras.Sequential(output_stack, name='output_stage'),
    ]


class ResNet18(ResNet):
  """18-layer residual network backbone."""
  blocks_per_group = (2, 2, 2, 2)


@gin.configurable
class ResNet18V1(ResNet18):
  """18-layer residual network with v1 structure."""
  residual_block = BasicBlockV1


@gin.configurable
class ResNet18V2(ResNet18):
  """18-layer residual network with v2 structure."""
  residual_block = BasicBlockV2


class ResNet34(ResNet):
  """34-layer residual network backbone."""
  blocks_per_group = (3, 4, 6, 3)


@gin.configurable
class ResNet34V1(ResNet34):
  """34-layer residual network with v1 structure."""
  residual_block = BasicBlockV1


@gin.configurable
class ResNet34V2(ResNet34):
  """34-layer residual network with v2 structure."""
  residual_block = BasicBlockV2


class ResNet50(ResNet):
  """50-layer residual network backbone."""
  blocks_per_group = (3, 4, 6, 3)


@gin.configurable
class ResNet50V1(ResNet50):
  """50-layer residual network with v1 structure."""
  residual_block = BottleneckBlockV1


@gin.configurable
class ResNet50V2(ResNet50):
  """50-layer residual network with v2 structure."""
  residual_block = BottleneckBlockV2


class ResNet101(ResNet):
  """101-layer residual network backbone."""
  blocks_per_group = (3, 4, 23, 3)


@gin.configurable
class ResNet101V1(ResNet101):
  """101-layer residual network with v1 structure."""
  residual_block = BottleneckBlockV1


@gin.configurable
class ResNet101V2(ResNet101):
  """101-layer residual network with v2 structure."""
  residual_block = BottleneckBlockV2


class ResNet152(ResNet):
  """152-layer residual network backbone."""
  blocks_per_group = (3, 8, 36, 3)


@gin.configurable
class ResNet152V1(ResNet152):
  """152-layer residual network with v1 structure."""
  residual_block = BottleneckBlockV1


@gin.configurable
class ResNet152V2(ResNet152):
  """152-layer residual network with v2 structure."""
  residual_block = BottleneckBlockV2


def weight_standardization_replacements(model):
  """Weight-standardize non-output kernels of `model`."""
  if not isinstance(model, ReparameterizableBackbone):
    raise ValueError(
        '`model` must be an instance of `ReparameterizableBackbone`.')
  kernels = filter(lambda v: 'kernel' in v.name and 'output' not in v.name,
                   model.reparameterizables())
  replacements = []
  for v in kernels:
    # Wrap a standardization around the kernel.
    # Kernel has shape HWIO, normalize over HWI
    mean, var = tf.nn.moments(v, axes=[0, 1, 2], keepdims=True)
    # Author code uses std + 1e-5
    replacements.append((v.ref(), (v - mean) / tf.sqrt(var + 1e-10)))
  return dict(replacements)


class GNResNet(ResNet):
  normalization_layer = functools.partial(
      tfa.layers.GroupNormalization, groups=32)

  def call(self, inputs, **kwargs):
    with self.reparameterize(weight_standardization_replacements(self)):
      return super(GNResNet, self).call(inputs, **kwargs)


class GNBasicBlockV2(BasicBlockV2):
  normalization_layer = functools.partial(
      tfa.layers.GroupNormalization, groups=32)


@gin.configurable
class GNResNet18V2(GNResNet):
  blocks_per_group = (2, 2, 2, 2)
  residual_block = GNBasicBlockV2


class GNBottleneckBlockV2(BottleneckBlockV2):
  normalization_layer = functools.partial(
      tfa.layers.GroupNormalization, groups=32)


@gin.configurable
class GNResNet50V2(GNResNet):
  blocks_per_group = (3, 4, 6, 3)
  residual_block = GNBottleneckBlockV2


class WideResNet(ResNet):
  """Abstract base class for wide residual network implementations.

  The 16-layer, 2-wide residual network [(Zagoruyko & Komodakis)][4] used in the
  meta-dataset few-shot classification experiments in [Triantafillou et al.
  (2020)][3] can be created as follows:

  ```python
  net = WideResNet16V1(width_multiplier=2)
  ```

  #### References

  [1]: Zagoruyko, Sergey, and Nikos Komodakis. Wide residual networks. In
       _Proceedings of the British Machine Vision Conference_, 2016.
       https://arxiv.org/abs/1605.07146

  [2]: Triantafillou, Eleni, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku
       Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky,
       Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of
       datasets for learning to learn from few examples. In _Proceedings of 8th
       International Conference on Learning Representations_, 2020.
       https://arxiv.org/abs/1903.03096

  """
  filter_multiplier = 1

  def __init__(self, width_multiplier, name='WideResNet', **kwargs):
    """Creates a `WideResNet`.

    Args:
      width_multiplier: The integer "width" of the `WideResNet`; the factor by
        which the filter dimensionality in inner 3x3 convolutional blocks is
        multiplied.
      name: The name for this `WideResNet`.
      **kwargs: Python dictionary; remaining keyword arguments to be passed to
        the parent constructor.
    """
    self.width_multiplier = width_multiplier
    super(WideResNet, self).__init__(name=name, **kwargs)


class WideResNet16(WideResNet):
  """16-layer wide residual network with v1 structure."""
  blocks_per_group = (
      (16 - 4) // 6,
      (16 - 4) // 6,
      (16 - 4) // 6,
  )  # `WideResNet` depth is 6n+4, n blocks.


@gin.configurable
class WideResNet16V1(WideResNet16):
  """16-layer wide residual network with v1 structure."""
  residual_block = BasicBlockV1


@gin.configurable
class WideResNet16V2(WideResNet16):
  """16-layer wide residual network with v2 structure."""
  residual_block = BasicBlockV2


class WideResNet28(WideResNet):
  """28-layer wide residual network."""
  blocks_per_group = (
      (28 - 4) // 6,
      (28 - 4) // 6,
      (28 - 4) // 6,
  )  # `WideResNet` depth is 6n+4, n blocks.


@gin.configurable
class WideResNet28V1(WideResNet28):
  """28-layer wide residual network with v1 structure."""
  residual_block = BasicBlockV1


@gin.configurable
class WideResNet28V2(WideResNet28):
  """28-layer wide residual network with v2 structure."""
  residual_block = BasicBlockV2
