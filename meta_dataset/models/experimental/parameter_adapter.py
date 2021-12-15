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
"""Utilities for adapting checkpoints for use with `ReparameterizableModule`s.

Parameter adapters are functions that take a single non-configurable `tf.Module`
argument, and initialize the variables of that `tf.Module` in a particular way;
see `bit_parameter_adapter` as an example. These functions can be passed
to the constructor of `reparameterizable_backbones.ReparameterizableBackbone`s
to allow greater control over how their variables are initialized.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import re

import gin.tf
from meta_dataset.models import functional_backbones
from meta_dataset.models.experimental import reparameterizable_backbones
import tensorflow as tf
import tensorflow_hub as hub


def assign_variables(variable_mapping):
  """Assign variables according to the provided `variable_mapping`.

  Args:
    variable_mapping: An iterable of variable pairs, each corresponding to a
      variable whose value is to be overwitten (destination) and a reference
      variable (source).

  Returns:
    If running in TensorFlow Eager mode, returns None; otherwise, returns a list
    of assignment operations.
  """
  for variable, reference_variable in variable_mapping:
    if tf.executing_eagerly():
      # Just perform the assignment.
      variable.assign(reference_variable)
    else:
      # Piggyback on the variable's initializer attribute, which is included in
      # `tf.global_variables_initializer`.
      initializer_ops = [variable._initializer_op]  # pylint: disable=protected-access
      if isinstance(reference_variable, tf.Variable):
        initializer_ops += [reference_variable._initializer_op]  # pylint: disable=protected-access
      with tf.control_dependencies(initializer_ops):
        assign_op = variable.assign(reference_variable)
      variable._initializer_op = assign_op  # pylint: disable=protected-access


def ckpt_parameter_adapter(model, ckpt_path, renamed_variable_generator):
  """Return a mapping of variable names in `ckpt_path` to variables in `model`.

  #### Examples

  ```python
  net = ResNet18V1(output_dim=None)
  net.build((None, 84, 84, 3))
  var_mapping = parameter_adapter(
      model=net,
      ckpt_path='CKPT_PATH/model_27500.ckpt',
      renamed_variable_generator=meta_dataset_variable_generator)
  ```

  The returned dictionary is suitable for instantiating a `tf.train.Saver`
  instance that restores the corresponding variables of `model` to values from
  `ckpt_path`:

  ```python
  ckpt_to_restore = 'CKPT_PATH/model_46000.ckpt',
  net = WideResNet16V2(
      width_multiplier=2, large_input_kernel=False, output_dim=None)
  net.build((None, 84, 84, 3))
  var_mapping = parameter_adapter(
      net,
      ckpt_to_restore,
      renamed_variable_generator=meta_dataset_variable_generator)

  saver = tf.train.Saver(var_mapping)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, ckpt_to_restore)
  ```

  Args:
    model: A subclass of `tf.keras.Model`; the model for which to load variable
      values from `ckpt_path`.
    ckpt_path: An absolute path to the checkpoint from which to load variables.
    renamed_variable_generator: A callable that takes in a subclass of
      `tf.Module` and and a list of variable names and yields elements of a
      one-to-one mapping of elements of the list to `tf.Variable`s of the
      `tf.Module`.

  Returns:
    A dictionary that maps variable names from the checkpoint at `ckpt_path` to
    variables in `model`, where the mapping is determined by
    `renamed_variable_generator`.
  """
  if not model.built:
    raise ValueError(
        '`model` must have been built before parameters can be mapped.')

  # Read the checkpoint.
  reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)

  var_to_shape_map = reader.get_variable_to_shape_map()
  ckpt_vars = var_to_shape_map.keys()
  ckpt_vars = list(filter(functional_backbones.is_backbone_variable, ckpt_vars))

  replacement_mapping = []
  for new_tensor, old_tensor_name in renamed_variable_generator(
      model, ckpt_vars):

    if not reader.has_tensor(old_tensor_name):
      raise ValueError('Tensor %s not found in checkpoint.' % old_tensor_name)

    old_tensor = reader.get_tensor(old_tensor_name)

    old_tensor_shape = tuple(var_to_shape_map[old_tensor_name])
    new_tensor_shape = tuple(new_tensor.shape.as_list())
    if old_tensor_shape != new_tensor_shape:
      old_tensor = old_tensor.reshape(new_tensor.shape)

    replacement_mapping += [(new_tensor, old_tensor)]

  return replacement_mapping


def single_to_double_digits(s):
  return re.sub(r'\d+', lambda m: '{:02d}'.format(int(m.group())), s)


def filter_and_sort_vars(list_of_vars, string):
  try:
    return sorted(
        filter(lambda x: string in x, list_of_vars),
        key=single_to_double_digits)
  except TypeError:
    # Try accessing the `name` attribute instead (e.g., of a `tf.Variable`).
    return sorted(
        filter(lambda x: string in x.name, list_of_vars),
        key=lambda x: single_to_double_digits(x.name))


def meta_dataset_functional_variable_generator(reparameterizable_model,
                                               functional_ckpt_vars):
  """Generate pairs of reparameterizable and functional variables."""
  new_vars = reparameterizable_model.trainable_variables

  # The previous meta-dataset implementation uses a projection layer in all
  # shortcut connections (corresponding to (C) in Table 3 of
  # https://arxiv.org/abs/1512.03385); cf. the TorchVision implementation
  # (https://github.com/pytorch/vision/blob/e61538cba036c42bab23ce8f9d205da9889977ae/torchvision/models/resnet.py#L183),
  # which uses a projection shortcut only if the dimensionality of the input and
  # output of the residual block are not the same (corresponding to (B) in Table
  # 3 of https://arxiv.org/abs/1512.03385).
  # (C) has slightly lower error rate but an increased parameter count over (B).
  # We adapt the old implementation using (C) to (B) by simply removing the
  # parameters of the shortcut projection layers unused by (B).
  ckpt_projection_vars = filter_and_sort_vars(functional_ckpt_vars,
                                              'projection')
  new_projection_vars = filter_and_sort_vars(new_vars, 'shortcut')
  functional_ckpt_vars = [
      x for x in functional_ckpt_vars if x not in ckpt_projection_vars
  ]
  new_vars = [x for x in new_vars if x not in new_projection_vars]

  if len(ckpt_projection_vars) != len(new_projection_vars):
    # pylint: disable=g-complex-comprehension
    ckpt_projection_vars = [
        x for x in ckpt_projection_vars if
        # TODO(eringrant): Remove this hack that is specific
        # to the WideResNet16V2 architecture from
        # meta-dataset-V2.
        ('wide_resnet' in x and 'block_0' in x) or
        # TODO(eringrant): Remove this hack that is specific
        # to the ResNet18V1 architecture from
        # meta-dataset-V2.
        ('resnet18' in x and 'conv2_x' not in x)
    ]
    # pylint: enable=g-complex-comprehension

  ckpt_output_layer_vars = filter_and_sort_vars(functional_ckpt_vars,
                                                'embedding')
  new_output_layer_vars = filter_and_sort_vars(new_vars, 'output_stage')
  functional_ckpt_vars = [
      x for x in functional_ckpt_vars if x not in ckpt_output_layer_vars
  ]
  new_vars = [x for x in new_vars if x not in new_output_layer_vars]

  ckpt_input_layer_vars = (
      list(filter_and_sort_vars(functional_ckpt_vars, 'resnet18/conv1')) +
      list(filter_and_sort_vars(functional_ckpt_vars, 'resnet/conv1')))
  new_input_layer_vars = filter_and_sort_vars(new_vars, 'input_stage')
  functional_ckpt_vars = [
      x for x in functional_ckpt_vars if x not in ckpt_input_layer_vars
  ]
  new_vars = [x for x in new_vars if x not in new_input_layer_vars]

  ckpt_conv_vars = filter_and_sort_vars(functional_ckpt_vars, '/conv')
  ckpt_conv_kernel_vars = filter_and_sort_vars(ckpt_conv_vars, 'weight')
  ckpt_conv_bias_vars = filter_and_sort_vars(ckpt_conv_vars, 'bias')

  new_conv_vars = filter_and_sort_vars(new_vars, '/conv')
  new_conv_kernel_vars = filter_and_sort_vars(new_conv_vars, 'kernel')
  new_conv_bias_vars = filter_and_sort_vars(new_conv_vars, 'bias')

  ckpt_bn_vars = filter_and_sort_vars(functional_ckpt_vars, 'batch_norm')
  ckpt_bn_scale_vars = filter_and_sort_vars(ckpt_bn_vars, 'scale')
  ckpt_bn_offset_vars = filter_and_sort_vars(ckpt_bn_vars, 'offset')

  new_bn_vars = filter_and_sort_vars(new_vars, 'batch_normalization')
  new_bn_scale_vars = filter_and_sort_vars(new_bn_vars, 'gamma')
  new_bn_offset_vars = filter_and_sort_vars(new_bn_vars, 'beta')

  return itertools.chain(
      zip(new_conv_kernel_vars, ckpt_conv_kernel_vars),
      zip(new_conv_bias_vars, ckpt_conv_bias_vars),
      zip(new_bn_scale_vars, ckpt_bn_scale_vars),
      zip(new_bn_offset_vars, ckpt_bn_offset_vars),
      zip(new_projection_vars, ckpt_projection_vars),
      zip(new_input_layer_vars, ckpt_input_layer_vars),
      zip(new_output_layer_vars, ckpt_output_layer_vars),
  )


@gin.configurable(denylist=['model'])
def meta_dataset_functional_parameter_adapter(model, ckpt_path):
  variable_mapping = ckpt_parameter_adapter(
      model=model,
      ckpt_path=ckpt_path,
      renamed_variable_generator=meta_dataset_functional_variable_generator)
  assign_variables(variable_mapping)


def bit_variable_generator(model, imported_model):
  """Pair variables for reparameterizable and BiT backbones.

  Args:
    model: A valid `reparameterizable_backbones.ReparameterizableBackbone`.
    imported_model: A `tf.AutoTrackable`.

  Returns:
    Pairs of corresponding `tf.Variable`s from `model` and `imported_model`.

  Raises:
    ValueError: If `model` is not a valid model.
  """
  valid_models = (
      reparameterizable_backbones.ResNet18V2,
      reparameterizable_backbones.GNResNet18V2,
      reparameterizable_backbones.ResNet50V2,
      reparameterizable_backbones.GNResNet50V2,
  )
  if all(not isinstance(model, valid_model) for valid_model in valid_models):
    raise ValueError(
        'BiT variable mapping defined only for models among: {}'.format(
            valid_models))

  pairs = []
  name_to_variable = dict((v.name, v) for v in model.variables)
  imported_name_to_variable = dict(
      (v.name, v) for v in imported_model.variables)

  group_norm = isinstance(model, reparameterizable_backbones.GNResNet)

  # Assign root layer variables.
  # TODO(eringrant): Determine why 'input_stage' is not prepended to variables
  # in tensorflow Eager mode.
  input_stage_prefix = 'input_stage/' if not tf.executing_eagerly() else ''
  output_stage_prefix = 'output_stage/' if not tf.executing_eagerly() else ''

  name = '{}conv2d/kernel:0'.format(input_stage_prefix)
  imported_name = 'resnet/root_block/{}conv2d/kernel:0'.format(
      'standardized_' if group_norm else '')

  pairs += [(name_to_variable[name], imported_name_to_variable[imported_name])]

  # Assign normalization variables.
  for n in ('gamma', 'beta') + (() if group_norm else
                                ('moving_mean', 'moving_variance')):
    name = '{}{}_normalization_{}/{}:0'.format(
        output_stage_prefix, 'group' if group_norm else 'batch', 16 if
        (isinstance(model, reparameterizable_backbones.ResNet18V2) or
         isinstance(model, reparameterizable_backbones.GNResNet18V2)) else 48,
        n)
    imported_name = 'resnet/{}/{}:0'.format(
        'group_norm' if group_norm else 'batch_normalization', n)
    pairs += [(name_to_variable[name], imported_name_to_variable[imported_name])
             ]

  # Assign other layer variables.
  if (isinstance(model, reparameterizable_backbones.ResNet18V2) or
      isinstance(model, reparameterizable_backbones.GNResNet18V2)):
    for i, (stage, block, layer) in enumerate(
        itertools.product(range(4), range(2), range(2))):

      imported_base_path = 'resnet/block{}/unit0{}/{}/'.format(
          stage + 1, block + 1, {
              0: 'a',
              1: 'b'
          }[layer])

      for n in ('gamma', 'beta') + (() if group_norm else
                                    ('moving_mean', 'moving_variance')):
        name = 'stage_{}/block_{}/{}_normalization{}/{}:0'.format(
            stage, block, 'group' if group_norm else 'batch',
            '_{}'.format(i) if i > 0 else '', n)
        imported_name = (
            imported_base_path + '{}/{}:0'.format(
                'group_norm' if group_norm else 'batch_normalization', n))
        pairs += [(name_to_variable[name],
                   imported_name_to_variable[imported_name])]

      if stage > 0 and block == layer == 0:
        name = 'stage_{}/block_{}/shortcut_conv/kernel:0'.format(stage, block)
        imported_name = (
            imported_base_path + 'proj/{}conv2d/kernel:0'.format(
                'standardized_' if group_norm else ''))
        pairs += [(name_to_variable[name],
                   imported_name_to_variable[imported_name])]

      name = 'stage_{}/block_{}/conv2d_{}/kernel:0'.format(stage, block, i + 1)
      imported_name = (
          imported_base_path +
          '{}conv2d/kernel:0'.format('standardized_' if group_norm else ''))
      pairs += [(name_to_variable[name],
                 imported_name_to_variable[imported_name])]

  elif (isinstance(model, reparameterizable_backbones.ResNet50V2) or
        isinstance(model, reparameterizable_backbones.GNResNet50V2)):
    conv_id = 1
    batch_norm_id = 0
    for stage, num_blocks in enumerate((3, 4, 6, 3)):
      for block in range(num_blocks):
        for layer in range(3):
          imported_base_path = 'resnet/block{}/unit0{}/{}/'.format(
              stage + 1, block + 1, {
                  0: 'a',
                  1: 'b',
                  2: 'c'
              }[layer])

          for n in ('gamma', 'beta') + (() if group_norm else
                                        ('moving_mean', 'moving_variance')):
            name = 'stage_{}/block_{}/{}_normalization{}/{}:0'.format(
                stage, block, 'group' if group_norm else 'batch',
                '_{}'.format(batch_norm_id) if batch_norm_id > 0 else '', n)
            imported_name = (
                imported_base_path + '{}/{}:0'.format(
                    'group_norm' if group_norm else 'batch_normalization', n))
            pairs += [(name_to_variable[name],
                       imported_name_to_variable[imported_name])]
          batch_norm_id += 1

          name = 'stage_{}/block_{}/conv2d_{}/kernel:0'.format(
              stage, block, conv_id)
          imported_name = (
              imported_base_path +
              '{}conv2d/kernel:0'.format('standardized_' if group_norm else ''))
          pairs += [(name_to_variable[name],
                     imported_name_to_variable[imported_name])]
          conv_id += 1

        if block == 0:
          name = 'stage_{}/block_{}/conv2d{}/kernel:0'.format(
              stage, block, '_{}'.format(conv_id) if conv_id > 0 else '')
          imported_name = 'resnet/block{}/unit01/a/proj/{}conv2d/kernel:0'.format(
              stage + 1, 'standardized_' if group_norm else '')
          pairs += [(name_to_variable[name],
                     imported_name_to_variable[imported_name])]
          conv_id += 1

  else:
    raise NotImplementedError('BiT variable mappings are not defined for '
                              'models other than {}.'.format(valid_models))

  return pairs


@gin.configurable(denylist=['model'])
def bit_parameter_adapter(model, ckpt_path, tags=None):
  """Initialize `model`s parameters from the BiT backbone at `ckpt_path`."""
  imported_model = hub.load(handle=ckpt_path, tags=tags)
  variable_mapping = bit_variable_generator(
      model=model, imported_model=imported_model)
  assign_variables(variable_mapping)
