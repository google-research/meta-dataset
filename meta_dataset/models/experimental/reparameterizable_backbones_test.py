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
"""Tests for `ReparameterizableBackbone`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
from meta_dataset.models.experimental import reparameterizable_backbones
from meta_dataset.models.experimental import reparameterizable_base_test
import numpy as np
import tensorflow.compat.v1 as tf

# Disabling torch import on 2021-06-30 as it failed to build then. -- lamblinp@
TORCH_AVAILABLE = False
# pylint: disable=g-import-not-at-top
if TORCH_AVAILABLE:
  import torch
  import torchvision
# pylint: enable=g-import-not-at-top

REPARAMETERIZABLE_MODULES = (
    reparameterizable_backbones.LinearModel,
    reparameterizable_backbones.FullyConnectedNet,
    reparameterizable_backbones.ConvNet,
    reparameterizable_backbones.RelationNetConvNet,
    reparameterizable_backbones.RelationModule,
    reparameterizable_backbones.ResNet18V1,
    reparameterizable_backbones.ResNet18V2,
    reparameterizable_backbones.ResNet34V1,
    reparameterizable_backbones.ResNet34V2,
    reparameterizable_backbones.ResNet50V1,
    reparameterizable_backbones.ResNet50V2,
    reparameterizable_backbones.ResNet101V1,
    reparameterizable_backbones.ResNet101V2,
    reparameterizable_backbones.ResNet152V1,
    reparameterizable_backbones.ResNet152V2,
    reparameterizable_backbones.WideResNet16V1,
    reparameterizable_backbones.WideResNet16V2,
    reparameterizable_backbones.WideResNet28V1,
    reparameterizable_backbones.WideResNet28V2,
)

IMAGE_INPUT_SHAPE = (1, 84, 84, 3)

VALID_MODULE_INIT_ARGS = {
    **reparameterizable_base_test.VALID_MODULE_INIT_ARGS,
    **{
        'output_dim': (1024,),
        'weight_decay': (0.001,),
        'keep_spatial_dims': (True, False),
        'width_multiplier': (2,),
        'num_filters_per_layer': ((64, 64, 64, 64),),
        'num_hiddens_per_layer': ((64, 64),),
    }
}

# TODO(eringrant): Finish integrating this.
VALID_MODULE_CALL_ARGS = {
    **reparameterizable_base_test.VALID_MODULE_CALL_ARGS,
    **{
        'input_shape': IMAGE_INPUT_SHAPE,
    }
}

TORCH_TENSORFLOW_PAIRS = ()
if TORCH_AVAILABLE:
  TORCH_TENSORFLOW_PAIRS = (
      ('resnet18', torchvision.models.resnet.resnet18,
       reparameterizable_backbones.ResNet18V1),
      ('resnet34', torchvision.models.resnet.resnet34,
       reparameterizable_backbones.ResNet34V1),
      ('resnet50', torchvision.models.resnet.resnet50,
       reparameterizable_backbones.ResNet50V1),
      ('resnet101', torchvision.models.resnet.resnet101,
       reparameterizable_backbones.ResNet101V1),
      ('resnet152', torchvision.models.resnet.resnet152,
       reparameterizable_backbones.ResNet152V1),
  )


def np_from_torch_tensor(t):
  return t.data.numpy()


def tf_initializer_from_numpy(t):
  return tf.compat.v1.constant_initializer(t)


def paired_named_parameter_generators(torch_model, tf_model):

  torch_parameter_names, torch_parameters = zip(*torch_model.named_parameters())
  tf_parameters = tf_model.trainable_variables
  tf_parameter_names = [t.name for t in tf_parameters]

  for (torch_parameter_name, tf_parameter_name, torch_parameter,
       tf_parameter) in itertools.zip_longest(torch_parameter_names,
                                              tf_parameter_names,
                                              torch_parameters, tf_parameters):

    if tf_parameter is None:
      # The PyTorch model includes the output layer, while the TF one does not.
      assert 'fc' in torch_parameter_name

    else:
      np_parameter = np_from_torch_tensor(torch_parameter)

      # PyTorch convolutions are channel-first.
      is_convolutional_kernel = (
          len(torch_parameter.shape) == 4 and
          ('conv' in torch_parameter_name or
           'downsample' in torch_parameter_name))
      if is_convolutional_kernel:
        np_parameter = np.transpose(np_parameter, [2, 3, 1, 0])

      yield (
          torch_parameter_name,
          tf_parameter_name,
          torch_parameter,
          tf_parameter,
          np_parameter,
      )


class ReparameterizableBackboneTest(tf.test.TestCase, parameterized.TestCase):
  """Tests whether backbones can be correctly reparameterized."""

  # TODO(eringrant): Write equivalent reparameterization tests as in
  # reparameterizable_base_test.py.
  pass


class BackboneTest(tf.test.TestCase, parameterized.TestCase):
  """Tests whether backbones can be built and run."""

  @parameterized.named_parameters(
      *reparameterizable_base_test.get_module_test_cases(
          REPARAMETERIZABLE_MODULES,
          reparameterizable_base_test.VARIABLE_REPARAMETERIZING_PREDICATES,
          reparameterizable_base_test.MODULE_REPARAMETERIZING_PREDICATES,
          VALID_MODULE_INIT_ARGS, VALID_MODULE_CALL_ARGS))
  def test_build_and_run_module(self, module_cls,
                                variable_reparameterizing_predicate,
                                module_reparameterizing_predicate,
                                module_init_kwargs, module_call_kwargs):
    # TODO(eringrant): Check reparameterization of these models.
    # TODO(eringrant): Integrate `module_call_kwargs`.
    del variable_reparameterizing_predicate
    del module_reparameterizing_predicate
    del module_call_kwargs

    try:
      model = module_cls(**module_init_kwargs)
    except ValueError:
      # Avoid tests that throw value errors to bypass invalid kwarg combos.
      pass
    else:
      model.build(IMAGE_INPUT_SHAPE)
      random_input = np.float32(np.random.normal(size=IMAGE_INPUT_SHAPE))

      with self.session(use_gpu=True):
        self.evaluate(tf.global_variables_initializer())
        model(tf.convert_to_tensor(random_input), training=True)


if TORCH_AVAILABLE:

  class ResNetTest(tf.test.TestCase, parameterized.TestCase):
    """Tests the implementation of ResNets against torch."""

    @parameterized.named_parameters(*TORCH_TENSORFLOW_PAIRS)
    def test_shapes_against_torch_model(self, torch_module_class,
                                        tf_module_class):
      torch_model = torch_module_class(pretrained=False)
      tf_model = tf_module_class(output_dim=None, keep_spatial_dims=False)
      tf_model.build(IMAGE_INPUT_SHAPE)

      for (torch_parameter_name, tf_parameter_name, _, tf_parameter,
           np_parameter) in paired_named_parameter_generators(
               torch_model, tf_model):

        parameter_names = 'lhs: {}; rhs: {}'.format(torch_parameter_name,
                                                    tf_parameter_name)
        # pylint: disable=g-assert-in-except
        try:
          self.assertShapeEqual(np_parameter, tf_parameter, msg=parameter_names)
        except TypeError:
          # If `tf_parameter` is a `ResourceVariable`, convert to a `Tensor`.
          tf_parameter = tf.convert_to_tensor(tf_parameter)
          self.assertShapeEqual(np_parameter, tf_parameter, msg=parameter_names)
        # pylint: enable=g-assert-in-except

    @parameterized.named_parameters(*TORCH_TENSORFLOW_PAIRS)
    def test_outputs_against_torch_model(self, torch_module_class,
                                         tf_module_class):
      torch_model = torch_module_class(pretrained=False)
      tf_model = tf_module_class(output_dim=1000, keep_spatial_dims=False)
      tf_model.build(IMAGE_INPUT_SHAPE)

      model_initialization_ops = []
      for (_, _, _, tf_parameter,
           np_parameter) in paired_named_parameter_generators(
               torch_model, tf_model):
        try:
          model_initialization_ops.append(tf.assign(tf_parameter, np_parameter))
        except ValueError:
          model_initialization_ops.append(
              tf.assign(tf_parameter, np.transpose(np_parameter, [1, 0])))
      model_initialization_ops = tf.group(model_initialization_ops)

      random_input = np.float32(np.random.normal(size=IMAGE_INPUT_SHAPE))

      torch_output = torch_model(
          torch.from_numpy(np.transpose(random_input, [0, 3, 1, 2])))

      with self.session(use_gpu=True):
        self.evaluate(tf.global_variables_initializer())
        self.evaluate(model_initialization_ops)

        for (_, _, _, tf_parameter,
             np_parameter) in paired_named_parameter_generators(
                 torch_model, tf_model):

          if len(np_parameter.shape) == 2:
            # TODO(eringrant): Integrate this hack for FC layers.
            np_parameter = np.transpose(np_parameter, [1, 0])

          self.assertAllEqual(np_parameter, self.evaluate(tf_parameter))

        tf_output = tf_model(tf.convert_to_tensor(random_input), training=True)

        # Error tolerance to account for compounded FPE in the forward pass.
        # For deeper models, we allow a higher absolute tolerance.
        if tf_module_class == reparameterizable_backbones.ResNet18V1:
          output_atol = 1e-5
        elif tf_module_class == reparameterizable_backbones.ResNet34V1:
          output_atol = 2e-5
        elif tf_module_class == reparameterizable_backbones.ResNet50V1:
          output_atol = 1e-3
        elif tf_module_class == reparameterizable_backbones.ResNet101V1:
          output_atol = 1e-3
        elif tf_module_class == reparameterizable_backbones.ResNet152V1:
          output_atol = 1e-2
        else:
          raise ValueError(
              'Unrecognized `tf_module_class`: {}'.format(tf_module_class))

        self.assertAllClose(
            torch_output.detach(), self.evaluate(tf_output), atol=output_atol)


if __name__ == '__main__':
  tf.test.main()
