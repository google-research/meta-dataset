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
"""Tests for `meta_dataset.models.experimental.reparameterizable_base`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
from meta_dataset import test_utils
from meta_dataset.models.experimental import reparameterizable_base
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)


class ReparameterizableDense(reparameterizable_base.ReparameterizableModule,
                             tf.keras.layers.Dense):
  pass


class ReparameterizableConv2D(reparameterizable_base.ReparameterizableModule,
                              tf.keras.layers.Conv2D):
  pass


class ReparameterizableBatchNormalization(
    reparameterizable_base.ReparameterizableModule,
    tf.keras.layers.BatchNormalization):

  pass


REPARAMETERIZABLE_MODULES = (
    ReparameterizableDense,
    ReparameterizableConv2D,
    ReparameterizableBatchNormalization,
)

VARIABLE_REPARAMETERIZING_PREDICATES = (
    # TODO(eringrant): Add `reparameterizable_base.is_variable` as an option
    # here once the behavior of `ReparameterizableBatchNormalization` is
    # smoothed out; currently, the layer attempts to update the reparameterized
    # moving mean and moving variances, which causes the tests to fail.
    reparameterizable_base.is_trainable_variable,)
MODULE_REPARAMETERIZING_PREDICATES = (
    reparameterizable_base.is_batch_norm_module,)

VALID_MODULE_INIT_ARGS = {
    'units': (12,),
    'activation': (tf.nn.relu, tf.nn.tanh, None),
    'kernel_initializer': ('ones',),
    'use_bias': (True, False),
    'num_filters': (32,),
    'filters': (32,),
    'kernel_size': (32,),
}

VALID_MODULE_CALL_ARGS = {}

VALID_MODULE_INPUT_SHAPE = {
    ReparameterizableDense: (1, 32),
    ReparameterizableConv2D: (1, 32, 32, 3),
    ReparameterizableBatchNormalization: (1, 32),
}

# Transformations with constant Jacobian (e.g., linear transforms) should
# produce the same output gradient when variables are replaced.
# Note that transforms in this list may take an `activation` parameter
# that when set to a non-linearity such as `tf.nn.relu`, induces a non-constant
# Jacobian.
CONSTANT_JACOBIAN_TRANSFORMS = (
    ReparameterizableDense,
    ReparameterizableConv2D,
    ReparameterizableBatchNormalization,
)


def has_constant_jacobian(layer):
  return (any(isinstance(layer, cls) for cls in CONSTANT_JACOBIAN_TRANSFORMS)
          and (not hasattr(layer, 'activation') or layer.activation is None or
               'linear' in str(layer.activation)))


def _get_module_predicate_test_cases(module_cls, variable_predicate,
                                     module_predicate, valid_module_init_args,
                                     valid_module_call_args):
  """Return parameters of tests for `module_cls` with the given arguments."""
  test_cases = []
  for valid_init_kwargs, valid_call_kwargs in test_utils.get_valid_kwargs(
      module_cls, valid_module_init_args, valid_module_call_args):
    test_cases += [
        dict((
            ('testcase_name', '{}_{}_{}_{}_{}'.format(
                str(module_cls),
                str(variable_predicate),
                str(module_predicate),
                str(valid_init_kwargs),
                str(valid_call_kwargs),
            )),
            ('module_cls', module_cls),
            ('variable_reparameterizing_predicate', variable_predicate),
            ('module_reparameterizing_predicate', module_predicate),
            ('module_init_kwargs', valid_init_kwargs),
            ('module_call_kwargs', valid_call_kwargs),
        ))
    ]
  return test_cases


def get_module_test_cases(reparameterizable_modules=None,
                          module_reparameterizing_predicates=None,
                          variable_reparameterizing_predicates=None,
                          valid_module_init_args=None,
                          valid_module_call_args=None):
  """Return test parameters for `reparameterizable_modules` and predicates."""

  if reparameterizable_modules is None:
    reparameterizable_modules = REPARAMETERIZABLE_MODULES
  if variable_reparameterizing_predicates is None:
    variable_reparameterizing_predicates = VARIABLE_REPARAMETERIZING_PREDICATES
  if module_reparameterizing_predicates is None:
    module_reparameterizing_predicates = MODULE_REPARAMETERIZING_PREDICATES
  if valid_module_init_args is None:
    valid_module_init_args = VALID_MODULE_INIT_ARGS
  if valid_module_call_args is None:
    valid_module_call_args = VALID_MODULE_CALL_ARGS

  test_cases = []
  for variable_predicate, module_predicate in itertools.product(
      variable_reparameterizing_predicates,
      (*module_reparameterizing_predicates, None)):
    if variable_predicate is None and module_predicate is None:
      continue
    for module_cls in reparameterizable_modules:
      test_cases += _get_module_predicate_test_cases(module_cls,
                                                     variable_predicate,
                                                     module_predicate,
                                                     valid_module_init_args,
                                                     valid_module_call_args)
  return test_cases


def _randomized_variables(variables):
  # `variables` may contain duplicates due to the way `tf.Module._flatten`
  # works (in particular, because a `tf.Variable` may be referenced by more
  # than one attribute of a `tf.Module`. We ensure that only as many variables
  # are generated as there are unique elements in `variables` by iterating over
  # `set(variables)`.
  variable_set = set(v.ref() for v in variables)

  def rv(size):
    return np.random.normal(scale=.01, size=size).astype(np.float32)

  randomized = dict((v_ref, tf.Variable(rv(size=v_ref.deref().shape.as_list())))
                    for v_ref in variable_set)
  return tuple(randomized[v.ref()] for v in variables)


def get_params_and_replacements(module, variable_predicate, module_predicate):
  paths, variables = zip(*module.reparameterizables(
      variable_predicate=variable_predicate,
      module_predicate=module_predicate,
      with_path=True))
  replacement_variables = _randomized_variables(variables)
  return paths, variables, replacement_variables


def _init_module(module_cls, module_init_kwargs):
  """Initialize and build a `module_cls` instance with `module_init_kwargs`."""
  module = module_cls(**module_init_kwargs)
  if hasattr(module, 'built'):  # for e.g., Keras modules
    module.build(VALID_MODULE_INPUT_SHAPE[module_cls])
  return module


def _init_reference_module(module_cls, module_init_kwargs, paths, variables):
  """Create a mock `module_cls` instance with `variables` as attributes."""
  reference_module = _init_module(module_cls, module_init_kwargs)

  # Manually set attributes of this module via `getattr` and `setattr`.
  for path, variable in zip(paths, variables):
    descoped_module = reparameterizable_base.chained_getattr(
        reference_module, path[:-1])
    reparameterizable_base.corner_case_setattr(descoped_module, path[-1],
                                               variable)

  return reference_module


def _setup_modules(module_cls, variable_reparameterizing_predicate,
                   module_reparameterizing_predicate, module_init_kwargs):
  """Return `module_cls` instances for reparameterization and for reference."""

  # Module to be tested.
  module_to_reparameterize = _init_module(module_cls, module_init_kwargs)

  # Replacement parameters.
  paths, variables, replacement_variables = get_params_and_replacements(
      module_to_reparameterize,
      variable_reparameterizing_predicate,
      module_reparameterizing_predicate,
  )

  # Reference modules.
  before_reference_module = _init_reference_module(module_cls,
                                                   module_init_kwargs, paths,
                                                   variables)
  after_reference_module = _init_reference_module(module_cls,
                                                  module_init_kwargs, paths,
                                                  replacement_variables)

  return (
      module_to_reparameterize,
      before_reference_module,
      after_reference_module,
      variables,
      replacement_variables,
  )


# TODO(eringrant): Implement the following tests:
#   - test that the correct Tensors are being accessed via `reparameterizables`.
#   - add tests to include exact verification of results in simple
#     cases (e.g., computing gradients with numpy for linear regression).
#   - add gradient correctness check via finite differences.
class TestReparameterizableModule(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*get_module_test_cases())
  def test_swap_and_revert_parameters(
      self,
      module_cls,
      variable_reparameterizing_predicate,
      module_reparameterizing_predicate,
      module_init_kwargs,
      module_call_kwargs,
  ):

    try:
      (module_to_reparameterize, before_reference_module,
       after_reference_module, variables,
       replacement_variables) = _setup_modules(
           module_cls, variable_reparameterizing_predicate,
           module_reparameterizing_predicate, module_init_kwargs)
    except ValueError:
      # TODO(eringrant): Assert that no variables are returned only in expected
      # cases.
      return

    # Random inputs.
    input_shape = VALID_MODULE_INPUT_SHAPE[module_cls]
    inputs = tf.cast(np.random.normal(size=input_shape), tf.float32)

    # Reference outputs.
    reference_before_result = before_reference_module(inputs)
    reference_after_result = after_reference_module(inputs)

    # The output should conform before variable replacement.
    before_reparameterization_result = module_to_reparameterize(inputs)

    replacement_map = dict(
        (v1.ref(), v2) for v1, v2 in zip(variables, replacement_variables))
    with module_to_reparameterize.reparameterize(replacement_map):
      # The output should conform after variable replacement.
      after_reparameterization_result = module_to_reparameterize(inputs)

    # The output should conform after variable reversion.
    after_reversion_result = module_to_reparameterize(inputs)

    with self.session(use_gpu=True) as sess:

      self.evaluate(tf.global_variables_initializer())

      # For reference.
      reference_before_value = sess.run(reference_before_result)
      reference_after_value = sess.run(reference_after_result)

      # For testing.
      (
          before_reparameterization_value,
          after_reparameterization_value,
          after_reversion_value,
      ) = sess.run([
          before_reparameterization_result,
          after_reparameterization_result,
          after_reversion_result,
      ])

      # The outputs should differ (by a transformation defined by the module).
      # Note this does not check that the transformation is correct.
      self.assertNotAllClose(reference_before_value, reference_after_value)

      # The output should conform before variable replacement.
      self.assertAllEqual(before_reparameterization_value,
                          reference_before_value)

      # The output should conform after variable replacement.
      self.assertAllEqual(after_reparameterization_value, reference_after_value)

      # The output should conform after variable reversion.
      self.assertAllEqual(after_reversion_value, reference_before_value)

  @parameterized.named_parameters(*get_module_test_cases())
  def test_weight_gradients_after_swap_and_revert(
      self, module_cls, variable_reparameterizing_predicate,
      module_reparameterizing_predicate, module_init_kwargs,
      module_call_kwargs):

    try:
      (module_to_reparameterize, before_reference_module,
       after_reference_module, variables,
       replacement_variables) = _setup_modules(
           module_cls, variable_reparameterizing_predicate,
           module_reparameterizing_predicate, module_init_kwargs)
    except ValueError:
      # TODO(eringrant): Assert that no variables are returned only in expected
      # cases.
      return

    # Random inputs.
    input_shape = VALID_MODULE_INPUT_SHAPE[module_cls]
    inputs = tf.cast(np.random.normal(size=input_shape), tf.float32)

    # Reference outputs.
    reference_before_result = tf.gradients(
        before_reference_module(inputs), variables)
    reference_after_result = tf.gradients(
        after_reference_module(inputs), replacement_variables)

    replacement_map = dict(
        (v1.ref(), v2) for v1, v2 in zip(variables, replacement_variables))

    results = {
        'before': {},
        'after': {},
        'none': {},
    }

    before_outputs = module_to_reparameterize(inputs)
    results['before']['before_replacement_outside_context'] = tf.gradients(
        before_outputs, variables)
    results['none']['before_replacement_outside_context'] = tf.gradients(
        before_outputs, replacement_variables)

    with module_to_reparameterize.reparameterize(replacement_map):
      results['before']['before_replacement_inside_context'] = tf.gradients(
          before_outputs, variables)
      results['none']['before_replacement_inside_context'] = tf.gradients(
          before_outputs, replacement_variables)

      during_outputs = module_to_reparameterize(inputs)
      results['none']['during_replacement_inside_context'] = tf.gradients(
          during_outputs, variables)
      results['after']['during_replacement_inside_context'] = tf.gradients(
          during_outputs, replacement_variables)

    results['before']['before_replacement_outside_context2'] = tf.gradients(
        before_outputs, variables)
    results['none']['before_replacement_outside_context2'] = tf.gradients(
        before_outputs, replacement_variables)

    results['none']['during_replacement_outside_context'] = tf.gradients(
        during_outputs, variables)
    results['after']['during_replacement_outside_context'] = tf.gradients(
        during_outputs, replacement_variables)

    after_outputs = module_to_reparameterize(inputs)
    results['before']['after_replacement_outside_context'] = tf.gradients(
        after_outputs, variables)
    results['none']['after_replacement_outside_context'] = tf.gradients(
        after_outputs, replacement_variables)

    for context in results['none']:
      for x in results['none'][context]:
        self.assertIsNone(x)
    del results['none']  # Don't try to fetch Nones.

    with self.session(use_gpu=True) as sess:

      self.evaluate(tf.global_variables_initializer())

      # For reference.
      before_value, after_value, values = sess.run(
          (reference_before_result, reference_after_result, results))

      if has_constant_jacobian(module_to_reparameterize):
        # The gradients should be the same.
        self.assertAllClose(before_value, after_value)
      else:
        # The gradients should differ.
        self.assertNotAllClose(before_value, after_value)

      # The gradients should conform before variable replacement.
      for grads in values['before']:
        for grad, grad_ref in zip(results['before'][grads],
                                  reference_before_result):
          self.assertAllClose(grad, grad_ref, rtol=1e-05)

      # The gradients should conform after variable replacement.
      for grads in values['after']:
        for grad, grad_ref in zip(results['after'][grads],
                                  reference_after_result):
          self.assertAllClose(grad, grad_ref, rtol=1e-05)

  @parameterized.named_parameters(*get_module_test_cases())
  def test_input_gradients_after_swap_and_revert(
      self, module_cls, variable_reparameterizing_predicate,
      module_reparameterizing_predicate, module_init_kwargs,
      module_call_kwargs):

    try:
      (module_to_reparameterize, before_reference_module,
       after_reference_module, variables,
       replacement_variables) = _setup_modules(
           module_cls, variable_reparameterizing_predicate,
           module_reparameterizing_predicate, module_init_kwargs)
    except ValueError:
      # TODO(eringrant): Assert that no variables are returned only in expected
      # cases.
      return

    # Random inputs.
    input_shape = VALID_MODULE_INPUT_SHAPE[module_cls]
    inputs = tf.cast(np.random.normal(size=input_shape), tf.float32)

    # Reference outputs.
    reference_before_result = tf.gradients(
        before_reference_module(inputs), inputs)
    reference_after_result = tf.gradients(
        after_reference_module(inputs), inputs)

    replacement_map = dict(
        (v1.ref(), v2) for v1, v2 in zip(variables, replacement_variables))

    results = {
        'before': {},
        'after': {},
    }

    before_outputs = module_to_reparameterize(inputs)
    results['before']['before_replacement_outside_context'] = tf.gradients(
        before_outputs, inputs)

    with module_to_reparameterize.reparameterize(replacement_map):
      results['before']['before_replacement_inside_context'] = tf.gradients(
          before_outputs, inputs)

      during_outputs = module_to_reparameterize(inputs)
      results['after']['during_replacement_inside_context'] = tf.gradients(
          during_outputs, inputs)

    results['before']['before_replacement_outside_context2'] = tf.gradients(
        before_outputs, inputs)
    results['after']['during_replacement_outside_context'] = tf.gradients(
        during_outputs, inputs)

    after_outputs = module_to_reparameterize(inputs)
    results['before']['after_replacement_outside_context'] = tf.gradients(
        after_outputs, inputs)

    with self.session(use_gpu=True) as sess:

      self.evaluate(tf.global_variables_initializer())

      # For reference.
      before_value, after_value, values = sess.run(
          (reference_before_result, reference_after_result, results))

      # The input gradients should differ because the weights have changed.
      self.assertNotAllClose(before_value, after_value)

      # The gradients should conform before variable replacement.
      for grads in values['before']:
        for grad, grad_ref in zip(results['before'][grads],
                                  reference_before_result):
          self.assertAllClose(grad, grad_ref, rtol=1e-06)

      # The gradients should conform after variable replacement.
      for grads in values['after']:
        for grad, grad_ref in zip(results['after'][grads],
                                  reference_after_result):
          self.assertAllClose(grad, grad_ref, rtol=1e-06)


if __name__ == '__main__':
  tf.test.main()
