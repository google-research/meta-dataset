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
"""Tests for meta_dataset.learners.experimental.metric_learners."""

import gin.tf
from meta_dataset.learners import base_test
from meta_dataset.learners.experimental import optimization_learners
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def mock_sgd():

  def init(x0):
    return x0

  def update(i, grad, state):
    del i
    x = state
    return x - 0.01 * grad

  def get_params(state):
    x = state
    return x

  return init, update, get_params


optimization_learner_kwargs = {
    'backprop_through_moments': True,
    'input_shape': [84, 84, 3],
    'logit_dim': 5,
    'is_training': True,
    'update_fn': mock_sgd,
    'additional_evaluation_update_steps': 5,
    'clip_grad_norm': 10.0,
    'num_update_steps': 5,
}


class FirstOrderMAMLTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.MAML
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': False,
          'proto_maml_fc_layer_init': False,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': True,
      })


class VanillaMAMLTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.MAML
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': True,
          'proto_maml_fc_layer_init': False,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': False,
      })


class ProtoMAMLTest(base_test.TestEpisodicLearner):

  def setUp(self):
    super().setUp()
    gin.bind_parameter('proto_maml_fc_layer_init_fn.prototype_multiplier', 1.0)

  def tearDown(self):
    gin.clear_config()
    super().tearDown()

  learner_cls = optimization_learners.MAML
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': False,
          'proto_maml_fc_layer_init': True,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': True,
      })


class FirstOrderANILTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.ANIL
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': False,
          'proto_maml_fc_layer_init': False,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': True,
      })


class VanillaANILTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.ANIL
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': True,
          'proto_maml_fc_layer_init': False,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': False,
      })


class ProtoANILTest(base_test.TestEpisodicLearner):

  def setUp(self):
    super().setUp()
    gin.bind_parameter('proto_maml_fc_layer_init_fn.prototype_multiplier', 1.0)

  def tearDown(self):
    gin.clear_config()
    super().tearDown()

  learner_cls = optimization_learners.ANIL
  learner_kwargs = dict(
      **optimization_learner_kwargs, **{
          'transductive_batch_norm': False,
          'proto_maml_fc_layer_init': True,
          'zero_fc_layer_init': False,
          'first_order': False,
          'adapt_batch_norm': True,
      })


if __name__ == '__main__':
  tf.test.main()
