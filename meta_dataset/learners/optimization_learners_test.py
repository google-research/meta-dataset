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
"""Tests for `meta_dataset.learners.optimization_learners`."""

import gin.tf

from meta_dataset.learners import base_test
from meta_dataset.learners import optimization_learners

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASELINE_FINETUNE_ARGS = {
    **base_test.VALID_LEARNER_INIT_ARGS,
    'knn_in_fc': False,
    'knn_distance': False,
    'cosine_classifier': False,
    'cosine_logits_multiplier': 1.0,
    'use_weight_norm': False,
    'is_training': False,
}

gin.bind_parameter('MAMLLearner.classifier_weight_decay', 0.01)


class BaselineFinetuneTest():

  def testLearnerConvergence(self):
    # `BaselineFinetuneLearner` differs from `BaselineLearner` only at
    # evaluation time.
    pass

  def testLearnerImprovement(self):
    # `BaselineFinetuneLearner` differs from `BaselineLearner` only at
    # evaluation time.
    pass


class BaselineFinetuneAllLayersAdamTest(BaselineFinetuneTest,
                                        base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.BaselineFinetuneLearner
  learner_kwargs = {
      **BASELINE_FINETUNE_ARGS,
      'num_finetune_steps': 5,
      'finetune_lr': 0.01,
      'finetune_all_layers': True,
      'finetune_with_adam': True,
  }


class BaselineFinetuneAllLayersGDTest(BaselineFinetuneTest,
                                      base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.BaselineFinetuneLearner
  learner_kwargs = {
      **BASELINE_FINETUNE_ARGS,
      'num_finetune_steps': 5,
      'finetune_lr': 0.1,
      'finetune_all_layers': True,
      'finetune_with_adam': False,
  }


class BaselineFinetuneLastLayerGDTest(BaselineFinetuneTest,
                                      base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.BaselineFinetuneLearner
  learner_kwargs = {
      **BASELINE_FINETUNE_ARGS,
      'num_finetune_steps': 10,
      'finetune_lr': 0.1,
      'finetune_all_layers': False,
      'finetune_with_adam': False,
  }


MAML_KWARGS = {
    **base_test.VALID_LEARNER_INIT_ARGS,
    'additional_evaluation_update_steps':
        0,
    'first_order':
        True,
    'adapt_batch_norm':
        True,
}


class MAMLLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.MAMLLearner
  learner_kwargs = {
      **MAML_KWARGS,
      'num_update_steps': 5,
      'alpha': 0.01,
      'zero_fc_layer': False,
      'proto_maml_fc_layer_init': False,
  }


class ZeroInitMAMLLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.MAMLLearner
  learner_kwargs = {
      **MAML_KWARGS,
      'num_update_steps': 10,
      'alpha': 0.01,
      'zero_fc_layer': True,
      'proto_maml_fc_layer_init': False,
  }


class ProtoMAMLLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = optimization_learners.MAMLLearner
  learner_kwargs = {
      **MAML_KWARGS,
      'num_update_steps': 5,
      'alpha': 0.01,
      'zero_fc_layer': False,
      'proto_maml_fc_layer_init': True,
  }


if __name__ == '__main__':
  tf.test.main()
