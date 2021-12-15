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
"""Tests for `meta_dataset.learners.baseline_learners`."""

import gin.tf

from meta_dataset.learners import base_test
from meta_dataset.learners import baseline_learners

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASELINE_ARGS = {
    **base_test.VALID_LEARNER_INIT_ARGS,
    'cosine_logits_multiplier': 1.0,
    'knn_in_fc': False,
    'knn_distance': 'l2',
}

gin.bind_parameter('linear_classifier.weight_decay', 0.01)


class BaselineTest(base_test.TestBatchLearner):
  learner_cls = baseline_learners.BaselineLearner
  learner_kwargs = {
      **BASELINE_ARGS,
      'cosine_classifier': False,
      'use_weight_norm': False,
  }


class WeightNormalizedBaselineTest(base_test.TestBatchLearner):
  learner_cls = baseline_learners.BaselineLearner
  learner_kwargs = {
      **BASELINE_ARGS,
      'cosine_classifier': False,
      'use_weight_norm': True,
  }


class CosineClassifierBaselineTest(base_test.TestBatchLearner):
  learner_cls = baseline_learners.BaselineLearner
  learner_kwargs = {
      **BASELINE_ARGS,
      'cosine_classifier': True,
      'use_weight_norm': False,
  }


class WeightNormalizedCosineClassifierBaselineTest(base_test.TestBatchLearner):
  learner_cls = baseline_learners.BaselineLearner
  learner_kwargs = {
      **BASELINE_ARGS,
      'cosine_classifier': True,
      'use_weight_norm': True,
  }


if __name__ == '__main__':
  tf.test.main()
