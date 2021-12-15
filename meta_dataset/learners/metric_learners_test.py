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
"""Tests for `meta_dataset.learners.metric_learners`."""

import gin

from meta_dataset.learners import base_test
from meta_dataset.learners import metric_learners

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

metric_learner_kwargs = {
    **base_test.VALID_LEARNER_INIT_ARGS,
}

gin.bind_parameter('relationnet_convnet.weight_decay', 0.01)
gin.bind_parameter('relation_module.weight_decay', 0.01)


class CosineMatchingNetworkLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.MatchingNetworkLearner
  learner_kwargs = {
      **metric_learner_kwargs,
      'exact_cosine_distance': False,
  }


class ExactCosineMatchingNetworkLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.MatchingNetworkLearner
  learner_kwargs = {
      **metric_learner_kwargs,
      'exact_cosine_distance': True,
  }


class PrototypicalNetworkLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.PrototypicalNetworkLearner
  learner_kwargs = {
      **metric_learner_kwargs,
  }


class RelationNetworkLearnerTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.RelationNetworkLearner
  learner_kwargs = {
      **metric_learner_kwargs,
  }


if __name__ == '__main__':
  tf.test.main()
