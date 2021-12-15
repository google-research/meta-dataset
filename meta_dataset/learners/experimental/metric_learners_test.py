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

from meta_dataset.learners import base_test
from meta_dataset.learners.experimental import metric_learners
from meta_dataset.models.experimental import reparameterizable_backbones
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

metric_learner_kwargs = {
    'backprop_through_moments': True,
    'transductive_batch_norm': True,
    'input_shape': [84, 84, 3],
    'logit_dim': 5,
    'is_training': True,
    'distance_metric': metric_learners.euclidean_distance,
}


class PrototypicalNetworkTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.PrototypicalNetwork
  learner_kwargs = metric_learner_kwargs


class MatchingNetworkTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.MatchingNetwork
  learner_kwargs = metric_learner_kwargs


class RelationNetworkTest(base_test.TestEpisodicLearner):
  learner_cls = metric_learners.RelationNetwork
  learner_kwargs = metric_learner_kwargs

  def set_up_learner(self):
    """Set up a `reparameterizable_backbones.RelationNetConvNet` backbone."""
    learner_kwargs = self.learner_kwargs
    learner_kwargs['embedding_fn'] = (
        reparameterizable_backbones.RelationNetConvNet(keep_spatial_dims=True))
    data = self.random_data()
    learner = self.learner_cls(**learner_kwargs)
    return data, learner


if __name__ == '__main__':
  tf.test.main()
