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

# Lint as: python2, python3
"""Module containing (meta-)Learners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meta_dataset.learners.base import BatchLearner
from meta_dataset.learners.base import EpisodicLearner
from meta_dataset.learners.base import Learner
from meta_dataset.learners.baseline_learners import BaselineLearner
from meta_dataset.learners.metric_learners import MatchingNetworkLearner
from meta_dataset.learners.metric_learners import MetricLearner
from meta_dataset.learners.metric_learners import PrototypicalNetworkLearner
from meta_dataset.learners.metric_learners import RelationNetworkLearner
from meta_dataset.learners.optimization_learners import BaselineFinetuneLearner
from meta_dataset.learners.optimization_learners import FLUTEFiLMLearner
from meta_dataset.learners.optimization_learners import MAMLLearner
from meta_dataset.learners.optimization_learners import OptimizationLearner

__all__ = [
    'BaselineFinetuneLearner',
    'BatchLearner',
    'BaselineLearner',
    'EpisodicLearner',
    'Learner',
    'MAMLLearner',
    'MatchingNetworkLearner',
    'MetricLearner',
    'OptimizationLearner',
    'PrototypicalNetworkLearner',
    'RelationNetworkLearner',
    'FLUTEFiLMLearner',
]
