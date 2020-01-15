# coding=utf-8
# Copyright 2020 The Meta-Dataset Authors.
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
"""Interfaces for learning specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum


class Split(enum.Enum):
  """The possible data splits."""
  TRAIN = 0
  VALID = 1
  TEST = 2


class BatchSpecification(
    collections.namedtuple('BatchSpecification', 'split, batch_size')):
  """The specification of an episode.

    Args:
      split: the Split from which to pick data.
      batch_size: an int, the number of (image, label) pairs in the batch.
  """
  pass


class EpisodeSpecification(
    collections.namedtuple(
        'EpisodeSpecification',
        'split, num_classes, num_train_examples, num_test_examples')):
  """The specification of an episode.

    Args:
      split: A Split from which to pick data.
      num_classes: The number of classes in the episode, or None for variable.
      num_train_examples: The number of examples to use per class in the train
        phase, or None for variable.
      num_test_examples: the number of examples to use per class in the test
        phase, or None for variable.
  """
