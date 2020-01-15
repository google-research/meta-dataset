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
"""Interfaces for data returned by the pipelines.

TODO(lamblinp): Integrate better with pipeline.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class EpisodeDataset(
    collections.namedtuple(
        'EpisodeDataset', 'train_images, test_images, '
        'train_labels, test_labels, train_class_ids, test_class_ids')):
  """Wraps an episode's data and facilitates creation of feed dict.

    Args:
      train_images: A Tensor of images for training.
      test_images: A Tensor of images for testing.
      train_labels: A 1D Tensor, the matching training labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      test_labels: A 1D Tensor, the matching testing labels (numbers between 0
        and K-1, with K the number of classes involved in the episode).
      train_class_ids: A 1D Tensor, the matching training class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
      test_class_ids: A 1D Tensor, the matching testing class ids (numbers
        between 0 and N-1, with N the number of classes in the full dataset).
  """
  pass


class Batch(collections.namedtuple('Batch', 'images, labels')):
  """Wraps an batch's data and facilitates creation of feed dict.

    Args:
      images: a Tensor of images of shape [self.batch_size] + image shape.
      labels: a Tensor of labels of shape [self.batch_size].
  """
  pass
