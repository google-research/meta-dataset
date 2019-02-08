# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
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

"""Configuration classes for data processing.

Config classes that parametrize the behaviour of different stages of the data
processing pipeline, and are set up via `gin`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf


@gin.configurable
class DataConfig(object):
  """Common configuration options for creating data processing pipelines."""

  def __init__(self, image_height, shuffle_buffer_size, read_buffer_size_bytes):
    """Initialize a DataConfig.

    Args:
      image_height: An integer, the desired height for the images output by the
        data pipeline. Images are squared and have 3 channels (RGB), so each
        image will have shape [image_height, image_height, 3],
      shuffle_buffer_size: An integer, the size of the example buffer in the
        tf.data.Dataset.shuffle operations (there is typically one shuffle per
        class in the episodic setting, one per dataset in the batch setting).
        Classes with fewer examples as this number are shuffled in-memory.
      read_buffer_size_bytes: An integer, the size (in bytes) of the read buffer
        for each tf.data.TFRecordDataset (there is typically one for each class
        of each dataset).
    """
    self.image_height = image_height
    self.shuffle_buffer_size = shuffle_buffer_size
    self.read_buffer_size_bytes = read_buffer_size_bytes


class DataAugmentation(object):
  """Configurations for performing data augmentation."""

  def __init__(self, enable_jitter, jitter_amount, enable_gaussian_noise,
               gaussian_noise_std):
    """Initialize a DataAugmentation.

    Args:
      enable_jitter: bool whether to use image jitter (pad each image using
        reflection along x and y axes and then random crop).
      jitter_amount: amount (in pixels) to pad on all sides of the image.
      enable_gaussian_noise: bool whether to use additive Gaussian noise.
      gaussian_noise_std: Standard deviation of the Gaussian distribution.
    """
    self.enable_jitter = enable_jitter
    self.jitter_amount = jitter_amount
    self.enable_gaussian_noise = enable_gaussian_noise
    self.gaussian_noise_std = gaussian_noise_std


@gin.configurable
class SupportSetDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation on support set."""
  pass


@gin.configurable
class QuerySetDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation on query set."""
  pass


@gin.configurable
class BatchDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation for batch."""
  pass
