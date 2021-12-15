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

"""Module responsible for decoding images as in SUR (Dvornik et al).

MIT License

Copyright (c) 2021 Nikita Dvornik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gin.tf
import tensorflow.compat.v1 as tf


@gin.configurable
class SURDataAugmentation(object):
  """Configurations for performing data augmentation."""

  def __init__(self, enable_jitter, jitter_amount, enable_gaussian_noise,
               gaussian_noise_std, enable_random_flip, enable_random_brightness,
               random_brightness_delta, enable_random_contrast,
               random_contrast_delta, enable_random_hue, random_hue_delta,
               enable_random_saturation, random_saturation_delta):
    """Initializes a DataAugmentation."""
    self.enable_jitter = enable_jitter
    self.jitter_amount = jitter_amount
    self.enable_gaussian_noise = enable_gaussian_noise
    self.gaussian_noise_std = gaussian_noise_std
    self.enable_random_flip = enable_random_flip
    self.enable_random_brightness = enable_random_brightness
    self.random_brightness_delta = random_brightness_delta
    self.enable_random_contrast = enable_random_contrast
    self.random_contrast_delta = random_contrast_delta
    self.enable_random_hue = enable_random_hue
    self.random_hue_delta = random_hue_delta
    self.enable_random_saturation = enable_random_saturation
    self.random_saturation_delta = random_saturation_delta


@gin.configurable
class SURImageDecoder(object):
  """Image decoder."""
  out_type = tf.float32

  def __init__(self, image_size=None, data_augmentation=None):
    """Class constructor.

    Args:
      image_size: int, desired image size. The extracted image will be resized
        to `[image_size, image_size]`.
      data_augmentation: A DataAugmentation object with parameters for
        perturbing the images.
    """

    self.image_size = image_size
    self.data_augmentation = data_augmentation

  def __call__(self, example_string):
    """Processes a single example string.

    Extracts and processes the image, and ignores the label. We assume that the
    image has three channels.
    Args:
      example_string: str, an Example protocol buffer.

    Returns:
      image_rescaled: the image, resized to `image_size x image_size` and
      rescaled to [-1, 1]. Note that Gaussian data augmentation may cause values
      to go beyond this range.
    """
    image_string = tf.parse_single_example(
        example_string,
        features={
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })['image']
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image_resized = tf.image.resize_images(
        image_decoded, [self.image_size, self.image_size],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)
    image = tf.cast(image_resized, tf.float32)

    if self.data_augmentation is not None:
      if self.data_augmentation.enable_random_brightness:
        delta = self.data_augmentation.random_brightness_delta
        image = tf.image.random_brightness(image, delta)

      if self.data_augmentation.enable_random_saturation:
        delta = self.data_augmentation.random_saturation_delta
        image = tf.image.random_saturation(image, 1 - delta, 1 + delta)

      if self.data_augmentation.enable_random_contrast:
        delta = self.data_augmentation.random_contrast_delta
        image = tf.image.random_contrast(image, 1 - delta, 1 + delta)

      if self.data_augmentation.enable_random_hue:
        delta = self.data_augmentation.random_hue_delta
        image = tf.image.random_hue(image, delta)

      if self.data_augmentation.enable_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = 2 * (image / 255.0 - 0.5)  # Rescale to [-1, 1].

    if self.data_augmentation is not None:
      if self.data_augmentation.enable_gaussian_noise:
        image = image + tf.random_normal(
            tf.shape(image)) * self.data_augmentation.gaussian_noise_std

      if self.data_augmentation.enable_jitter:
        j = self.data_augmentation.jitter_amount
        paddings = tf.constant([[j, j], [j, j], [0, 0]])
        image = tf.pad(image, paddings, 'REFLECT')
        image = tf.image.random_crop(image,
                                     [self.image_size, self.image_size, 3])

    return image
