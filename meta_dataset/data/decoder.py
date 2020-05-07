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
"""Module responsible for decoding image/feature examples."""
import gin.tf
import tensorflow.compat.v1 as tf


def read_single_example(example_string):
  """Parses the record string."""
  return tf.parse_single_example(
      example_string,
      features={
          'image': tf.FixedLenFeature([], dtype=tf.string),
          'label': tf.FixedLenFeature([], tf.int64)
      })


def read_example_and_parse_image(example_string):
  """Reads the string and decodes the image."""
  parsed_example = read_single_example(example_string)
  image_decoded = tf.image.decode_image(parsed_example['image'], channels=3)
  image_decoded.set_shape([None, None, 3])
  parsed_example['image'] = image_decoded
  return parsed_example


@gin.configurable
class ImageDecoder(object):
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
    return self.decode_with_label(example_string)[0]

  def decode_with_label(self, example_string):
    """Processes a single example string.

    Extracts and processes the image, and ignores the label. We assume that the
    image has three channels.

    Args:
      example_string: str, an Example protocol buffer.

    Returns:
      image_rescaled: the image, resized to `image_size x image_size` and
        rescaled to [-1, 1]. Note that Gaussian data augmentation may cause
        values to go beyond this range.
      label: tf.int
    """
    ex_decoded = read_example_and_parse_image(example_string)
    image_decoded = ex_decoded['image']
    image_resized = tf.image.resize_images(
        image_decoded, [self.image_size, self.image_size],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)
    image_resized = tf.cast(image_resized, tf.float32)
    image = 2 * (image_resized / 255.0 - 0.5)  # Rescale to [-1, 1].

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
    return image, tf.cast(ex_decoded['label'], dtype=tf.int32)


@gin.configurable
class FeatureDecoder(object):
  """Feature decoder."""
  out_type = tf.float32

  def __init__(self, feat_len):
    """Class constructor.

    Args:
      feat_len: The expected length of the feature vectors.
    """

    self.feat_len = feat_len

  def __call__(self, example_string):
    """Processes a single example string.

    Extracts and processes the feature, and ignores the label.

    Args:
      example_string: str, an Example protocol buffer.

    Returns:
      feat: The feature tensor.
    """
    feat = tf.parse_single_example(
        example_string,
        features={
            'image/embedding':
                tf.FixedLenFeature([self.feat_len], dtype=tf.float32),
            'image/class/label':
                tf.FixedLenFeature([], tf.int64)
        })['image/embedding']

    return feat


@gin.configurable
class StringDecoder(object):
  """Simple decoder that reads the image without decoding."""
  out_type = tf.string

  def __init__(self):
    """Class constructor."""

  def __call__(self, example_string):
    """Processes a single example string.

    Extracts the image as string, and ignores the label.

    Args:
      example_string: str, an Example protocol buffer.

    Returns:
      img_string: tf.Tensor of type tf.string.
    """
    img_string = read_single_example(example_string)['image']
    return img_string
