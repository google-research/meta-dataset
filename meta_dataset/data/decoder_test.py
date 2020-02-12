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

# Lint as: python3
"""Tests for meta_dataset.data.decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meta_dataset.data import decoder
from meta_dataset.dataset_conversion import dataset_to_records
import numpy as np
import tensorflow.compat.v1 as tf


class DecoderTest(tf.test.TestCase):

  def test_image_decoder(self):
    # Make random image.
    image_size = 84
    image = np.random.randint(
        low=0, high=255, size=[image_size, image_size, 3]).astype(np.ubyte)

    # Encode
    image_bytes = dataset_to_records.encode_image(image, image_format='PNG')
    label = np.zeros(1).astype(np.int64)
    image_example = dataset_to_records.make_example([
        ('image', 'bytes', [image_bytes]), ('label', 'int64', [label])
    ])

    # Decode
    image_decoder = decoder.ImageDecoder(image_size=image_size)
    image_decoded = image_decoder(image_example)
    # Assert perfect reconstruction.
    with self.session(use_gpu=False) as sess:
      image_rec_numpy = sess.run(image_decoded)
    self.assertAllClose(2 * (image.astype(np.float32) / 255.0 - 0.5),
                        image_rec_numpy)

  def test_feature_decoder(self):
    # Make random feature.
    feat_size = 64
    feat = np.random.randn(feat_size).astype(np.float32)
    label = np.zeros(1).astype(np.int64)

    # Encode
    feat_example = dataset_to_records.make_example([
        ('image/embedding', 'float32', feat),
        ('image/class/label', 'int64', [label]),
    ])

    # Decode
    feat_decoder = decoder.FeatureDecoder(feat_len=feat_size)
    feat_decoded = feat_decoder(feat_example)

    # Assert perfect reconstruction.
    with self.session(use_gpu=False) as sess:
      feat_rec_numpy = sess.run(feat_decoded)
    self.assertAllEqual(feat_rec_numpy, feat)


if __name__ == '__main__':
  tf.test.main()
