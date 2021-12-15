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

"""Test utility functions and constants."""

import numpy as np
import tensorflow as tf

_SOURCES = {
    'aircraft': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (70, 15, 15)
    },
    'cu_birds': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (140, 30, 30)
    },
    'dtd': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (33, 7, 7)
    },
    'fungi': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (994, 200, 200)
    },
    'ilsvrc_2012': {
        'same_across_md_versions': False,
        'meta_splits': {'v1': ('train', 'valid', 'test'), 'v2': ('train',)},
        'num_classes': {'v1': (712, 158, 130), 'v2': (1000,)},
        'remap_labels': {'v1': False, 'v2': True},
        'md_source': {'v1': 'ilsvrc_2012', 'v2': 'ilsvrc_2012_v2'},
    },
    'mscoco': {
        'same_across_md_versions': True,
        'meta_splits': ('valid', 'test'),
        'num_classes': (40, 40)
    },
    'omniglot': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (883, 81, 659)
    },
    'quickdraw': {
        'same_across_md_versions': True,
        'meta_splits': ('train', 'valid', 'test'),
        'num_classes': (241, 52, 52)
    },
    'traffic_sign': {
        'same_across_md_versions': True,
        'meta_splits': ('test',), 'num_classes': (42,)
    },
    'vgg_flower': {
        'same_across_md_versions': False,
        'meta_splits': {'v1': ('train', 'valid', 'test')},
        'num_classes': {'v1': (71, 15, 16)},
    },
}


def make_class_dataset_comparison_test_cases():
  """Returns class dataset test cases for Meta-Dataset."""
  testcases = []
  for source, info in _SOURCES.items():
    meta_splits = ({'v1': info['meta_splits'], 'v2': info['meta_splits']}
                   if info['same_across_md_versions']
                   else info['meta_splits'])
    num_classes = ({'v1': info['num_classes'], 'v2': info['num_classes']}
                   if info['same_across_md_versions']
                   else info['num_classes'])

    for md_version in meta_splits:
      offsets = np.cumsum((0,) + num_classes[md_version][:-1])
      remap_labels = (info['remap_labels'][md_version] if 'remap_labels' in info
                      else False)
      md_source = (info['md_source'][md_version] if 'md_source' in info
                   else source)
      for meta_split, num_labels, offset in zip(
          meta_splits[md_version], num_classes[md_version], offsets):
        testcases.append((f'{source}_{md_version}_{meta_split}', source,
                          md_source, md_version, meta_split, num_labels, offset,
                          remap_labels))

  return testcases


def parse_example(example_string):
  return tf.io.parse_single_example(
      example_string,
      features={'image': tf.io.FixedLenFeature([], dtype=tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)})
