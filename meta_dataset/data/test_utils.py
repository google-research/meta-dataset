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
"""Utility functions for input pipeline tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
from meta_dataset.dataset_conversion import dataset_to_records
import numpy as np
import tensorflow.compat.v1 as tf

# DatasetSpecification to use in tests
DATASET_SPEC = DatasetSpecification(
    name=None,
    classes_per_split={
        Split.TRAIN: 15,
        Split.VALID: 5,
        Split.TEST: 10
    },
    images_per_class=dict(enumerate([10, 20, 30] * 10)),
    class_names=None,
    path=None,
    file_pattern='{}.tfrecords')

# Define defaults for the input pipeline.
MIN_WAYS = 5
MAX_WAYS_UPPER_BOUND = 50
MAX_NUM_QUERY = 10
MAX_SUPPORT_SET_SIZE = 500
MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
MIN_LOG_WEIGHT = np.log(0.5)
MAX_LOG_WEIGHT = np.log(2)


def set_episode_descr_config_defaults():
  """Sets default values for EpisodeDescriptionConfig using gin."""
  gin.parse_config('import meta_dataset.data.config')

  gin.bind_parameter('EpisodeDescriptionConfig.num_ways', None)
  gin.bind_parameter('EpisodeDescriptionConfig.num_support', None)
  gin.bind_parameter('EpisodeDescriptionConfig.num_query', None)
  gin.bind_parameter('EpisodeDescriptionConfig.min_ways', MIN_WAYS)
  gin.bind_parameter('EpisodeDescriptionConfig.max_ways_upper_bound',
                     MAX_WAYS_UPPER_BOUND)
  gin.bind_parameter('EpisodeDescriptionConfig.max_num_query', MAX_NUM_QUERY)
  gin.bind_parameter('EpisodeDescriptionConfig.max_support_set_size',
                     MAX_SUPPORT_SET_SIZE)
  gin.bind_parameter(
      'EpisodeDescriptionConfig.max_support_size_contrib_per_class',
      MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS)
  gin.bind_parameter('EpisodeDescriptionConfig.min_log_weight', MIN_LOG_WEIGHT)
  gin.bind_parameter('EpisodeDescriptionConfig.max_log_weight', MAX_LOG_WEIGHT)
  gin.bind_parameter('EpisodeDescriptionConfig.ignore_dag_ontology', False)
  gin.bind_parameter('EpisodeDescriptionConfig.ignore_bilevel_ontology', False)

  # Following is set in a different scope.
  gin.bind_parameter('none/EpisodeDescriptionConfig.min_ways', None)
  gin.bind_parameter('none/EpisodeDescriptionConfig.max_ways_upper_bound', None)
  gin.bind_parameter('none/EpisodeDescriptionConfig.max_num_query', None)
  gin.bind_parameter('none/EpisodeDescriptionConfig.max_support_set_size', None)
  gin.bind_parameter(
      'none/EpisodeDescriptionConfig.max_support_size_contrib_per_class', None)
  gin.bind_parameter('none/EpisodeDescriptionConfig.min_log_weight', None)
  gin.bind_parameter('none/EpisodeDescriptionConfig.max_log_weight', None)


def write_feature_records(features, label, output_path):
  """Creates a record file from features and labels.

  Args:
    features: An [n, m] numpy array of features.
    label: An integer, the label common to all records.
    output_path: A string specifying the location of the record.
  """
  writer = tf.python_io.TFRecordWriter(output_path)
  for feat in list(features):
    # Write the example.
    serialized_example = dataset_to_records.make_example([
        ('image/embedding', 'float32', feat.tolist()),
        ('image/class/label', 'int64', [label])
    ])
    writer.write(serialized_example)
  writer.close()
