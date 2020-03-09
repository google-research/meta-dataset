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
r"""Tests for meta_dataset.data.pipeline.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gin
from meta_dataset.data import config
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import test_utils
from meta_dataset.data.dataset_spec import DatasetSpecification
import numpy as np
import tensorflow.compat.v1 as tf


class PipelineTest(tf.test.TestCase):

  def test_make_multisource_episode_pipeline_feature(self):

    # Create some feature records and write them to a temp directory.
    feat_size = 64
    num_examples = 100
    num_classes = 10
    output_path = self.get_temp_dir()
    gin.parse_config_file(
        'third_party/py/meta_dataset/learn/gin/setups/data_config_feature.gin')

    # 1-Write feature records to temp directory.
    self.rng = np.random.RandomState(0)
    class_features = []
    for class_id in range(num_classes):
      features = self.rng.randn(num_examples, feat_size).astype(np.float32)
      label = np.array(class_id).astype(np.int64)
      output_file = os.path.join(output_path, str(class_id) + '.tfrecords')
      test_utils.write_feature_records(features, label, output_file)
      class_features.append(features)
    class_features = np.stack(class_features)

    # 2-Read records back using multi-source pipeline.
    # DatasetSpecification to use in tests
    dataset_spec = DatasetSpecification(
        name=None,
        classes_per_split={
            learning_spec.Split.TRAIN: 5,
            learning_spec.Split.VALID: 2,
            learning_spec.Split.TEST: 3
        },
        images_per_class={i: num_examples for i in range(num_classes)},
        class_names=None,
        path=output_path,
        file_pattern='{}.tfrecords')

    # Duplicate the dataset to simulate reading from multiple datasets.
    use_bilevel_ontology_list = [False] * 2
    use_dag_ontology_list = [False] * 2
    all_dataset_specs = [dataset_spec] * 2

    fixed_ways_shots = config.EpisodeDescriptionConfig(
        num_query=5, num_support=5, num_ways=5)

    dataset_episodic = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=all_dataset_specs,
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        episode_descr_config=fixed_ways_shots,
        split=learning_spec.Split.TRAIN,
        image_size=None)

    episode, _ = self.evaluate(
        dataset_episodic.make_one_shot_iterator().get_next())

    # 3-Check that support and query features are in class_features and have
    # the correct corresponding label.
    support_features, support_class_ids = episode[0], episode[2]
    query_features, query_class_ids = episode[3], episode[5]

    for feat, class_id in zip(list(support_features), list(support_class_ids)):
      abs_err = np.abs(np.sum(class_features - feat[None][None], axis=-1))
      # Make sure the feature is present in the original data.
      self.assertEqual(abs_err.min(), 0.0)
      found_class_id = np.where(abs_err == 0.0)[0][0]
      self.assertEqual(found_class_id, class_id)

    for feat, class_id in zip(list(query_features), list(query_class_ids)):
      abs_err = np.abs(np.sum(class_features - feat[None][None], axis=-1))
      # Make sure the feature is present in the original data.
      self.assertEqual(abs_err.min(), 0.0)
      found_class_id = np.where(abs_err == 0.0)[0][0]
      self.assertEqual(found_class_id, class_id)


if __name__ == '__main__':
  tf.test.main()
