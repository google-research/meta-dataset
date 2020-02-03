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
from meta_dataset.data.dataset_spec import DatasetSpecification
import numpy as np
import tensorflow.compat.v1 as tf


def make_example(feat_floats, class_label, input_key, label_key):
  """Create an Example protocol buffer for the given image.

  Create a protocol buffer with an integer feature for the class label, and a
  bytes feature for the input (image or feature)

  Args:
    feat_floats: A list of floats.
    class_label: the integer class label of the image.
    input_key: String used as key for the input (feature).
    label_key: String used as key for the label.

  Returns:
    example_serial: A string correponding to the serialized example.

  """

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  feature = {
      input_key: _float32_feature(feat_floats),
      label_key: _int64_feature(class_label)
  }

  # Create an example protocol buffer.
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  example_serial = example.SerializeToString()
  return example_serial


class PipelineTest(tf.test.TestCase):

  def test_make_multisource_episode_pipeline_feature(self):

    def iterate_dataset(dataset, n):
      """Iterate over dataset."""
      if not tf.executing_eagerly():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with self.session(use_gpu=False) as sess:
          for idx in range(n):
            yield idx, sess.run(next_element)
      else:
        for idx, episode in enumerate(dataset):
          if idx == n:
            break
          yield idx, episode

    def write_feature_records(features, label, output_path):
      """Create a record file from features and labels.

      Args:
        features: An [n, m] numpy array of features.
        label: A numpy array containing the label.
        output_path: A string specifying the location of the record.
      """
      writer = tf.python_io.TFRecordWriter(output_path)
      for feat in list(features):
        # Write the example.
        serialized_example = make_example(
            feat.tolist(), label, input_key='embedding', label_key='label')
        writer.write(serialized_example)
      writer.close()

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
      write_feature_records(features, label, output_file)
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

    _, episode = next(iterate_dataset(dataset_episodic, 1))
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
