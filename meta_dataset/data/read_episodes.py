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

# Lint as: python3
"""Demonstrates how to read dumped episodes from disk.

Dumped episodes are assumed to be stored as `{episode_number}-train.tfrecords`
and `{episode_number}-test.tfrecords` file pairs containing the support and
query set, respectively.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import json
import math
import os

from meta_dataset.data import decoder
from meta_dataset.data import utils
from task_adaptation import data_loader

import tensorflow.compat.v1 as tf

VTAB_NATURAL = [
    "caltech101",
    "cifar(num_classes=100)",
    "dtd",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "sun397",
    "svhn"
]
VTAB_SPECIALIZED = [
    "diabetic_retinopathy(config='btgraham-300')",
    "eurosat",
    "resisc45",
    "patch_camelyon",
]
VTAB_STRUCTURED = [
    "clevr(task='closest_object_distance')",
    "clevr(task='count_all')",
    "dmlab",
    "dsprites(predicted_attribute='label_orientation', num_classes=16)",
    "dsprites(predicted_attribute='label_x_position', num_classes=16)",
    "smallnorb(predicted_attribute='label_azimuth')",
    "smallnorb(predicted_attribute='label_elevation')",
    "kitti(task='closest_vehicle_distance')",
]
VTAB_DATASETS = VTAB_NATURAL + VTAB_SPECIALIZED + VTAB_STRUCTURED


def read_episode_as_dataset(episodes_dir,
                            episode_index,
                            split,
                            with_info=False):
  """This function reads a single episode from the directory.

  Args:
    episodes_dir: str, directory that has tf_record files.
    episode_index: int, of the episode to be loaded.
    split: str, `test` or `train`
    with_info: bool, if True reads the json file in the folder and returns
      number of

  Returns:
    decoded_dataset: tf.data.Dataset, with `image` and `label` fields.
    num_images_per_class_dict: tf.data.Dataset, if with_info=True.

  Raises:
    ValueError, when `split` is not one of {'train', 'test'}.
  """
  episode_path = utils.get_file_path(episodes_dir, episode_index, split)
  raw_dataset = tf.data.TFRecordDataset(episode_path)
  decoded_dataset = raw_dataset.map(decoder.read_example_and_parse_image)
  if with_info:
    info_path = utils.get_info_path(episodes_dir)
    with tf.io.gfile.GFile(info_path, "r") as f:
      all_info = json.load(f)
    # Convert keys to integer.
    key = os.path.basename(episode_path)
    num_images_per_class_dict = {int(k): v for k, v in all_info[key].items()}
    return decoded_dataset, num_images_per_class_dict

  return decoded_dataset


def read_episodes_from_records(episodes_dir,
                               train_suffix=utils.TRAIN_SUFFIX,
                               test_suffix=utils.TEST_SUFFIX):
  """This function reads all episodes from a given directory.

  Additionally it returns total number of episodes available in one epoch.
  Args:
    episodes_dir: str, directory that has tf_record files.
    train_suffix: str, used during dumping of episodes to indicate training
      records. Default value should be kept unless the default is overwritten
      during creation of these episodes.
    test_suffix: str, used during dumping of episodes to indicate test records.
      Default value should be kept unless the default is overwritten during
      creation of these episodes.

  Returns:
    tf.data.Dataset, that returns a tuple of training and test datasets. Each
      dataset has `image` and `label` fields.
    int, number of episodes read.
  Raises:
    RuntimeError: when some episodes are missing.
  """
  all_files = sorted(tf.io.gfile.listdir(episodes_dir))
  train_files = [f for f in all_files if f.endswith(train_suffix)]
  test_files = [f for f in all_files if f.endswith(test_suffix)]
  for test_file, train_file in zip(test_files, train_files):
    # Check whether ids match: expected format episode-0001-train.tfrecords
    # TODO(evcu) maybe use regex.
    if test_file.split("-")[1] != train_file.split("-")[1]:
      test_id = int(test_file.split("-")[1])
      train_id = int(train_file.split("-")[1])
      if test_id < train_id:
        raise RuntimeError("Train data missing for %d th episode." % test_id)
      else:
        raise RuntimeError("Test data missing for %d th episode." % train_id)

  # Load episode information to obtain total number of images for each episode.
  info_path = utils.get_info_path(episodes_dir)
  with tf.io.gfile.GFile(info_path, "r") as f:
    all_info = json.load(f)

  def get_total_img_count(file_name):
    return sum(all_info[file_name].values())

  # Define function to load individual tf-record files.
  def _load_and_batch(file_path, n_images):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.batch(tf.cast(n_images, tf.int64))
    batch_data = dataset.make_initializable_iterator().get_next()
    # This pipeline doesn't have original label ids.
    return batch_data

  train_paths = tf.data.Dataset.from_tensor_slices(
      [os.path.join(episodes_dir, f) for f in train_files])
  train_n_images = tf.data.Dataset.from_tensor_slices(
      [get_total_img_count(f) for f in train_files])
  test_paths = tf.data.Dataset.from_tensor_slices(
      [os.path.join(episodes_dir, f) for f in test_files])
  test_n_images = tf.data.Dataset.from_tensor_slices(
      [get_total_img_count(f) for f in test_files])
  dataset_of_datasets = tf.data.Dataset.zip(
      (tf.data.Dataset.zip((train_paths, train_n_images)).map(_load_and_batch),
       tf.data.Dataset.zip((test_paths, test_n_images)).map(_load_and_batch)))
  return dataset_of_datasets, len(test_files)


def read_episodes_from_records_multiple_sources(episodes_dir_list,
                                                train_suffix=utils.TRAIN_SUFFIX,
                                                test_suffix=utils.TEST_SUFFIX):
  """This function reads episodes from a list of episode directories.

  Additionally it returns list of total number of episodes available in per
  directory in one epoch.

  Args:
    episodes_dir_list: list, list of directories that have tf_record files.
    train_suffix: str, used during dumping of episodes to indicate training
      records. Default value should be kept unless the default is overwritten
      during creation of these episodes.
    test_suffix: str, used during dumping of episodes to indicate test records.
      Default value should be kept unless the default is overwritten during
      creation of these episodes.

  Returns:
    tf.data.Dataset, that returns a tuple of training and test datasets. These
    tuples are randomly chosen from one of the directories in episode_dir_list
    with uniform probability. Each dataset has `image` and `label` fields.
    list(int), number of episodes read in each directory.
  Raises:
    RuntimeError: when some episodes are missing.
  """

  num_files_per_dir = []
  dataset_per_dir = []
  for episode_dir in episodes_dir_list:
    episode_dataset, episode_num_files = read_episodes_from_records(
        episode_dir, train_suffix, test_suffix)
    num_files_per_dir.append(episode_num_files)
    dataset_per_dir.append(episode_dataset)
  return tf.data.experimental.sample_from_datasets(
      dataset_per_dir), num_files_per_dir


def read_vtab_as_episode(vtab_key,
                         image_size=224,
                         query_size_limit=500,
                         data_dir=None):
  """This function reads VTAB-1k datasets as episodes.

  The training set becomes support set and the test is the query set.
  Query set size can be large and this could prevent evaluation without
  batching. Therefore we split the query set in to batches of size
  `query_size_limit`. In addition to these 2 datasets, the function returns
  total number of episodes(batches) after splitting the query set in
  `query_size_limit` batches.

  Args:
    vtab_key: str, one of constants.VTAB_DATASETS.
    image_size: int, used to resize the images read.
    query_size_limit: int, used to batch the query set.
    data_dir: str, optional data directory path for tf-datasets.

  Returns:
    tf.data.Dataset, of support set in one batch.
    tf.data.Dataset, of query set in batches of size `query_size_limit`.
    int, number of batches available in the query set.
  """
  dataset_instance = data_loader.get_dataset_instance({
      "dataset": "data.%s" % vtab_key,
      "data_dir": data_dir
  })
  query_ds = dataset_instance.get_tf_data(
      split_name="test",
      batch_size=query_size_limit,
      preprocess_fn=functools.partial(
          data_loader.preprocess_fn, input_range=(-1.0, 1.0), size=image_size),
      epochs=1,
      drop_remainder=False,
      for_eval=True,
      shuffle_buffer_size=0,
      prefetch=1,
  )
  n_query = math.ceil(
      dataset_instance.get_num_samples("test") / float(query_size_limit))
  support_ds = dataset_instance.get_tf_data(
      split_name="train800val200",
      # We get all 1000 images at once.
      batch_size=dataset_instance.get_num_samples("train800val200"),
      preprocess_fn=functools.partial(
          data_loader.preprocess_fn, input_range=(-1.0, 1.0), size=image_size),
      epochs=1,
      drop_remainder=False,
      for_eval=False,
      shuffle_buffer_size=0,
      prefetch=1,
  )
  n_classes = dataset_instance.get_num_classes()

  def cast_label_fn(ex):
    ex["label"] = tf.cast(ex["label"], tf.int32)
    return ex

  support_ds = support_ds.map(cast_label_fn)
  query_ds = query_ds.map(cast_label_fn)
  return support_ds, query_ds, n_query, n_classes
