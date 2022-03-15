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
"""Tests for meta_dataset.data.{dump|read}_episodes."""
import collections
import json
import os
import random
import absl.testing.parameterized as parameterized

from meta_dataset.data import decoder
from meta_dataset.data import read_episodes
from meta_dataset.data import utils
from task_adaptation import data_loader
import tensorflow.compat.v1 as tf


def gen_rand_img_string(n):
  img_str = tf.io.encode_jpeg(
      tf.cast(
          tf.random.uniform(shape=(n, n, 3), maxval=255, dtype=tf.int32),
          dtype=tf.uint8))
  return img_str


class DumpAndReadEpisodesTest(tf.test.TestCase, parameterized.TestCase):

  def dump_episodes(self, n_episode, n_image_train=5, n_image_test=10):
    """Dumps `n_episode` many episodes to a temporary directory."""
    record_dir = self.get_temp_dir()
    episodes = []
    images_per_class_dict = {}
    for i in range(n_episode):
      train_path = utils.get_file_path(record_dir, i, 'train')
      train_images = tf.stack([
          gen_rand_img_string(random.randint(1, 10))
          for _ in range(n_image_train)
      ])
      train_labels = tf.random.uniform(
          shape=(n_image_train,), maxval=10, dtype=tf.int32)
      utils.dump_as_tfrecord(train_path, train_images, train_labels)
      test_path = utils.get_file_path(record_dir, i, 'test')
      test_images = tf.stack([
          gen_rand_img_string(random.randint(1, 10))
          for _ in range(n_image_test)
      ])
      test_labels = tf.random.uniform(
          shape=(n_image_test,), maxval=10, dtype=tf.int32)
      utils.dump_as_tfrecord(test_path, test_images, test_labels)
      decode_fn = lambda image, label: (tf.image.decode_image(image), label)
      train_ds = tf.data.Dataset.from_tensor_slices(
          (train_images, train_labels)).map(decode_fn)
      test_ds = tf.data.Dataset.from_tensor_slices(
          (test_images, test_labels)).map(decode_fn)
      episodes.append((train_ds, test_ds))
      images_per_class_dict[os.path.basename(train_path)] = (
          utils.get_label_counts(train_labels))
      images_per_class_dict[os.path.basename(test_path)] = (
          utils.get_label_counts(test_labels))

    info_path = utils.get_info_path(record_dir)
    with tf.io.gfile.GFile(info_path, 'w') as f:
      f.write(json.dumps(images_per_class_dict, indent=2))
    return record_dir, episodes

  def test_single_dataset(self):
    record_dir, episodes = self.dump_episodes(2)
    for i, (train_ds, test_ds) in enumerate(episodes):
      train_loader = read_episodes.read_episode_as_dataset(
          record_dir, i, 'train')
      for (example, (img2, label2)) in zip(train_loader, train_ds):
        img1, label1 = example['image'], example['label']
        self.assertAllEqual(img1, img2)
        self.assertAllEqual(label1, label2)
      test_loader = read_episodes.read_episode_as_dataset(record_dir, i, 'test')
      for (example, (img2, label2)) in zip(test_loader, test_ds):
        img1, label1 = example['image'], example['label']
        self.assertAllEqual(img1, img2)
        self.assertAllEqual(label1, label2)

  def test_single_dataset_with_info(self):
    record_dir, episodes = self.dump_episodes(2)
    for i, (train_ds, test_ds) in enumerate(episodes):
      _, num_imgs_per_class = read_episodes.read_episode_as_dataset(
          record_dir, i, 'train', with_info=True)
      all_train_labels = [label2.numpy() for _, label2 in train_ds]
      expected_counts_train = dict(collections.Counter(all_train_labels))
      self.assertDictEqual(expected_counts_train, num_imgs_per_class)
      _, num_imgs_per_class = read_episodes.read_episode_as_dataset(
          record_dir, i, 'test', with_info=True)
      all_test_labels = [label2.numpy() for _, label2 in test_ds]
      expected_counts_test = dict(collections.Counter(all_test_labels))
      self.assertDictEqual(expected_counts_test, num_imgs_per_class)

  def test_read_as_meta_dataset(self):
    record_dir, episodes = self.dump_episodes(2)
    episode_ds, n_episodes = read_episodes.read_episodes_from_records(
        record_dir)
    total_episodes = 0
    for episode, (train_ds, test_ds) in zip(episode_ds, episodes):
      total_episodes += 1
      train_episode, test_episode = episode
      for (example_string, (img2, label2)) in zip(train_episode, train_ds):
        example = decoder.read_example_and_parse_image(example_string)
        img1, label1 = example['image'], example['label']
        self.assertAllEqual(img1, img2)
        self.assertAllEqual(label1, label2)
      for (example_string, (img2, label2)) in zip(test_episode, test_ds):
        example = decoder.read_example_and_parse_image(example_string)
        img1, label1 = example['image'], example['label']
        self.assertAllEqual(img1, img2)
        self.assertAllEqual(label1, label2)
    self.assertEqual(total_episodes, n_episodes)

  @parameterized.parameters(
      ['dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'sun397', 'svhn'])
  def test_read_as_vtab(self, dataset_name):
    image_size = 84
    sds, qds, n_episodes, n_classes = read_episodes.read_vtab_as_episode(
        dataset_name, image_size)
    sds = sds.cache().repeat()  # Since we can have >=1 query batches.
    total_episodes = 0
    all_labels = set()
    for s_data, q_data in zip(sds, qds):
      s_images, s_labels = s_data['image'], s_data['label']
      q_images, q_labels = q_data['image'], q_data['label']
      total_episodes += 1
      all_labels |= set(s_labels.numpy()) | set(q_labels.numpy())
      print(dataset_name, q_images.shape, s_images.shape)
      self.assertAllEqual(s_images.shape[1:], [84, 84, 3])
      self.assertAllEqual(q_images.shape[1:], [84, 84, 3])
      self.assertEqual(s_images.shape[0], s_labels.shape[0])
      self.assertEqual(q_images.shape[0], q_labels.shape[0])
    self.assertLen(all_labels, n_classes)
    self.assertEqual(total_episodes, n_episodes)

  @parameterized.parameters([100, 500])
  def test_read_as_vtab_with_query_size_limit(self, query_size_limit):
    image_size = 84
    n_repeat = 2
    vtab_key = read_episodes.VTAB_NATURAL[0]
    dataset_instance = data_loader.get_dataset_instance({
        'dataset': 'data.%s' % vtab_key,
        'data_dir': None
    })
    sds, qds, n_episodes, _ = read_episodes.read_vtab_as_episode(
        vtab_key, image_size, query_size_limit=query_size_limit)
    sds = sds.cache().repeat()  # Since we can have >=1 query batches.
    qds = qds.cache().repeat()
    total_query_images = dataset_instance.get_num_samples('test')
    total_support_images = dataset_instance.get_num_samples('train800val200')
    query_images_seen = 0
    prev_s_labels = None
    for _, s_data, q_data in zip(range(n_episodes * n_repeat), sds, qds):
      s_images, s_labels = s_data['image'], s_data['label']
      q_images, _ = q_data['image'], q_data['label']
      self.assertEqual(s_images.shape[0], total_support_images)
      if prev_s_labels is not None:
        # The support set should be same in all episodes.
        self.assertAllEqual(s_labels, prev_s_labels)
      prev_s_labels = s_labels
      self.assertLessEqual(q_images.shape[0].value, query_size_limit)
      query_images_seen += q_images.shape[0].value
    self.assertEqual(query_images_seen, total_query_images * n_repeat)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
