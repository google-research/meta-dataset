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

"""Tests for trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from meta_dataset import learner
from meta_dataset import trainer
from meta_dataset.data import config
from meta_dataset.data import providers
import tensorflow as tf


class TrainerTest(tf.test.TestCase):
  """Test for the Trainer class."""

  def test_episodic_trainer(self):
    # Inspired from `learn/gin/default/debug_proto_mini_imagenet.gin`, but
    # building the objects explicitly.
    learn_config = trainer.LearnConfig(
        num_updates=100,
        batch_size=8,  # unused
        num_eval_episodes=10,
        checkpoint_every=10,
        validate_every=5,
        log_every=1,
        transductive_batch_norm=False,
    )

    learner_config = trainer.LearnerConfig(
        episodic=True,
        train_learner='PrototypicalNet',
        eval_learner='PrototypicalNet',
        pretrained_checkpoint='',
        checkpoint_for_eval='',
        embedding_network='four_layer_convnet',
        learning_rate=1e-4,
        decay_learning_rate=True,
        decay_every=5000,
        decay_rate=0.5,
        experiment_name='test',
        pretrained_source='',
    )

    # PrototypicalNetworkLearner is built automatically and this test does not
    # have the opportunity to pass values to its constructor except through gin.
    gin.bind_parameter('PrototypicalNetworkLearner.weight_decay', 1e-4)

    data_config = config.DataConfig(
        image_height=84,
        shuffle_buffer_size=20,
        read_buffer_size_bytes=(1024**2),
    )

    episodic_trainer = trainer.EpisodicTrainer(
        train_learner=learner.PrototypicalNetworkLearner,
        eval_learner=learner.PrototypicalNetworkLearner,
        is_training=True,
        dataset_list=['mini_imagenet'],
        checkpoint_dir='',
        summary_dir='',
        eval_finegrainedness=False,
        eval_finegrainedness_split='',
        eval_imbalance_dataset='',
        num_train_classes=None,
        num_test_classes=None,
        num_train_examples=None,
        num_test_examples=None,
        learn_config=learn_config,
        learner_config=learner_config,
        data_config=data_config,
    )

    # Get the next train / valid / test episodes.
    train_episode = episodic_trainer.get_next('train')
    self.assertIsInstance(train_episode, providers.EpisodeDataset)

    # This isn't really a test. It just checks that things don't crash...
    print(
        episodic_trainer.sess.run([
            episodic_trainer.train_op, episodic_trainer.losses['train'],
            episodic_trainer.accs['train']
        ]))


if __name__ == '__main__':
  tf.test.main()
