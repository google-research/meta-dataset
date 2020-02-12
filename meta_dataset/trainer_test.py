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
"""Tests for trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import gin.tf
from meta_dataset import learner
from meta_dataset import trainer
from meta_dataset.data import config
from meta_dataset.data import decoder
from meta_dataset.data import providers
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS



class TrainerTest(tf.test.TestCase):
  """Test for the Trainer class.

  In order to run this test, the records root directory needs to be set via the
  `--records_root_dir` flag.
  """

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

    # Values that can't be passed directly to EpisodeDescriptionConfig
    gin.bind_parameter('process_episode.support_decoder',
                       decoder.ImageDecoder())
    gin.bind_parameter('process_episode.query_decoder', decoder.ImageDecoder())

    episode_config = config.EpisodeDescriptionConfig(
        num_ways=None,
        num_support=None,
        num_query=None,
        min_ways=5,
        max_ways_upper_bound=50,
        max_num_query=10,
        max_support_set_size=500,
        max_support_size_contrib_per_class=100,
        min_log_weight=np.log(0.5),
        max_log_weight=np.log(2),
        ignore_dag_ontology=False,
        ignore_bilevel_ontology=False)

    data_config = config.DataConfig(
        image_height=84,
        shuffle_buffer_size=20,
        read_buffer_size_bytes=(1024**2),
        num_prefetch=2,
    )

    episodic_trainer = trainer.EpisodicTrainer(
        train_learner=learner.PrototypicalNetworkLearner,
        eval_learner=learner.PrototypicalNetworkLearner,
        is_training=True,
        train_dataset_list=['mini_imagenet'],
        eval_dataset_list=['mini_imagenet'],
        restrict_classes={},
        restrict_num_per_class={},
        checkpoint_dir='',
        summary_dir='',
        records_root_dir=FLAGS.records_root_dir,
        eval_finegrainedness=False,
        eval_finegrainedness_split='',
        eval_imbalance_dataset='',
        omit_from_saving_and_reloading='',
        train_episode_config=episode_config,
        eval_episode_config=episode_config,
        learn_config=learn_config,
        learner_config=learner_config,
        data_config=data_config,
    )

    # Get the next train / valid / test episodes.
    train_episode = episodic_trainer.get_next(trainer.TRAIN_SPLIT)
    self.assertIsInstance(train_episode, providers.EpisodeDataset)

    # This isn't really a test. It just checks that things don't crash...
    print(
        episodic_trainer.sess.run([
            episodic_trainer.train_op,
            episodic_trainer.losses[trainer.TRAIN_SPLIT],
            episodic_trainer.accs[trainer.TRAIN_SPLIT]
        ]))


if __name__ == '__main__':
  tf.test.main()
