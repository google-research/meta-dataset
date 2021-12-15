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

# Lint as: python2, python3
"""Tests for trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import textwrap

from absl import flags
from absl.testing import parameterized
import gin.tf
from meta_dataset import learners
from meta_dataset import trainer
from meta_dataset.data import config
from meta_dataset.data import dataset_spec
from meta_dataset.data import decoder
from meta_dataset.data import learning_spec
from meta_dataset.data import providers
from meta_dataset.data import test_utils
from meta_dataset.models import functional_backbones
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

tf.flags.DEFINE_string('records_root_dir', '',
                       'Root directory containing a subdirectory per dataset.')
FLAGS = flags.FLAGS



class TrainerTest(tf.test.TestCase):
  """Test for the Trainer class.

  In order to run this test, the records root directory needs to be set via the
  `--records_root_dir` flag.
  """

  def test_trainer(self):
    # PrototypicalNetworkLearner is built automatically and this test does not
    # have the opportunity to pass values to its constructor except through gin.
    gin.bind_parameter('PrototypicalNetworkLearner.backprop_through_moments',
                       True)
    gin.bind_parameter('PrototypicalNetworkLearner.transductive_batch_norm',
                       False)
    gin.bind_parameter('PrototypicalNetworkLearner.embedding_fn',
                       functional_backbones.four_layer_convnet)
    gin.bind_parameter('four_layer_convnet.weight_decay', 1e-4)

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
        ignore_bilevel_ontology=False,
        ignore_hierarchy_probability=0.,
        simclr_episode_fraction=0.)

    # Inspired from `learn/gin/default/debug_proto_mini_imagenet.gin`, but
    # building the objects explicitly.
    data_config = config.DataConfig(
        image_height=84,
        shuffle_buffer_size=20,
        read_buffer_size_bytes=(1024**2),
        num_prefetch=2,
    )

    trainer_instance = trainer.Trainer(
        train_learner_class=learners.PrototypicalNetworkLearner,
        eval_learner_class=learners.PrototypicalNetworkLearner,
        is_training=True,
        train_dataset_list=['mini_imagenet'],
        eval_dataset_list=['mini_imagenet'],
        restrict_classes={},
        restrict_num_per_class={},
        checkpoint_dir='',
        summary_dir='',
        records_root_dir=FLAGS.records_root_dir,
        eval_split=trainer.VALID_SPLIT,
        eval_finegrainedness=False,
        eval_finegrainedness_split='',
        eval_imbalance_dataset='',
        omit_from_saving_and_reloading='',
        train_episode_config=episode_config,
        eval_episode_config=episode_config,
        data_config=data_config,
        num_updates=100,
        batch_size=8,  # unused
        num_eval_episodes=10,
        checkpoint_every=10,
        validate_every=5,
        log_every=1,
        checkpoint_to_restore=None,
        learning_rate=1e-4,
        decay_learning_rate=True,
        decay_every=5000,
        decay_rate=0.5,
        experiment_name='test',
        pretrained_source='',
        distribute=False,
        enable_tf_optimizations=True,
        normalized_gradient_descent=False,
    )

    # Get the next train / valid / test episodes.
    train_episode = trainer_instance.next_data[trainer.TRAIN_SPLIT]
    self.assertIsInstance(
        tf.data.get_output_types(train_episode), providers.Episode)

    # This isn't really a test. It just checks that things don't crash...
    print(
        trainer_instance.sess.run([
            trainer_instance.train_op,
            trainer_instance.losses[trainer.TRAIN_SPLIT],
            trainer_instance.accuracies[trainer.TRAIN_SPLIT]
        ]))


class TrainerIntegrationTest(parameterized.TestCase, tf.test.TestCase):
  """Integration test for the whole procedure driven by the Trainer."""

  # Number of total examples (also way)
  NUM_EXAMPLES = 5
  FEAT_SIZE = 64

  # Spec for all-way 1-shot fake dataset. valid classes and examples are the
  # same as train, to force over-fitting.
  DATASET_SPEC = dataset_spec.DatasetSpecification(
      name=None,
      classes_per_split={
          learning_spec.Split.TRAIN: NUM_EXAMPLES,
          learning_spec.Split.VALID: NUM_EXAMPLES,
          learning_spec.Split.TEST: 0
      },
      images_per_class=dict(enumerate([2] * NUM_EXAMPLES * 2)),
      class_names=dict(
          enumerate(['%d' % i for i in range(NUM_EXAMPLES)] +
                    ['%d_v' % i for i in range(NUM_EXAMPLES)])),
      path=None,
      file_pattern='{}.tfrecords')

  BASE_GIN_CONFIG = textwrap.dedent("""\
      DataConfig.image_height = None
      DataConfig.shuffle_buffer_size = 0
      DataConfig.read_buffer_size_bytes = 1048576  # 1024 ** 2
      DataConfig.num_prefetch = 0

      process_episode.support_decoder = @FeatureDecoder()
      process_episode.query_decoder = @FeatureDecoder()
      process_batch.batch_decoder = @FeatureDecoder()
      FeatureDecoder.feat_len = {feat_len}

      EpisodeDescriptionConfig.min_ways = None
      EpisodeDescriptionConfig.max_ways_upper_bound = None
      EpisodeDescriptionConfig.max_num_query = None
      EpisodeDescriptionConfig.max_support_set_size = None
      EpisodeDescriptionConfig.max_support_size_contrib_per_class = None
      EpisodeDescriptionConfig.min_log_weight = None
      EpisodeDescriptionConfig.max_log_weight = None
      EpisodeDescriptionConfig.ignore_dag_ontology = False
      EpisodeDescriptionConfig.ignore_bilevel_ontology = False
      EpisodeDescriptionConfig.ignore_hierarchy_probability = 0.
      EpisodeDescriptionConfig.simclr_episode_fraction = 0.

      Trainer.batch_size = 10
      Trainer.checkpoint_every = 1000
      Trainer.validate_every = 1
      Trainer.log_every = 1
      Trainer.num_updates = 20
      Trainer.num_eval_episodes = 1
      Trainer.learning_rate = 1e-3
      Trainer.normalized_gradient_descent = False
      Trainer.decay_learning_rate = False
      Trainer.decay_every = None
      Trainer.decay_rate = None
      Trainer.checkpoint_to_restore = None
      Trainer.distribute = False
      Trainer.enable_tf_optimizations = True
      Trainer.eval_finegrainedness = False
      Trainer.eval_finegrainedness_split = None
      Trainer.eval_imbalance_dataset = False
      Trainer.eval_split = 'valid'
      Trainer.experiment_name = 'test_episodic_overfit'
      Trainer.omit_from_saving_and_reloading = ''
      Trainer.pretrained_source = ''
      Trainer.restrict_classes = {{}}
      Trainer.restrict_num_per_class = {{}}
      Trainer.summary_dir = None

      Learner.backprop_through_moments = True
      Learner.embedding_fn = @fully_connected_network
      Learner.transductive_batch_norm = False

      fully_connected_network.n_hidden_units = (64,)
      fully_connected_network.weight_decay = 0
      """).format(feat_len=FEAT_SIZE)

  # Defaults for different Learners
  baseline_config = textwrap.dedent("""\
      BaselineLearner.knn_in_fc = False
      BaselineLearner.knn_distance = 'l2'
      BaselineLearner.cosine_classifier = False
      BaselineLearner.cosine_logits_multiplier = 1
      BaselineLearner.use_weight_norm = False
      linear_classifier.weight_decay = 0
      Trainer.num_updates = 50
      """)

  baselinefinetune_config = '\n'.join((
      baseline_config,
      'BaselineFinetuneLearner.num_finetune_steps = 20',
      'BaselineFinetuneLearner.finetune_lr = 1e-3',
      'BaselineFinetuneLearner.finetune_all_layers = False',
      'BaselineFinetuneLearner.finetune_with_adam = True',
      'Trainer.num_updates = 100',
  ))

  proto_config = ''

  matching_config = 'MatchingNetworkLearner.exact_cosine_distance = False'

  maml_config = textwrap.dedent("""\
      MAMLLearner.adapt_batch_norm = False
      MAMLLearner.classifier_weight_decay = 0
      MAMLLearner.debug = False
      MAMLLearner.first_order = True
      MAMLLearner.additional_evaluation_update_steps = 0
      MAMLLearner.alpha = 0.1
      MAMLLearner.num_update_steps = 5
      MAMLLearner.proto_maml_fc_layer_init = False
      MAMLLearner.zero_fc_layer = True
      """)

  proto_maml_config = '\n'.join(
      (maml_config, 'MAMLLearner.proto_maml_fc_layer_init = True'))

  # TODO(lamblinp): RelationNetworkLearner expect a backbone with spatial
  # dimensions, and cannot be tested with a fully connected one.

  def setUp(self):
    super(TrainerIntegrationTest, self).setUp()
    self.rng = np.random.RandomState(20200505)

    # Set up tests by creating random fake examples to train on.
    self.temp_dir = self.get_temp_dir()
    dataset_dir = os.path.join(self.temp_dir, 'fake')
    tf.io.gfile.mkdir(dataset_dir)
    for example_idx in range(self.NUM_EXAMPLES):
      # Write the same random 2 examples (one support and one query) twice each:
      # - In a meta-train class (example_idx)
      # - In a meta-validation class (example_idx + NUM_EXAMPLES)
      fake_features = self.rng.randn(2, self.FEAT_SIZE).astype(np.float32)
      for class_offset in (0, self.NUM_EXAMPLES):
        class_idx = example_idx + class_offset
        tfrecord_path = os.path.join(
            dataset_dir, self.DATASET_SPEC.file_pattern.format(class_idx))
        test_utils.write_feature_records(
            features=fake_features, label=class_idx, output_path=tfrecord_path)
    # Record DATASET_SPEC as well, as it will be loaded by Trainer.
    with tf.io.gfile.GFile(os.path.join(dataset_dir, 'dataset_spec.json'),
                           'w') as f:
      json.dump(self.DATASET_SPEC.to_dict(), f, indent=2)

  def tearDown(self):
    # Gin settings should not persist between tests.
    gin.clear_config()
    super().tearDown()

  # TODO(evcu) Come-up with non-flaky tests.
  @parameterized.named_parameters(
      ('Baseline', learners.BaselineLearner, baseline_config, 0.5, 20),
      ('BaselineFinetune', learners.BaselineFinetuneLearner,
       baselinefinetune_config, 0.5, 20),
      ('ProtoNets', learners.PrototypicalNetworkLearner, proto_config),
      ('MatchingNets', learners.MatchingNetworkLearner, matching_config),
      ('MAML', learners.MAMLLearner, maml_config),
      ('ProtoMAML', learners.MAMLLearner, proto_maml_config))
  def test_episodic_overfit(self,
                            learner_class,
                            learner_config,
                            threshold=1.,
                            attempts=1):
    """Test that error goes down when training on a single episode.

    This can help check that the trained model and the evaluated one share
    the trainable parameters correctly.

    Args:
      learner_class: A subclass of Learner.
      learner_config: A string, the Learner-specific gin configuration string.
      threshold: A float (default 1.), the performance to reach at least once.
      attempts: An int (default 1), how many of the last steps should be checked
        when looking for a validation value reaching the threshold (default 1).
    """
    gin_config = '\n'.join((self.BASE_GIN_CONFIG, learner_config))
    gin.parse_config(gin_config)

    episode_config = config.EpisodeDescriptionConfig(
        num_ways=self.NUM_EXAMPLES, num_support=1, num_query=1)

    trainer_instance = trainer.Trainer(
        train_learner_class=learner_class,
        eval_learner_class=learner_class,
        is_training=True,
        train_dataset_list=['fake'],
        eval_dataset_list=['fake'],
        records_root_dir=self.temp_dir,
        checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
        train_episode_config=episode_config,
        eval_episode_config=episode_config,
        data_config=config.DataConfig(),
    )
    # Train 1 update at a time for the last `attempts - 1` steps.
    trainer_instance.num_updates -= (attempts - 1)
    trainer_instance.train()
    valid_accs = [trainer_instance.valid_acc]
    for _ in range(attempts - 1):
      trainer_instance.num_updates += 1
      trainer_instance.train()
      valid_accs.append(trainer_instance.valid_acc)
    self.assertGreaterEqual(max(valid_accs), threshold)


if __name__ == '__main__':
  tf.test.main()
