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

"""Interface for a learner that uses BenchmarkReaderDataSource to get data."""
# TODO(lamblinp): Update variable names to be more consistent
# - target, class_idx, label
# - support, query

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import os

import gin.tf
from meta_dataset import learner
# The following import is needed for gin to know about DataConfig.
from meta_dataset.data import config  # pylint: disable=unused-import
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import providers
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import

# The following flag specifies substrings of variable names that should not be
# reloaded. `num_left_in_epoch' is a variable that influences the behavior of
# the EpochTrackers. Since the state of those trackers is not reloaded, neither
# should this variable. `finetune' is a substring of the names of the variables
# of the linear layer in the finetune baseline, e.g. `fc_finetune'. We may want
# to not attempt to reload this for example if we are reloading the baseline's
# weights for further episodic training. In that case, that fc layer will not be
# in the computational graph of the episodic model and an error will be thrown.
# If we are instead continuing the training of the baseline after pre-emption
# for example, `finetune' should not be included in this list since it should.
# then be reloaded. `linear_classifier' plays that role but for the MAML model.
tf.flags.DEFINE_string(
    'omit_from_saving_and_reloading', 'num_left_in_epoch,finetune,'
    'linear_classifier', 'A comma-separated string of substrings such that all '
    'variables containing them should not be saved and reloaded.')

FLAGS = tf.flags.FLAGS

# Enable TensorFlow optimizations. It can add a few minutes to the first
# calls to session.run(), but decrease memory usage.
ENABLE_TF_OPTIMIZATIONS = True
# Enable tf.data optimizations, which are applied to the input data pipeline.
# It may be helpful to disable them when investigating regressions due to
# changes in tf.data (see b/121130181 for instance), but they seem to be helpful
# (or at least not detrimental) in general.
ENABLE_DATA_OPTIMIZATIONS = True

EMBEDDING_KEYWORDS = ('conv', 'resnet')

DATASETS_WITH_EXAMPLE_SPLITS = ()
TF_DATA_OPTIONS = tf.data.Options()
if ENABLE_DATA_OPTIMIZATIONS:
  # The Options object can be used to control which static or dynamic
  # optimizations to apply.
  TF_DATA_OPTIONS.experimental_optimization.apply_default_optimizations = False


def apply_dataset_options(dataset):
  """Apply the module-wide set of dataset options to dataset.

  In particular, this is used to enable or disable tf.data optimizations.
  This applies to the whole pipeline, so we can just set it at the end.

  Args:
    dataset: a tf.data.Dataset object.

  Returns:
    A tf.data.Dataset object with options applied.
  """
  return dataset.with_options(TF_DATA_OPTIONS)


def compute_train_class_proportions(episode, shots, dataset_spec):
  """Computes the proportion of each class' examples in the support set.

  Args:
    episode: An EpisodeDataset.
    shots: A 1D Tensor whose length is the `way' of the episode that stores the
      shots for this episode.
    dataset_spec: A DatasetSpecification.

  Returns:
    class_props: A 1D Tensor whose length is the `way' of the episode, storing
      for each class the proportion of its examples that are in the support set.
  """
  # Get the total number of examples of each class in the dataset.
  num_dataset_classes = len(dataset_spec.images_per_class)
  num_images_per_class = [
      dataset_spec.get_total_images_per_class(class_id)
      for class_id in range(num_dataset_classes)
  ]

  # Get the (absolute) class ID's that appear in the episode.
  class_ids, _ = tf.unique(episode.train_class_ids)  # [?, ]

  # Make sure that class_ids are valid indices of num_images_per_class. This is
  # important since tf.gather will fail silently and return zeros otherwise.
  num_classes = tf.shape(num_images_per_class)[0]
  check_valid_inds_op = tf.assert_less(class_ids, num_classes)
  with tf.control_dependencies([check_valid_inds_op]):
    # Get the total number of examples of each class that is in the episode.
    num_images_per_class = tf.gather(num_images_per_class, class_ids)  # [?, ]

  # Get the proportions of examples of each class that appear in the train set.
  class_props = tf.truediv(shots, num_images_per_class)
  return class_props


def get_split_enum(split):
  """Returns the Enum value corresponding to the given split.

  Args:
    split: A String.

  Raises:
    ValueError: Split must be one of 'train', 'valid' or 'test'.
  """
  # Get the int representing the chosen split.
  if split == 'train':
    split_enum = learning_spec.Split.TRAIN
  elif split == 'valid':
    split_enum = learning_spec.Split.VALID
  elif split == 'test':
    split_enum = learning_spec.Split.TEST
  else:
    raise ValueError('Split must be one of "train", "valid" or "test".')
  return split_enum


def compute_episode_stats(episode):
  """Computes various episode stats: way, shots, and class IDs.

  Args:
    episode: An EpisodeDataset.

  Returns:
    way: An int constant tensor. The number of classes in the episode.
    shots: An int 1D tensor: The number of support examples per class.
    class_ids: An int 1D tensor: (absolute) class IDs.
  """
  # The train labels of the next episode.
  train_labels = episode.train_labels
  # Compute way.
  episode_classes, _ = tf.unique(train_labels)
  way = tf.size(episode_classes)
  # Compute shots.
  class_ids = tf.reshape(tf.range(way), [way, 1])
  class_labels = tf.reshape(train_labels, [1, -1])
  is_equal = tf.equal(class_labels, class_ids)
  shots = tf.reduce_sum(tf.cast(is_equal, tf.int32), axis=1)
  # Compute class_ids.
  class_ids, _ = tf.unique(episode.train_class_ids)
  return way, shots, class_ids


@gin.configurable
class LearnConfig(object):
  """A collection of values pertaining to learning."""

  def __init__(self, num_updates, batch_size, num_eval_episodes,
               checkpoint_every, validate_every, log_every,
               transductive_batch_norm):
    """Initializes a LearnConfig.

    Args:
      num_updates: An integer, the number of training updates.
      batch_size: An integer, the size of batches for non-episodic models.
      num_eval_episodes: An integer, the number of episodes for evaluation.
      checkpoint_every: An integer, the number of episodes between consecutive
        checkpoints.
      validate_every: An integer, the number of episodes between consecutive
        validatations.
      log_every: An integer, the number of episodes between consecutive logging.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
    """
    self.num_updates = num_updates
    self.batch_size = batch_size
    self.num_eval_episodes = num_eval_episodes
    self.checkpoint_every = checkpoint_every
    self.validate_every = validate_every
    self.log_every = log_every
    self.transductive_batch_norm = transductive_batch_norm


# TODO(manzagop): consider moving here num_updates/batch_size from LearnConfig.
# TODO(manzagop): consider combining train and eval learner and inferring
#   episodic.
@gin.configurable
class LearnerConfig(object):
  """A collection of values pertaining to the model."""

  def __init__(
      self,
      episodic,
      train_learner,
      eval_learner,
      pretrained_checkpoint,
      checkpoint_for_eval,
      embedding_network,
      learning_rate,
      decay_learning_rate,
      decay_every,
      decay_rate,
      experiment_name,
      pretrained_source,
  ):
    # pyformat: disable
    """Initializes a LearnerConfig.

    Args:
      episodic: A boolean, whether meta-training is episodic.
      train_learner: A string, the name of the learner to use for meta-training.
      eval_learner: A string, the name of the learner to use for
        meta-evaluation.
      pretrained_checkpoint: A string, the path to a checkpoint to use for
        initializing a model prior to training.
      checkpoint_for_eval: A string, the path to a checkpoint to restore for
        evaluation.
      embedding_network: A string, the embedding network to use.
      learning_rate: A float, the meta-learning learning rate.
      decay_learning_rate: A boolean, whether to decay the learning rate.
      decay_every: An integer, the learning rate is decayed for every multiple
        of this value.
      decay_rate: A float, the decay to apply to the learning rate.
      experiment_name: A string, a name for the experiment.
      pretrained_source: A string, the pretraining setup to use.
    """
    # pyformat: enable
    self.episodic = episodic
    self.train_learner = train_learner
    self.eval_learner = eval_learner
    self.pretrained_checkpoint = pretrained_checkpoint
    self.checkpoint_for_eval = checkpoint_for_eval
    self.embedding_network = embedding_network
    self.learning_rate = learning_rate
    self.decay_learning_rate = decay_learning_rate
    self.decay_every = decay_every
    self.decay_rate = decay_rate
    self.experiment_name = experiment_name
    self.pretrained_source = pretrained_source


@gin.configurable
class Trainer(object):
  """A Trainer for training a Learner on data provided by ReaderDataSource."""

  def __init__(self, train_learner, eval_learner, is_training, dataset_list,
               checkpoint_dir, summary_dir, eval_finegrainedness,
               eval_finegrainedness_split, eval_imbalance_dataset,
               num_train_classes, num_test_classes, num_train_examples,
               num_test_examples, learn_config, learner_config, data_config):
    """Initializes a Trainer.

    Args:
      train_learner: A Learner to be used for meta-training.
      eval_learner: A Learner to be used for meta-validation or meta-testing.
      is_training: Bool, whether or not to train or just evaluate.
      dataset_list: A list of names of datasets to include in the benchmark.
        This can be any subset of the supported datasets.
      checkpoint_dir: A string, the path to the checkpoint directory, or None if
        no checkpointing should occur.
      summary_dir: A string, the path to the checkpoint directory, or None if no
        summaries should be saved.
      eval_finegrainedness: Whether to perform binary ImageNet evaluation for
        assessing the performance on fine- vs coarse- grained tasks.
      eval_finegrainedness_split: The subgraph of ImageNet to perform the
        aforementioned analysis on. Notably, if this is 'train', we need to
        ensure that an training data is used episodically, even if the given
        model is the baseline model which usually uses batches for training.
      eval_imbalance_dataset: A dataset on which to perform evaluation for
        assessing how class imbalance affects performance in binary episodes. By
        default it is empty and no imbalance analysis is performed.
      num_train_classes: An int or None, the number of classes in episodic
        meta-training.
      num_test_classes: An int or None, the number of classes in episodic
        meta-testing.
      num_train_examples: An int or None, the number of support examples.
      num_test_examples: An int or None, the number of query examples.
      learn_config: A LearnConfig, the learning configuration.
      learner_config: A LearnerConfig, the learner configuration.
      data_config: A DataConfig, the data configuration.
    """
    self.train_learner_class = train_learner
    self.eval_learner_class = eval_learner
    self.is_training = is_training
    self.dataset_list = dataset_list
    self.checkpoint_dir = checkpoint_dir
    self.summary_dir = summary_dir
    self.eval_finegrainedness = eval_finegrainedness
    self.eval_finegrainedness_split = eval_finegrainedness_split
    self.eval_imbalance_dataset = eval_imbalance_dataset

    self.eval_split = 'test'
    if eval_finegrainedness:
      # The fine- vs coarse- grained evaluation may potentially be performed on
      # the training graph as it exhibits greater variety in this aspect.
      self.eval_split = eval_finegrainedness_split

    if eval_finegrainedness or eval_imbalance_dataset:
      # We restrict this analysis to the binary classification setting.
      tf.logging.info(
          'Forcing the number of {} classes to be 2, since '
          'the finegrainedness analysis is applied on binary '
          'classification tasks only.'.format(eval_finegrainedness_split))
      if eval_finegrainedness and eval_finegrainedness_split == 'train':
        num_train_classes = 2
      else:
        num_test_classes = 2

    self.num_train_classes = num_train_classes
    self.num_test_classes = num_test_classes
    self.num_train_examples = num_train_examples
    self.num_test_examples = num_test_examples
    msg = ('num_train_classes: {}, num_test_classes: {}, '
           'num_train_examples: {}, num_test_examples: {}').format(
               num_train_classes, num_test_classes, num_train_examples,
               num_test_examples)
    tf.logging.info(msg)

    self.learn_config = learn_config
    self.learner_config = learner_config

    if self.learn_config.transductive_batch_norm:
      tf.logging.warn('Using transductive batch norm!')

    # Only applicable for non-transudctive batch norm. The correct
    # implementation here involves computing the mean and variance based on the
    # support set and then using them to batch normalize the query set. During
    # meta-learning, we allow the gradients to flow through those moments.
    self.backprop_through_moments = True

    self.data_config = data_config
    # Get the image shape.
    self.image_shape = [data_config.image_height] * 2 + [3]

    # Create the benchmark specification.
    (self.benchmark_spec,
     self.valid_benchmark_spec) = self.get_benchmark_specification()
    if self.valid_benchmark_spec is None:
      # This means that ImageNet is not a dataset in the given benchmark spec.
      # In this case the validation will be carried out on randomly-sampled
      # episodes from the meta-validation sets of all given datasets.
      self.valid_benchmark_spec = self.benchmark_spec

    # Which splits to support depends on whether we are in the meta-training
    # phase or not. If we are, we need the train split, and the valid one for
    # early-stopping. If not, we only need the test split.
    if self.is_training:
      self.required_splits = ['train', 'valid']
    else:
      self.required_splits = [self.eval_split]

    # Get the training, validation and testing specifications.
    # Each is either an EpisodeSpecification or a BatchSpecification.
    split_episode_or_batch_specs = {}
    if 'train' in self.required_splits:
      split_episode_or_batch_specs['train'] = self._create_train_specification()
    for split in ['valid', 'test']:
      if split not in self.required_splits:
        continue
      split_episode_or_batch_specs[split] = self._create_held_out_specification(
          split)
    self.split_episode_or_batch_specs = split_episode_or_batch_specs

    # Get the next data (episode or batch) for the different splits.
    self.next_data = {}
    for split in self.required_splits:
      self.next_data[split] = self.build_data(split)

    # Initialize the learners.
    self.ema_object = None  # Using dummy EMA object for now.
    self.learners = {}
    self.embedding_fn = learner.NAME_TO_EMBEDDING_NETWORK[
        self.learner_config.embedding_network]
    if 'train' in self.required_splits:
      self.learners['train'] = (
          self.create_train_learner(self.train_learner_class,
                                    self.get_next('train')))
    if self.eval_learner_class is not None:
      for split in ['valid', 'test']:
        if split not in self.required_splits:
          continue
        self.learners[split] = self.create_eval_learner(self.eval_learner_class,
                                                        self.get_next(split))

    # Get the Tensors for the losses / accuracies of the different learners.
    self.losses = dict(
        zip(self.required_splits, [
            self.learners[split].compute_loss()
            for split in self.required_splits
        ]))
    self.accs = dict(
        zip(self.required_splits, [
            self.learners[split].compute_accuracy()
            for split in self.required_splits
        ]))

    # Set self.way, self.shots to Tensors for the way/shots of the next episode.
    self.set_way_shots_classes_logits_targets()

    # Get an optimizer and the operation for meta-training.
    self.train_op = None
    if self.is_training:
      global_step = tf.train.get_or_create_global_step()
      learning_rate = self.learner_config.learning_rate
      if self.learner_config.decay_learning_rate:
        learning_rate = tf.train.exponential_decay(
            self.learner_config.learning_rate,
            global_step,
            decay_steps=self.learner_config.decay_every,
            decay_rate=self.learner_config.decay_rate,
            staircase=True)
      tf.summary.scalar('learning_rate', learning_rate)
      self.optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train_op = self.get_train_op(global_step)

    vars_to_restore = []
    # Omit from reloading any variables that contains as a substring anything in
    # the following list. For example, those that track iterator state, as
    # iterator state is not saved.
    omit_substrings = FLAGS.omit_from_saving_and_reloading.split(',')
    tf.logging.info(
        'Omitting from saving / reloading any variable that '
        'contains any of the following substrings: %s' % omit_substrings)
    for var in tf.global_variables():
      if not any([substring in var.name for substring in omit_substrings]):
        vars_to_restore.append(var)
      else:
        tf.logging.info('Omitting variable %s' % var.name)
    self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=500)

    if self.checkpoint_dir is not None:
      if not tf.gfile.Exists(self.checkpoint_dir):
        tf.gfile.MakeDirs(self.checkpoint_dir)

    # Initialize a Session.
    self.initialize_session()
    self.create_summary_writer()

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def maybe_set_way_shots_classes_logits_targets(self, skip_train=False):
    """Wrapper for set_way_shots_classes_logits_targets.

    Args:
      skip_train: Whether to assign each of the way, shot, (absolute) class id,
        test_logits and test_targets of the training split to None. This is used
        when called with the training split from the batch trainer, since
        training does not happen in episodes there.
    """
    # The batch trainer receives episodes only for the valid and test splits.
    # Therefore for the train split there is no defined way and shots.
    (way, shots, class_props, class_ids, test_logits,
     test_targets) = [], [], [], [], [], []
    for split in self.required_splits:
      if split == 'train' and skip_train:
        (way_, shots_, class_props_, class_ids_, test_logits_,
         test_targets_) = [None] * 6
      else:
        way_, shots_, class_ids_ = compute_episode_stats(self.next_data[split])
        class_props_ = None
        if self.eval_imbalance_dataset:
          class_props_ = compute_train_class_proportions(
              self.next_data[split], shots_, self.eval_imbalance_dataset_spec)
        test_logits_ = self.learners[split].test_logits
        test_targets_ = self.learners[split].test_targets
      way.append(way_)
      shots.append(shots_)
      class_props.append(class_props_)
      class_ids.append(class_ids_)
      test_logits.append(test_logits_)
      test_targets.append(test_targets_)

    self.way = dict(zip(self.required_splits, way))
    self.shots = dict(zip(self.required_splits, shots))
    self.class_props = dict(zip(self.required_splits, class_props))
    self.class_ids = dict(zip(self.required_splits, class_ids))
    self.test_logits = dict(zip(self.required_splits, test_logits))
    self.test_targets = dict(zip(self.required_splits, test_targets))

  def create_summary_writer(self):
    """Create summaries and writer."""
    # Add summaries for the losses / accuracies of the different learners.
    standard_summaries = []
    for split in self.required_splits:
      loss_summary = tf.summary.scalar('%s_loss' % split, self.losses[split])
      acc_summary = tf.summary.scalar('%s_acc' % split, self.accs[split])
      standard_summaries.append(loss_summary)
      standard_summaries.append(acc_summary)

    # Add summaries for the way / shot / logits / targets of the learners.
    evaluation_summaries = self.add_eval_summaries()

    # All summaries.
    self.standard_summaries = tf.summary.merge(standard_summaries)
    self.evaluation_summaries = tf.summary.merge(evaluation_summaries)

    # Get a writer.
    self.summary_writer = None
    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir)
      if not tf.gfile.Exists(self.summary_dir):
        tf.gfile.MakeDirs(self.summary_dir)

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner.

    Args:
      train_learner_class: A train learner subclass of the Learner class.
      episode_or_batch: An EpisodeDataset or Batch.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner.

    Args:
      eval_learner_class: An eval learner subclass of the Learner class.
      episode: An EpisodeDataset.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def get_benchmark_specification(self):
    """Returns a BenchmarkSpecification."""
    valid_benchmark_spec = None  # a benchmark spec for validation only.
    data_spec_list, has_dag_ontology, has_bilevel_ontology = [], [], []
    for dataset_name in self.dataset_list:
      dataset_records_path = os.path.join(FLAGS.records_root_dir, dataset_name)

      dataset_spec_path = os.path.join(dataset_records_path, 'dataset_spec.pkl')
      if not tf.gfile.Exists(dataset_spec_path):
        raise ValueError(
            'Dataset specification for {} is not found in the expected path '
            '({}).'.format(dataset_name, dataset_spec_path))

      with tf.gfile.Open(dataset_spec_path, 'rb') as f:
        data_spec = pkl.load(f)

      # Replace outdated path of where to find the dataset's records.
      data_spec = data_spec._replace(path=dataset_records_path)

      if dataset_name in DATASETS_WITH_EXAMPLE_SPLITS:
        # Check the file_pattern field is correct now.
        if data_spec.file_pattern != '{}_{}.tfrecords':
          raise RuntimeError(
              'The DatasetSpecification should be regenerated, as it does not '
              'have the correct value for "file_pattern". Expected "%s", but '
              'got "%s".' % ('{}_{}.tfrecords', data_spec.file_pattern))

      tf.logging.info('Adding dataset {}'.format(data_spec.name))
      data_spec_list.append(data_spec)

      # Only ImageNet has a DAG ontology.
      has_dag = False
      if dataset_name == 'ilsvrc_2012':
        has_dag = True
      has_dag_ontology.append(has_dag)

      # Only Omniglot has a bi-level ontology.
      is_bilevel = True if dataset_name == 'omniglot' else False
      has_bilevel_ontology.append(is_bilevel)

      if self.eval_imbalance_dataset:
        self.eval_imbalance_dataset_spec = data_spec
        assert len(data_spec_list) == 1, ('Imbalance analysis is only '
                                          'supported on one dataset at a time.')

      # Validation should happen on ImageNet only.
      if dataset_name == 'ilsvrc_2012':
        valid_benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
            'valid_benchmark', self.image_shape, [data_spec], [has_dag],
            [is_bilevel])

    benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
        'benchmark', self.image_shape, data_spec_list, has_dag_ontology,
        has_bilevel_ontology)

    return benchmark_spec, valid_benchmark_spec

  def initialize_session(self):
    """Initializes a tf Session."""
    if ENABLE_TF_OPTIMIZATIONS:
      self.sess = tf.Session()
    else:
      rewriter_config = rewriter_config_pb2.RewriterConfig(
          disable_model_pruning=True,
          constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          remapping=rewriter_config_pb2.RewriterConfig.OFF,
          shape_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          function_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
          loop_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          memory_optimization=rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
      graph_options = tf.GraphOptions(rewrite_options=rewriter_config)
      session_config = tf.ConfigProto(graph_options=graph_options)
      self.sess = tf.Session(config=session_config)

    # Restore or initialize the variables.
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    if self.learner_config.checkpoint_for_eval:
      # Requested a specific checkpoint.
      self.saver.restore(self.sess, self.learner_config.checkpoint_for_eval)
      tf.logging.info(
          'Restored checkpoint: %s' % self.learner_config.checkpoint_for_eval)
    else:
      # Continue from the latest checkpoint if one exists.
      # This handles fault-tolerance.
      latest_checkpoint = None
      if self.checkpoint_dir is not None:
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
      if latest_checkpoint:
        self.saver.restore(self.sess, latest_checkpoint)
        tf.logging.info('Restored checkpoint: %s' % latest_checkpoint)
      else:
        tf.logging.info('No previous checkpoint.')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    # For episodic models, potentially use pretrained weights at the start of
    # training. If this happens it will overwrite the embedding weights, but
    # taking care to not restore the Adam parameters.
    if self.learner_config.pretrained_checkpoint and not self.sess.run(
        tf.train.get_global_step()):
      self.saver.restore(self.sess, self.learner_config.pretrained_checkpoint)
      tf.logging.info(
          'Restored checkpoint: %s' % self.learner_config.pretrained_checkpoint)
      # We only want the embedding weights of the checkpoint we just restored.
      # So we re-initialize everything that's not an embedding weight. Also,
      # since this episodic finetuning procedure is a different optimization
      # problem than the original training of the baseline whose embedding
      # weights are re-used, we do not reload ADAM's variables and instead learn
      # them from scratch.
      vars_to_reinit, embedding_var_names, vars_to_reinit_names = [], [], []
      for var in tf.global_variables():
        if (any(keyword in var.name for keyword in EMBEDDING_KEYWORDS) and
            'adam' not in var.name.lower()):
          embedding_var_names.append(var.name)
          continue
        vars_to_reinit.append(var)
        vars_to_reinit_names.append(var.name)
      tf.logging.info(
          'Initializing all variables except for %s.' % embedding_var_names)
      self.sess.run(tf.variables_initializer(vars_to_reinit))
      tf.logging.info('Re-initialized vars %s.' % vars_to_reinit_names)

  def _create_held_out_specification(self, split='test'):
    """Create an EpisodeSpecification for either validation or testing.

    Note that testing is done episodically whether or not training was episodic.
    This is why the different subclasses should not override this method.

    Args:
      split: one of 'valid' or 'test'

    Returns:
      an EpisodeSpecification.

    Raises:
      ValueError: Invalid split.
    """
    split_enum = get_split_enum(split)
    return learning_spec.EpisodeSpecification(split_enum, self.num_test_classes,
                                              self.num_train_examples,
                                              self.num_test_examples)

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training.

    Raises:
      NotImplementedError: Should be implemented in each subclass.
    """
    raise NotImplementedError('Abstract Method.')

  def build_data(self, split):
    raise NotImplementedError('Abstract method.')

  def build_episode(self, split):
    """Builds an EpisodeDataset containing the next data for "split".

    Args:
      split: A string, either 'train', 'valid', or 'test'.

    Returns:
      An EpisodeDataset.
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    benchmark_spec = (
        self.valid_benchmark_spec if split == 'valid' else self.benchmark_spec)
    (_, image_shape, dataset_spec_list, has_dag_ontology,
     has_bilevel_ontology) = benchmark_spec
    episode_spec = self.split_episode_or_batch_specs[split]
    dataset_split, num_classes, num_train_examples, num_test_examples = \
        episode_spec
    # TODO(lamblinp): Support non-square shapes if necessary. For now, all
    # images are resized to square, even if it changes the aspect ratio.
    image_size = image_shape[0]
    if image_shape[1] != image_size:
      raise ValueError(
          'Expected a square image shape, not {}'.format(image_shape))

    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:

      use_dag_ontology = has_dag_ontology[0]
      if self.eval_finegrainedness or self.eval_imbalance_dataset:
        use_dag_ontology = False
      data_pipeline = pipeline.make_one_source_episode_pipeline(
          dataset_spec_list[0],
          use_dag_ontology=use_dag_ontology,
          use_bilevel_ontology=has_bilevel_ontology[0],
          split=dataset_split,
          num_ways=num_classes,
          num_support=num_train_examples,
          num_query=num_test_examples,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          image_size=image_size)
    else:
      data_pipeline = pipeline.make_multisource_episode_pipeline(
          dataset_spec_list,
          use_dag_ontology_list=has_dag_ontology,
          use_bilevel_ontology_list=has_bilevel_ontology,
          split=dataset_split,
          num_ways=num_classes,
          num_support=num_train_examples,
          num_query=num_test_examples,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          image_size=image_size)
      data_pipeline = apply_dataset_options(data_pipeline)

    iterator = data_pipeline.make_one_shot_iterator()
    (support_images, support_labels, support_class_ids, query_images,
     query_labels, query_class_ids) = iterator.get_next()
    return providers.EpisodeDataset(
        train_images=support_images,
        test_images=query_images,
        train_labels=support_labels,
        test_labels=query_labels,
        train_class_ids=support_class_ids,
        test_class_ids=query_class_ids)

  def build_batch(self, split):
    """Builds a Batch object containing the next data for "split".

    Args:
      split: A string, either 'train', 'valid', or 'test'.

    Returns:
      An EpisodeDataset.
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    _, image_shape, dataset_spec_list, _, _ = self.benchmark_spec
    dataset_split, batch_size = self.split_episode_or_batch_specs[split]
    for dataset_spec in dataset_spec_list:
      if dataset_spec.name in DATASETS_WITH_EXAMPLE_SPLITS:
        raise ValueError(
            'Batch pipeline is used only at meta-train time, and does not '
            'handle datasets with example splits, which should only be used '
            'at meta-test (evaluation) time.')
    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:
      data_pipeline = pipeline.make_one_source_batch_pipeline(
          dataset_spec_list[0],
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          image_size=image_shape[0])
    else:
      data_pipeline = pipeline.make_multisource_batch_pipeline(
          dataset_spec_list,
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          image_size=image_shape[0])

    data_pipeline = apply_dataset_options(data_pipeline)
    iterator = data_pipeline.make_one_shot_iterator()
    images, class_ids = iterator.get_next()
    return providers.Batch(images=images, labels=class_ids)

  def get_next(self, split):
    """Returns the next batch or episode.

    Args:
      split: A str, one of 'train', 'valid', or 'test'.

    Raises:
      ValueError: Invalid split.
    """
    if split not in ['train', 'valid', 'test']:
      raise ValueError('Invalid split. Expected one of "train", "valid", or '
                       '"test".')
    return self.next_data[split]

  def get_train_op(self, global_step):
    """Returns the operation that performs a training update."""
    # UPDATE_OPS picks up batch_norm updates.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = self.optimizer.minimize(
          self.losses['train'], global_step=global_step)
    return train_op

  def get_updated_global_step(self):
    with tf.control_dependencies([self.train_op]):
      global_step = tf.identity(tf.train.get_global_step())
    return global_step

  def train(self):
    """The training loop."""
    global_step = self.sess.run(tf.train.get_global_step())
    tf.logging.info('Starting training from global_step: %d', global_step)
    updated_global_step = self.get_updated_global_step()

    # Dummy variables so that logging works even if called before evaluation.
    self.valid_acc = np.nan
    self.valid_ci = np.nan

    # Compute the initial validation performance before starting the training.
    self.maybe_evaluate(global_step)

    while global_step < self.learn_config.num_updates:
      # Perform the next update.
      (_, train_loss, train_acc, global_step) = self.sess.run([
          self.train_op, self.losses['train'], self.accs['train'],
          updated_global_step
      ])

      # Maybe validate, depending on the global step's value.
      self.maybe_evaluate(global_step)

      # Log training progress.
      if not global_step % self.learn_config.log_every:
        message = (
            'Update %d. Train loss: %f, Train accuracy: %f, '
            'Valid accuracy %f +/- %f.\n' %
            (global_step, train_loss, train_acc, self.valid_acc, self.valid_ci))
        tf.logging.info(message)

        # Update summaries.
        summaries = self.sess.run(self.standard_summaries)
        if self.summary_writer:
          self.summary_writer.add_summary(summaries, global_step)

      should_save = self.checkpoint_dir is not None
      if should_save and global_step % self.learn_config.checkpoint_every == 0:
        save_path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, 'model_%d.ckpt' % global_step))
        tf.logging.info('Model checkpoint saved: %s' % save_path)

  def maybe_evaluate(self, global_step):
    """Maybe perform evaluation, depending on the value of global_step."""
    if not global_step % self.learn_config.validate_every:
      # Get the validation accuracy and confidence interval.
      (valid_acc, valid_ci, valid_acc_summary,
       valid_ci_summary) = self.evaluate('valid')
      # Validation summaries are updated every time validation happens which is
      # every validate_every steps instead of log_every steps.
      if self.summary_writer:
        self.summary_writer.add_summary(valid_acc_summary, global_step)
        self.summary_writer.add_summary(valid_ci_summary, global_step)
      self.valid_acc = valid_acc
      self.valid_ci = valid_ci

  def evaluate(self, split):
    """Returns performance metrics across num_eval_trials episodes / batches."""
    num_eval_trials = self.learn_config.num_eval_episodes
    tf.logging.info(
        'Performing evaluation of the %s split using %d episodes...' %
        (split, num_eval_trials))
    accuracies = []
    for eval_trial_num in range(num_eval_trials):
      acc, summaries = self.sess.run(
          [self.accs[split], self.evaluation_summaries])
      accuracies.append(acc)
      # Write evaluation summaries.
      if split == self.eval_split and self.summary_writer:
        self.summary_writer.add_summary(summaries, eval_trial_num)
    tf.logging.info('Done.')

    mean_acc = np.mean(accuracies)
    ci_acc = np.std(accuracies) * 1.96 / np.sqrt(len(accuracies))  # confidence

    if split == 'test':
      tf.logging.info('Test accuracy: %f, +/- %f.\n' % (mean_acc, ci_acc))

    mean_acc_summary = tf.Summary()
    mean_acc_summary.value.add(tag='mean %s acc' % split, simple_value=mean_acc)
    ci_acc_summary = tf.Summary()
    ci_acc_summary.value.add(tag='%s acc CI' % split, simple_value=ci_acc)

    return mean_acc, ci_acc, mean_acc_summary, ci_acc_summary

  def add_eval_summaries_split(self, split):
    """Returns split's summaries of way / shot / classes / logits / targets."""
    split_eval_summaries = []
    way_summary = tf.summary.scalar('%s_way' % split, self.way[split])
    shots_summary = tf.summary.tensor_summary('%s_shots' % split,
                                              self.shots[split])
    classes_summary = tf.summary.tensor_summary('%s_class_ids' % split,
                                                self.class_ids[split])
    logits_summary = tf.summary.tensor_summary('%s_test_logits' % split,
                                               self.test_logits[split])
    targets_summary = tf.summary.tensor_summary('%s_test_targets' % split,
                                                self.test_targets[split])
    if self.eval_imbalance_dataset:
      class_props_summary = tf.summary.tensor_summary('%s_class_props' % split,
                                                      self.class_props[split])
      split_eval_summaries.append(class_props_summary)
    split_eval_summaries.append(way_summary)
    split_eval_summaries.append(shots_summary)
    split_eval_summaries.append(classes_summary)
    split_eval_summaries.append(logits_summary)
    split_eval_summaries.append(targets_summary)
    return split_eval_summaries


class EpisodicTrainer(Trainer):
  """A Trainer that trains a learner through a series of episodes."""

  def build_data(self, split):
    # An EpisodicTrainer will use episodes for all splits.
    return self.build_episode(split)

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training."""
    return learning_spec.EpisodeSpecification(
        learning_spec.Split.TRAIN, self.num_train_classes,
        self.num_train_examples, self.num_test_examples)

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode."""
    self.maybe_set_way_shots_classes_logits_targets()

  def add_eval_summaries(self):
    """Returns summaries of way / shot / classes/ logits / targets."""
    evaluation_summaries = []
    for split in self.required_splits:
      evaluation_summaries.extend(self.add_eval_summaries_split(split))
    return evaluation_summaries

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner."""
    return train_learner_class(True, self.learn_config.transductive_batch_norm,
                               self.backprop_through_moments, self.ema_object,
                               self.embedding_fn, episode_or_batch)

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner."""
    return eval_learner_class(False, self.learn_config.transductive_batch_norm,
                              self.backprop_through_moments, self.ema_object,
                              self.embedding_fn, episode)


class BatchTrainer(Trainer):
  """A Trainer that trains a learner through a series of batches."""

  def build_data(self, split):
    # The 'train' split is read as batches, 'valid' and 'test' as episodes.
    if split == 'train' and self.eval_split != 'train':
      return self.build_batch(split)
    elif split in ('train', 'valid', 'test'):
      return self.build_episode(split)
    else:
      raise ValueError('Unexpected value for split: {}'.format(split))

  def _get_num_total_classes(self):
    """Returns total number of classes in the benchmark."""
    # Total class is needed because for ImageNet instead of linearly assigning
    # class id's we use the hierarchy to do this, thus the assumption:
    # {train_ids} < {val_ids} < {test_ids} doesn't hold.
    total_classes = 0
    for dataset_spec in self.benchmark_spec.dataset_spec_list:
      for split in learning_spec.Split:
        total_classes += len(dataset_spec.get_classes(split))
    return total_classes

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training."""
    if self.eval_split == 'train':
      return learning_spec.EpisodeSpecification(
          learning_spec.Split.TRAIN, self.num_train_classes,
          self.num_train_examples, self.num_test_examples)
    else:
      return learning_spec.BatchSpecification(learning_spec.Split.TRAIN,
                                              self.learn_config.batch_size)

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode."""
    skip_train = True
    if self.eval_split == 'train':
      skip_train = False
    self.maybe_set_way_shots_classes_logits_targets(skip_train=skip_train)

  def add_eval_summaries(self):
    """Returns summaries of way / shot / class id's / logits / targets."""
    evaluation_summaries = []
    for split in self.required_splits:
      # In the Batch case, training is non-episodic but evaluation is episodic.
      if split == 'train' and self.eval_split != 'train':
        continue
      evaluation_summaries.extend(self.add_eval_summaries_split(split))
    return evaluation_summaries

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner."""
    num_total_classes = self._get_num_total_classes()
    is_training = False if self.eval_split == 'train' else True
    return train_learner_class(
        is_training, self.learn_config.transductive_batch_norm,
        self.backprop_through_moments, self.ema_object, self.embedding_fn,
        episode_or_batch, num_total_classes, self.num_test_classes)

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner."""
    num_total_classes = self._get_num_total_classes()
    return eval_learner_class(False, self.learn_config.transductive_batch_norm,
                              self.backprop_through_moments, self.ema_object,
                              self.embedding_fn, episode, num_total_classes,
                              self.num_test_classes)
