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
"""Interface for a learner that uses BenchmarkReaderDataSource to get data."""
# TODO(lamblinp): Update variable names to be more consistent
# - target, class_idx, label
# - support, query
# TODO(lamblinp): Simplify the logic around performing evaluation on the
# `TRAIN_SPLIT` by, for instance, recording which data is episodic, and which
# split it is coming from (independently from how it is used).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import os
import re

from absl import logging
import gin.tf
from meta_dataset import distribute_utils
from meta_dataset import learners
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import providers
from meta_dataset.data import read_episodes
from meta_dataset.learners import baseline_learners
from meta_dataset.learners import metric_learners
from meta_dataset.learners import experimental as experimental_learners  # pylint: disable=g-bad-import-order
from meta_dataset.models import functional_backbones
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

# Enable tf.data optimizations, which are applied to the input data pipeline.
# It may be helpful to disable them when investigating regressions due to
# changes in tf.data (see b/121130181 for instance), but they seem to be helpful
# (or at least not detrimental) in general.
ENABLE_DATA_OPTIMIZATIONS = True

TF_DATA_OPTIONS = tf.data.Options()
if not ENABLE_DATA_OPTIMIZATIONS:
  # The Options object can be used to control which static or dynamic
  # optimizations to apply.
  TF_DATA_OPTIONS.experimental_optimization.apply_default_optimizations = False

# Objective labels for hyperparameter optimization.
ACC_MEAN_FORMAT_STRING = '%s_acc/mean'
ACC_CI95_FORMAT_STRING = '%s_acc/ci95'

# TODO(eringrant): Use `learning_spec.Split.TRAIN`, `learning_spec.Split.VALID`,
# and `learning_spec.Split.TEST` instead of string constants, and replace all
# remaining string redefinitions.
TRAIN_SPLIT = 'train'
VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'

FLAGS = tf.flags.FLAGS

DATASET_CONDITIONAL_LEARNERS = []




DATASET_CONDITIONAL_LEARNERS = [
    baseline_learners.DatasetConditionalBaselineLearner,
    metric_learners.DatasetConditionalPrototypicalNetworkLearner,
    baseline_learners.DatasetLearner,
]


def is_dataset_conditional(learner_class):
  for learner in DATASET_CONDITIONAL_LEARNERS:
    if issubclass(learner_class, learner):
      return True
  return False


class UnexpectedSplitError(ValueError):

  def __init__(self,
               unexpected_split,
               expected_splits=(TRAIN_SPLIT, TEST_SPLIT, VALID_SPLIT)):
    super(UnexpectedSplitError,
          self).__init__('Split must be one of {}, but received `{}`. '.format(
              expected_splits, unexpected_split))


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


def compute_class_proportions(unique_class_ids, shots, dataset_spec):
  """Computes the proportion of the total number of examples appearing as shots.

  Args:
    unique_class_ids: A 1D int Tensor of unique class IDs.
    shots: A 1D Tensor of the number of shots for each class in
      `unique_class_ids`.
    dataset_spec: A DatasetSpecification that contains informations about the
      class labels in `unique_class_ids`.

  Returns:
    A 1D Tensor with the proportion of examples appearing as shots per class in
    `unique_class_ids`, normalized by the total number of examples for each
    class in the dataset according to `dataset_spec`.
  """
  # Get the total number of examples of each class in the dataset.
  num_dataset_classes = len(dataset_spec.images_per_class)
  num_images_per_class = [
      dataset_spec.get_total_images_per_class(class_id)
      for class_id in range(num_dataset_classes)
  ]

  # Make sure that `unique_class_ids` are valid indices of
  # `num_images_per_class`. This is important since `tf.gather` will fail
  # silently and return zeros otherwise.
  num_classes = tf.shape(num_images_per_class)[0]
  check_valid_inds_op = tf.assert_less(unique_class_ids, num_classes)
  with tf.control_dependencies([check_valid_inds_op]):
    # Get the total number of examples of each class that is in the episode.
    num_images_per_class = tf.gather(num_images_per_class,
                                     unique_class_ids)  # [?, ]

  # Get the proportions of examples of each class that appear in the episode.
  class_props = tf.truediv(shots, num_images_per_class)
  return class_props


def get_split_enum(split):
  """Returns the Enum value corresponding to the given split.

  Args:
    split: A string, one of TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT.

  Raises:
    UnexpectedSplitError: split not TRAIN_SPLIT, VALID_SPLIT, or TEST_SPLIT.
  """
  # Get the int representing the chosen split.
  if split == TRAIN_SPLIT:
    split_enum = learning_spec.Split.TRAIN
  elif split == VALID_SPLIT:
    split_enum = learning_spec.Split.VALID
  elif split == TEST_SPLIT:
    split_enum = learning_spec.Split.TEST
  else:
    raise UnexpectedSplitError(split)
  return split_enum


def restore_or_log_informative_error(saver, sess, checkpoint_to_restore):
  """Attempt to restore from `checkpoint_to_restore` in `sess` using `saver`."""
  try:
    saver.restore(sess, checkpoint_to_restore)
  except tf.errors.NotFoundError as e:
    logging.error('Tried to restore from checkpoint %s but failed.',
                  checkpoint_to_restore)
    raise e
  else:
    logging.info('Restored from checkpoint %s.', checkpoint_to_restore)


# TODO(eringrant): Split the current `Trainer` class into `Trainer` and
# `Evaluator` classes to partition the constructor arguments into meaningful
# groups.
# TODO(eringrant): Refactor the current `Trainer` class to more transparently
# deal with operations per split, since the present logic surrounding the
# `eval_finegrainedness_split` is confusing.
# TODO(eringrant): Better organize `Trainer` Gin configurations, which are
# currently set in many configuration files.
@gin.configurable('Trainer_flute')
class Trainer(object):
  """A Trainer for training a Learner on data provided by ReaderDataSource."""

  def __init__(
      self,
      num_updates,
      batch_size,
      num_eval_episodes,
      checkpoint_every,
      validate_every,
      log_every,
      train_learner_class,
      eval_learner_class,
      is_training,
      checkpoint_to_restore,
      dataset_classifier_to_restore,
      learning_rate,
      decay_learning_rate,
      decay_every,
      decay_rate,
      normalized_gradient_descent,
      optimizer_type,
      learn_rate_scheduler,
      meta_batch_size,
      sample_half_from_imagenet,
      num_gpus,
      experiment_name,
      pretrained_source,
      train_dataset_list,
      eval_dataset_list,
      restrict_classes,
      restrict_num_per_class,
      checkpoint_dir,
      summary_dir,
      records_root_dir,
      eval_finegrainedness,
      eval_finegrainedness_split,
      eval_imbalance_dataset,
      omit_from_saving_and_reloading,
      eval_split,
      train_episode_config,
      eval_episode_config,
      data_config,
      distribute,
      enable_tf_optimizations):
    # pyformat: disable
    """Initializes a Trainer.

    Args:
      num_updates: An integer, the number of training updates.
      batch_size: An integer, the size of batches for non-episodic models.
      num_eval_episodes: An integer, the number of episodes for evaluation.
      checkpoint_every: An integer, the number of episodes between consecutive
        checkpoints.
      validate_every: An integer, the number of episodes between consecutive
        validations.
      log_every: An integer, the number of episodes between consecutive logging.
      train_learner_class: A Learner to be used for meta-training.
      eval_learner_class: A Learner to be used for meta-validation or
        meta-testing.
      is_training: Bool, whether or not to train or just evaluate.
      checkpoint_to_restore: A string, the path to a checkpoint from which to
        restore variables.
      dataset_classifier_to_restore: A string, the path to a checkpoint from
        which to restore the dataset classifier variables.
      learning_rate: A float, the meta-learning learning rate.
      decay_learning_rate: A boolean, whether to decay the learning rate.
      decay_every: An integer, the learning rate is decayed for every multiple
        of this value.
      decay_rate: A float, the decay to apply to the learning rate.
      normalized_gradient_descent: A boolean, whether to use normalized
        gradient descent in addition to ADAM; improves stability for
        crosstransformers.
      optimizer_type: One of 'momentum' or 'adam'.
      learn_rate_scheduler: One of 'cosine_decay' or 'cosine_decay_restarts'.
      meta_batch_size: An integer, the meta-batch size. Only supported if
        normalized_gradient_descent is False. The processing of each batch of
        episodes or batches will be divided across GPUs, if num_gpus > 0.
      sample_half_from_imagenet: A bool, whether to sample twice as often from
        ImageNet than the other datasets (all other datasets are sampled equally
        often as each other).
      num_gpus: An int. The number of available gpus.
      experiment_name: A string, a name for the experiment.
      pretrained_source: A string, the pretraining setup to use.
      train_dataset_list: A list of names of datasets to train on. This can be
        any subset of the supported datasets.
      eval_dataset_list: A list of names of datasets to evaluate on either for
        validation during train or for final test evaluation, depending on the
        nature of the experiment, as dictated by `is_training'.
      restrict_classes: A dict that maps dataset names to a dict that specifies
        for each of TRAIN_SPLIT, VALID_SPLIT and TEST_SPLIT the number of
        classes to restrict to. This can lead to some classes of a particular
        split of a particular dataset never participating in episode creation.
      restrict_num_per_class: A dict that maps dataset names to a dict that
        specifies for each of TRAIN_SPLIT, VALID_SPLIT and TEST_SPLIT the number
        of examples per class to restrict to. For datasets / splits that are not
        mentioned, no restriction is applied. If restrict_num_per_class is the
        empty dict, no restriction is applied to any split of any dataset.
      checkpoint_dir: A string, the path to the checkpoint directory, or None if
        no checkpointing should occur.
      summary_dir: A string, the path to the checkpoint directory, or None if no
        summaries should be saved.
      records_root_dir: A string, the path to the dataset records directory.
      eval_finegrainedness: Whether to perform binary ImageNet evaluation for
        assessing the performance on fine- vs coarse- grained tasks.
      eval_finegrainedness_split: The subgraph of ImageNet to perform the
        aforementioned analysis on. Notably, if this is TRAIN_SPLIT, we need to
        ensure that an training data is used episodically, even if the given
        model is the baseline model which usually uses batches for training.
      eval_imbalance_dataset: A dataset on which to perform evaluation for
        assessing how class imbalance affects performance in binary episodes. By
        default it is empty and no imbalance analysis is performed.
      omit_from_saving_and_reloading: A list of strings that specifies
        substrings of variable names that should not be reloaded.
      eval_split: One of the constants TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT
        or None, according to the split whose results we want to
        use for the analysis.
      train_episode_config: An instance of EpisodeDescriptionConfig (in
        data/config.py). This is a config for setting the ways and shots of
        training episodes or the parameters for sampling them, if variable.
      eval_episode_config: An instance of EpisodeDescriptionConfig. Analogous to
        train_episode_config but used for eval episodes (validation or testing).
      data_config: A DataConfig, the data configuration.
      distribute: (Experimental) use tf.distribute to distribute computation
        across multiple GPUs using a MirroredStrategy.  This has been tested
        with CrossTransformers, and may work with other episodic learners.
        It will split Episodes into multiple EpisodePieces before passing them
        to the learners, where each EpisodePiece contains a portion of the
        original episode's query and support-set images.  Batch learners are
        not currently supported.
      enable_tf_optimizations: Enable TensorFlow optimizations. It can add a
        few minutes to the first calls to session.run(), but decrease memory
        usage.

    Raises:
      RuntimeError: If requested to meta-learn the initialization of the linear
          layer weights but they are unexpectedly omitted from saving/restoring.
      UnexpectedSplitError: If split configuration is not as expected.
    """
    # pyformat: enable
    self.num_updates = num_updates
    self.batch_size = batch_size
    self.num_eval_episodes = num_eval_episodes
    self.checkpoint_every = checkpoint_every
    self.validate_every = validate_every
    self.log_every = log_every

    self.checkpoint_to_restore = checkpoint_to_restore
    self.dataset_classifier_to_restore = dataset_classifier_to_restore
    self.learning_rate = learning_rate
    self.decay_learning_rate = decay_learning_rate
    self.decay_every = decay_every
    self.decay_rate = decay_rate
    self.experiment_name = experiment_name
    self.pretrained_source = pretrained_source

    self.train_learner_class = train_learner_class
    self.eval_learner_class = eval_learner_class
    self.is_training = is_training
    self.train_dataset_list = train_dataset_list
    self.eval_dataset_list = eval_dataset_list
    self.normalized_gradient_descent = normalized_gradient_descent
    self.optimizer_type = optimizer_type
    self.learn_rate_scheduler = learn_rate_scheduler
    self.meta_batch_size = meta_batch_size
    self.sample_half_from_imagenet = sample_half_from_imagenet
    self.num_gpus = num_gpus
    self.enable_tf_optimizations = enable_tf_optimizations
    if self.sample_half_from_imagenet and self.is_training and (
        len(train_dataset_list) == 1 or train_dataset_list[0] != 'ilsvrc_2012'):
      raise ValueError('Since `sample_half_from_imagenet` is True, expected '
                       'more than one training dataset, with ImageNet being '
                       'the first training dataset.')

    # Currently we are supporting single dataset when we read from fixed
    # datasets like VTAB or dumped episodes.
    # Check whether we evaluate on VTAB
    if (len(self.eval_dataset_list) == 1 and
        self.eval_dataset_list[0].startswith('vtab')):
      self._fixed_eval = 'vtab'
    elif (len(self.eval_dataset_list) == 1 and
          self.eval_dataset_list[0].startswith('dumped')):
      self._fixed_eval = 'dumped'
    else:
      self._fixed_eval = None
    self.restrict_classes = restrict_classes
    self.restrict_num_per_class = restrict_num_per_class
    self.checkpoint_dir = checkpoint_dir
    self.summary_dir = summary_dir
    self.records_root_dir = records_root_dir
    self.eval_finegrainedness = eval_finegrainedness
    self.eval_finegrainedness_split = eval_finegrainedness_split
    self.eval_imbalance_dataset = eval_imbalance_dataset
    self.omit_from_saving_and_reloading = omit_from_saving_and_reloading

    self.data_initializeable_iterators = []

    if eval_finegrainedness:
      # The fine- vs coarse- grained evaluation may potentially be performed on
      # the training graph as it exhibits greater variety in this aspect.
      self.eval_split = eval_finegrainedness_split
    elif eval_split:
      if eval_split not in (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT):
        raise UnexpectedSplitError(eval_split)
      self.eval_split = eval_split
    elif is_training:
      self.eval_split = VALID_SPLIT
    else:
      self.eval_split = TEST_SPLIT

    if eval_finegrainedness or eval_imbalance_dataset:
      # We restrict this analysis to the binary classification setting.
      logging.info(
          'Forcing the number of %s classes to be 2, since '
          'the finegrainedness analysis is applied on binary '
          'classification tasks only.', eval_finegrainedness_split)
      if eval_finegrainedness and eval_finegrainedness_split == TRAIN_SPLIT:
        train_episode_config.num_ways = 2
      else:
        eval_episode_config.num_ways = 2

    self.num_classes_train = train_episode_config.num_ways
    self.num_classes_eval = eval_episode_config.num_ways
    self.num_support_train = train_episode_config.num_support
    self.num_query_train = train_episode_config.num_query
    self.num_support_eval = eval_episode_config.num_support
    self.num_query_eval = eval_episode_config.num_query

    self.train_episode_config = train_episode_config
    self.eval_episode_config = eval_episode_config

    self.data_config = data_config

    # TODO(eringrant): Adapt these image-specific expectations to feature
    # inputs.
    self.image_shape = [data_config.image_height] * 2 + [3]
    # Create the benchmark specification.
    self.benchmark_spec = self.get_benchmark_specification()

    # Which splits to support depends on whether we are in the meta-training
    # phase or not. If we are, we need the train split, and the valid one for
    # early-stopping. If not, we only need the test split.
    self.required_splits = [TRAIN_SPLIT] if self.is_training else []
    self.required_splits += [self.eval_split]

    # Get the training, validation and testing specifications.
    # Each is either an EpisodeSpecification or a BatchSpecification.
    self.split_episode_or_batch_specs = dict(
        zip(self.required_splits,
            map(self.get_batch_or_episodic_specification,
                self.required_splits)))

    # For each split, needs_source stores whether the associated learner
    # requires knowing the dataset from which each batch / episode was sampled.
    self.needs_source = {}
    for split in self.required_splits:
      needs_source = (split == TRAIN_SPLIT and
                      is_dataset_conditional(self.train_learner_class)) or (
                          split != TRAIN_SPLIT and
                          is_dataset_conditional(self.eval_learner_class))
      if needs_source and distribute:
        raise NotImplementedError('Distributed training is not yet supported '
                                  'for dataset-conditional learners.')
      self.needs_source[split] = needs_source

    # Get the next data (episode or batch) for the different splits.
    self.next_data = dict(
        zip(self.required_splits, map(self.build_data, self.required_splits)))

    self.distribute = distribute
    if self.distribute:
      self.strategy = tf.distribute.MirroredStrategy()
    else:
      self.strategy = None

    # Create the global step to pass to the learners.
    global_step = tf.train.get_or_create_global_step()

    if len(self.required_splits) > 1:
      if issubclass(self.train_learner_class,
                    experimental_learners.ExperimentalLearner):
        assert issubclass(
            self.eval_learner_class, experimental_learners.ExperimentalLearner
        ), ('If the `Learner` for the train split is an `ExperimentalLearner`,'
            ' the `Learner` for the evaluation split must be as well, since '
            'otherwise parameters cannot be shared.')
      else:
        assert not issubclass(
            self.eval_learner_class, experimental_learners.ExperimentalLearner
        ), ('If the `Learner` for the evaluation split is not an '
            '`ExperimentalLearner`, the `Learner` for the train split must not'
            ' be either, since otherwise parameters cannot be shared.')

    # Initialize the learners.
    self.learners = {}
    for split in self.required_splits:
      if split == TRAIN_SPLIT:
        # The learner for the training split should only be in training mode if
        # the evaluation split is not the training split.
        learner_is_training = self.eval_split != TRAIN_SPLIT
        learner_class = self.train_learner_class
        tied_learner = None
      else:
        learner_is_training = False
        learner_class = self.eval_learner_class
        # Share parameters between the training and evaluation `Learner`s.
        tied_learner = (
            self.learners[TRAIN_SPLIT]
            if TRAIN_SPLIT in self.required_splits else None)

      learner = self.create_learner(
          is_training=learner_is_training,
          learner_class=learner_class,
          split=get_split_enum(split),
          tied_learner=tied_learner)

      if (isinstance(learner, learners.MAMLLearner) and
          not learner.zero_fc_layer and not learner.proto_maml_fc_layer_init):
        if 'linear_classifier' in FLAGS.omit_from_saving_and_reloading:
          raise ValueError('The linear layer is requested to be meta-learned '
                           'since both `MAMLLearner.zero_fc_layer` and '
                           '`MAMLLearner.proto_maml_fc_layer_init` are False, '
                           'but the `linear_classifier` tags is found in '
                           'FLAGS.omit_from_saving_and_reloading so they will '
                           'not be properly restored. Please exclude these '
                           'weights from omit_from_saving_and_reloading for '
                           'this setting to work as expected.')

      self.learners[split] = learner

    # Build the data-dependent functions (run_fn returns prediction,
    # un-regularized loss,  accuracy, and episode statistics), the iterators
    # producing data (data_fn), and regularizer, for each learner.
    run_fn, data_fn, regularizer_fn = zip(
        *[self.build_learner(split) for split in self.required_splits])
    self.run_fns = dict(zip(self.required_splits, run_fn))
    self.data_fns = dict(zip(self.required_splits, data_fn))
    self.regularizer_fns = dict(zip(self.required_splits, regularizer_fn))
    # Get an optimizer and the operation for meta-training.
    self.train_op = None
    if self.is_training:
      learning_rate = self.learning_rate
      if self.decay_learning_rate:
        if self.learn_rate_scheduler == 'cosine_decay':
          learning_rate = tf.compat.v1.train.cosine_decay(
              self.learning_rate, global_step, self.decay_every)
        elif self.learn_rate_scheduler == 'cosine_decay_restarts':
          learning_rate = tf.train.cosine_decay_restarts(
              self.learning_rate,
              global_step,
              first_decay_steps=self.decay_every)
        elif self.learn_rate_scheduler == 'exponential_decay':
          learning_rate = tf.train.exponential_decay(
              self.learning_rate,
              global_step,
              decay_steps=self.decay_every,
              decay_rate=self.decay_rate,
              staircase=True)
      tf.summary.scalar('learning_rate', learning_rate)
      if self.distribute:
        if self.optimizer_type != 'adam':
          raise ValueError('Expected `optimizer_type` to be `adam` when '
                           '`distribute` is `True`.')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
      else:
        if self.optimizer_type == 'adam':
          self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer_type == 'momentum':
          self.optimizer = tf.train.MomentumOptimizer(
              learning_rate, momentum=0.9)

      self.run_fns[TRAIN_SPLIT] = self.get_run_fn_with_train_op(
          self.run_fns[TRAIN_SPLIT], self.regularizer_fns[TRAIN_SPLIT],
          global_step)
    self.predictions = {}
    self.losses = {}
    self.accuracies = {}
    self.episode_info = {}
    for split in self.required_splits:
      if self.distribute:
        with self.strategy.scope():
          output = self.strategy.experimental_run(self.run_fns[split],
                                                  self.data_fns[split])
          if self.strategy.num_replicas_in_sync > 1:
            output['predictions'] = tf.concat(
                output['predictions'].values, axis=0)
            output['loss'] = tf.concat(output['loss'].values, axis=0)
            output['accuracy'] = tf.concat(output['accuracy'].values, axis=0)
            if split == TRAIN_SPLIT and self.is_training:
              output['train_op'] = tf.group(output['train_op'].values)

            # The computed episode_info should be identical for all replicas.
            episode_info = {}
            for key, val in output['episode_info'].items():
              if val is not None:
                # This control_dependencies is required or the call to
                # session.run() gets deadlocked.
                with tf.control_dependencies(val.values):
                  episode_info[key] = tf.identity(val.values[0])
              else:
                episode_info[key] = None
            output['episode_info'] = episode_info

      else:
        data_tensors = tf.data.make_one_shot_iterator(
            self.data_fns[split]()).get_next()
        output = self.run_fns[split](data_tensors)

      loss = tf.reduce_mean(output['loss'])
      loss += self.regularizer_fns[split]()

      self.losses[split] = loss
      self.accuracies[split] = tf.reduce_mean(output['accuracy'])
      self.predictions[split] = output['predictions']
      self.episode_info[split] = output['episode_info']

      if split == TRAIN_SPLIT and self.is_training:
        self.train_op = output['train_op']


    if self.checkpoint_dir is not None:
      if not tf.io.gfile.exists(self.checkpoint_dir):
        tf.io.gfile.makedirs(self.checkpoint_dir)

    # Meaningless values so that logging works even if called before evaluation.
    self.valid_acc = np.nan
    self.valid_ci = np.nan

    self.initialize_session()
    self.initialize_saver()
    self.create_summary_writer()

  def build_learner(self, split):
    """Return predictions, losses and accuracies for the learner on split.

    Args:
      split: A `learning_spec.Split` that identifies the data split for which
        the learner is to be built.

    Returns:
      run_fn: a function which, when called on data, will run the network
        forward pass in split mode `split` and return:
          predictions: A `tf.Tensor`; the predictions of the learner on `split`.
          losses: A `tf.Tensor`; the losses of the learner on `split`.
          accuracies: A `tf.Tensor`; the accuracies of the learner on `split`.
          episode_info: A map of string to `tf.Tensor`, which has statistics
            about the input data.
      data_fn: a function which returns a `tf.Dataset` which can be used to
        provide input to run_fn
      regularizer: a `tf.Tensor` which computes a data-independent regularizer
        (e.g. weight decay) that is to be applied at every iteration.


    """
    learner = self.learners[split]

    # Build the learner and its variables outside the name scope.
    learner.build()

    with tf.name_scope(split):
      data_src = self.next_data[split]
      if self.distribute:
        with self.strategy.scope():
          # We need to split both support and query sets across GPUs, and
          # tf.data doesn't make this straightforward, as there are few
          # functions for splitting up datasets.  We use unbatch to accomplish
          # this, but unbatch splits the first dimension and creates one
          # example for every possible index along that dimension.
          #
          # Therefore, the strategy is to first compute the chunk boundaries
          # for the support and query sets (chunk_bounds), pad the last
          # chunks to match the earlier chunks along the first axis,
          # and then stack all chunks along a new first axis.
          # unbatch() then correctly splits the episode across gpus.  We then
          # trim the padding from later chunks (trim_extra). The result is a
          # single dataset with chunks for each GPU interleaved with one
          # another.  We use shard() to split this single dataset into one
          # dataset per gpu.

          def chunk_bounds(x, num_gpu, idx):
            num_per_gpu = tf.cast(
                tf.ceil(
                    tf.cast(tf.shape(x)[0], tf.float32) /
                    tf.cast(num_gpu, tf.float32)), tf.int32)
            lb = tf.minimum(idx * num_per_gpu, tf.shape(x)[0])
            ub = tf.minimum((idx + 1) * num_per_gpu, tf.shape(x)[0])
            return lb, ub, tf.minimum((idx + 1) * num_per_gpu - ub, num_per_gpu)

          def chunk_array(arr):
            """Deterministically chunk the arrays across devices."""
            num_gpu = self.strategy.num_replicas_in_sync
            chunks = []
            num = []
            for idx in range(num_gpu):
              lb, ub, num_extra = chunk_bounds(arr, num_gpu, idx)
              pad = tf.tile(arr[0:1], [num_extra] + [1] *
                            (len(arr.shape) - 1)) * 0 - 1
              chunks.append(tf.concat([arr[lb:ub], pad], axis=0))
              num.append(ub - lb)
            return tf.stack(chunks), tf.stack(num)

          def chunk_episode(episode):
            way_tiled = episode.way + tf.zeros(
                [self.strategy.num_replicas_in_sync], dtype=tf.int32)
            return ((way_tiled, chunk_array(episode.support_images)[0],
                     chunk_array(episode.support_labels)[0]) +
                    chunk_array(episode.support_class_ids) +
                    (chunk_array(episode.query_images)[0],
                     chunk_array(episode.query_labels)[0]) +
                    chunk_array(episode.query_class_ids))

          def trim_extra(way, support_images, support_labels, support_class_ids,
                         support_num, query_images, query_labels,
                         query_class_ids, query_num):
            return providers.EpisodePiece(support_images[:support_num],
                                          query_images[:query_num],
                                          support_labels[:support_num],
                                          query_labels[:query_num],
                                          support_class_ids[:support_num],
                                          query_class_ids[:query_num], way)

          chunked_data = data_src.map(chunk_episode).unbatch().map(trim_extra)

          def input_fn(input_context):
            return chunked_data.shard(input_context.num_input_pipelines,
                                      input_context.input_pipeline_id)

          data = self.strategy.make_input_fn_iterator(input_fn)
          regularizer = self.learners[split].compute_regularizer
          self.data_initializeable_iterators.append(data)
      else:
        data = lambda: data_src
        regularizer = self.learners[split].compute_regularizer

      def run(data_local):
        """Run the forward pass of the model."""
        args = [data_local]
        kwargs = {}
        data = data_local
        if self.needs_source[split]:
          data, source = data_local
          args = [data, source]
          kwargs = {'source': source} if split == TRAIN_SPLIT else {}

        onehot_labels = data.onehot_labels
        if (split == TRAIN_SPLIT and issubclass(
            self.train_learner_class, baseline_learners.DatasetLearner) or
            split == VALID_SPLIT and issubclass(
                self.eval_learner_class, baseline_learners.DatasetLearner)):
          # In the case of a dataset classifier, the target is the dataset ID.
          num_sets = len(self.train_dataset_list)
          one_hot_source = tf.expand_dims(tf.one_hot(source, depth=num_sets), 0)
          onehot_labels = one_hot_source

        predictions_dist = self.learners[split].forward_pass(*args)
        loss_dist = self.learners[split].compute_loss(
            predictions=predictions_dist, onehot_labels=onehot_labels, **kwargs)
        accuracy_dist = self.learners[split].compute_accuracy(
            predictions=predictions_dist, onehot_labels=onehot_labels, **kwargs)
        episode_info = self.get_episode_info(data)
        return {
            'predictions': predictions_dist,
            'loss': loss_dist,
            'accuracy': accuracy_dist,
            'episode_info': episode_info,
        }

      return run, data, regularizer

  def get_episode_info(self, data):
    """Sets the Tensors for the info about the learner's next episode."""
    res = {}
    # The batch trainer receives episodes only for the valid and test splits.
    # Therefore for the train split there is no defined way and shots.
    if isinstance(data, providers.Batch):
      (way_, shots_, class_props_, class_ids_, query_targets_) = [None] * 5
    else:
      way_ = data.way
      shots_ = data.support_shots
      class_ids_ = data.unique_class_ids
      class_props_ = None
      if self.eval_imbalance_dataset:
        class_props_ = compute_class_proportions(
            class_ids_, shots_, self.eval_imbalance_dataset_spec)
      query_targets_ = data.query_labels
    res['way'] = way_
    res['shots'] = shots_
    res['class_props'] = class_props_
    res['class_ids'] = class_ids_
    res['query_targets'] = query_targets_
    return res

  def create_summary_writer(self):
    """Create summaries and writer."""
    # Add summaries for the losses / accuracies of the different learners.
    standard_summaries = []
    for split in self.required_splits:
      with tf.name_scope(split):
        loss_summary = tf.summary.scalar('loss', self.losses[split])
        acc_summary = tf.summary.scalar('acc',
                                        tf.reduce_mean(self.accuracies[split]))
      standard_summaries.append(loss_summary)
      standard_summaries.append(acc_summary)

    # Add summaries for the way / shot / logits / targets of the learner.
    evaluation_summaries = self.add_eval_summaries()

    # All summaries.
    self.standard_summaries = tf.summary.merge(standard_summaries)
    self.evaluation_summaries = tf.summary.merge(evaluation_summaries)

    # Get a writer.
    self.summary_writer = None
    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir)
      if not tf.io.gfile.exists(self.summary_dir):
        tf.io.gfile.makedirs(self.summary_dir)

  def create_learner(self,
                     is_training,
                     learner_class,
                     split,
                     tied_learner=None):
    """Instantiates `learner_class`, tying weights to `tied_learner`."""
    if issubclass(learner_class, learners.BatchLearner):
      logit_dim = self._get_logit_dim(
          split,
          is_batch_learner=True,
          is_training=is_training,
          return_list=is_dataset_conditional(learner_class))
    elif issubclass(learner_class, learners.EpisodicLearner):
      logit_dim = self._get_logit_dim(
          split, is_batch_learner=False, is_training=is_training)
    else:
      raise ValueError('The specified `learner_class` should be a subclass of '
                       '`learners.BatchLearner` or `learners.EpisodicLearner`, '
                       'but received {}.'.format(learner_class))

    if (issubclass(learner_class, experimental_learners.ExperimentalLearner) and
        tied_learner is not None):
      return learner_class(
          is_training=is_training,
          logit_dim=logit_dim,
          input_shape=self.image_shape,
          embedding_fn=tied_learner.embedding_fn,
      )
    else:
      return learner_class(
          is_training=is_training,
          logit_dim=logit_dim,
          input_shape=self.image_shape,
      )

  def get_benchmark_specification(self, records_root_dir=None):
    """Returns a BenchmarkSpecification.

    Args:
      records_root_dir: Optional. If provided, a list or string that sets the
        directory in which a child directory will be searched for each dataset
        to locate that dataset's records and dataset specification. If it's a
        string, that path will be used for all datasets. If it's a list, its
        length must be the same as the number of datasets, in order to specify a
        different such directory for each. If None, self.records_root_dir will
        be used for all datasets.

    Raises:
      RuntimeError: Incorrect file_pattern detected in a dataset specification.
    """
    (data_spec_list, has_dag_ontology, has_bilevel_ontology,
     splits_to_contribute) = [], [], [], []
    seen_datasets = set()

    eval_dataset_list = self.eval_dataset_list
    # No need to read specs when specs not available and not needed.
    if self._fixed_eval:
      eval_dataset_list = []
    if self.is_training:
      benchmark_datasets = self.train_dataset_list + eval_dataset_list
    else:
      benchmark_datasets = eval_dataset_list

    if isinstance(records_root_dir, list):
      if len(records_root_dir) != len(benchmark_datasets):
        raise ValueError('The given records_root_dir is a list whose length is '
                         'not the same as the number of benchmark datasets. '
                         'Found datasets {} (for the {} phase) but '
                         'len(records_root_dir) is {}. Expected their lengths '
                         'to match or records_path to be a string').format(
                             benchmark_datasets, len(records_root_dir))
      records_roots_for_datasets = records_root_dir
    elif isinstance(records_root_dir, six.text_type):
      records_roots_for_datasets = [records_root_dir] * len(benchmark_datasets)
    elif records_root_dir is None:
      records_roots_for_datasets = [self.records_root_dir
                                   ] * len(benchmark_datasets)

    for dataset_name, dataset_records_root in zip(benchmark_datasets,
                                                  records_roots_for_datasets):

      # Might be seeing a dataset for a second time if it belongs to both the
      # train and eval dataset lists.
      if dataset_name in seen_datasets:
        continue

      dataset_records_path = os.path.join(dataset_records_root, dataset_name)
      data_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
      # Only ImageNet has a DAG ontology.
      has_dag = (dataset_name.startswith('ilsvrc_2012'))
      # Only Omniglot has a bi-level ontology.
      is_bilevel = (dataset_name == 'omniglot')

      # The meta-splits that this dataset will contribute data to.
      if not self.is_training:
        # If we're meta-testing, all datasets contribute only to meta-test.
        splits = {self.eval_split}
      else:
        splits = set()
        if dataset_name in self.train_dataset_list:
          splits.add(TRAIN_SPLIT)
        if dataset_name in self.eval_dataset_list:
          splits.add(VALID_SPLIT)

      # By default, all classes of each split will eventually be used for
      # episode creation. But it might be that for some datasets, it is
      # requested to restrict the available number of classes of some splits.
      restricted_classes_per_split = {}
      if dataset_name in self.restrict_classes:
        classes_per_split = self.restrict_classes[dataset_name]
        for split, num_classes in classes_per_split.items():
          # The option to restrict classes is not supported in conjuction with
          # non-uniform (bilevel or hierarhical) class sampling.
          episode_descr_config = (
              self.train_episode_config
              if split == TRAIN_SPLIT else self.eval_episode_config)
          if has_dag and not episode_descr_config.ignore_dag_ontology:
            raise ValueError('Restrictions on the class set of a dataset with '
                             'a DAG ontology are not supported when '
                             'ignore_dag_ontology is False.')
          if is_bilevel and not episode_descr_config.ignore_bilevel_ontology:
            raise ValueError('Restrictions on the class set of a dataset with '
                             'a bilevel ontology are not supported when '
                             'ignore_bilevel_ontology is False.')

          restricted_classes_per_split[get_split_enum(split)] = num_classes
        # Initialize the DatasetSpecificaton to account for this restriction.
        data_spec.initialize(restricted_classes_per_split)

        # Log the applied restrictions.
        logging.info('Restrictions for dataset %s:', dataset_name)
        for split in list(splits):
          num_classes = data_spec.get_classes(get_split_enum(split))
          logging.info('\t split %s is restricted to %d classes', split,
                       num_classes)

      # Add this dataset to the benchmark.
      logging.info('Adding dataset %s', data_spec.name)
      data_spec_list.append(data_spec)
      has_dag_ontology.append(has_dag)
      has_bilevel_ontology.append(is_bilevel)
      splits_to_contribute.append(splits)

      # Book-keeping.
      seen_datasets.add(dataset_name)

    if self.eval_imbalance_dataset:
      self.eval_imbalance_dataset_spec = data_spec
      assert len(data_spec_list) == 1, ('Imbalance analysis is only '
                                        'supported on one dataset at a time.')

    benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
        'benchmark', self.image_shape, data_spec_list, has_dag_ontology,
        has_bilevel_ontology, splits_to_contribute)

    # Logging of which datasets will be used for the different meta-splits.
    splits_to_datasets = collections.defaultdict(list)
    for dataset_spec, splits_to_contribute in zip(data_spec_list,
                                                  splits_to_contribute):
      for split in splits_to_contribute:
        splits_to_datasets[split].append(dataset_spec.name)
    for split, datasets in splits_to_datasets.items():
      logging.info('Episodes for split %s will be created from %s', split,
                   datasets)

    return benchmark_spec

  def initialize_session(self):
    """Initializes a tf.Session."""
    if self.enable_tf_optimizations:
      self.sess = tf.Session()
    else:
      session_config = tf.ConfigProto()
      rewrite_options = session_config.graph_options.rewrite_options
      rewrite_options.disable_model_pruning = True
      rewrite_options.constant_folding = rewrite_options.OFF
      rewrite_options.arithmetic_optimization = rewrite_options.OFF
      rewrite_options.remapping = rewrite_options.OFF
      rewrite_options.shape_optimization = rewrite_options.OFF
      rewrite_options.dependency_optimization = rewrite_options.OFF
      rewrite_options.function_optimization = rewrite_options.OFF
      rewrite_options.layout_optimizer = rewrite_options.OFF
      rewrite_options.loop_optimization = rewrite_options.OFF
      rewrite_options.memory_optimization = rewrite_options.NO_MEM_OPT
      self.sess = tf.Session(config=session_config)

    # Restore or initialize the variables.
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    for it in self.data_initializeable_iterators:
      self.sess.run(it.initialize())

  def initialize_saver(self):
    """Initializes a tf.train.Saver and possibly restores parameters."""
    # TODO(eringrant): Implement saving and restoring for
    # `ExperimentalLearner`s.

    # We omit from saving and restoring any variables that contains as a
    # substring anything in the list `self.omit_from_saving_and_reloading.
    # For example, those that track iterator state.
    logging.info(
        'Omitting from saving / restoring any variable that '
        'contains any of the following substrings: %s',
        self.omit_from_saving_and_reloading)

    def is_not_requested_to_omit(variable_name):
      return all([
          substring not in variable_name
          for substring in self.omit_from_saving_and_reloading
      ])

    # TODO(doersch): the replica_ variables are created by the keras
    # distributed optimizer, and are copies of the optimizer (e.g. Adam)
    # variables.  There's probably a smarter way to avoid saving them.
    var_list = list([
        var for var in tf.global_variables()
        if is_not_requested_to_omit(var.name) and 'replica_' not in var.name and
        '_for_film_learner' not in var.name
    ])

    if self.dataset_classifier_to_restore:
      # If loading a dataset classifier for FLUTE evaluation, the dataset
      # classifier variables won't be in `checkpoint_to_restore`, so omit them
      # from this reloading.
      var_list = [
          v for v in var_list
          if 'dataset_fc' not in v.name and 'set_encoder' not in v.name
      ]

    if var_list:
      self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1200)
    else:
      self.saver = None
      logging.info('Variables not being saved since no variables left after '
                   'filtering.')

    if self.checkpoint_to_restore:
      if not self.saver:
        raise ValueError(
            'Checkpoint not restored, since there is no Saver created. This is '
            'likely due to no parameters being available. If you intend to run '
            'parameterless training, set `checkpoint_to_restore` to None.')

    if self.is_training:

      # To handle pre-emption, we continue from the latest checkpoint if
      # checkpoints already exist in the checkpoint directory.
      latest_checkpoint = None
      if self.checkpoint_dir is not None:
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

      if latest_checkpoint is not None:
        if not self.saver:
          raise ValueError(
              'Checkpoint not restored, since there is no Saver created. This '
              'is likely due to no parameters being available. ')
        restore_or_log_informative_error(self.saver, self.sess,
                                         latest_checkpoint)

      elif self.checkpoint_to_restore:
        logging.info('No training checkpoints found.')
        # For training episodic models from a checkpoint, we restore the
        # backbone weights but omit other (e.g., optimizer) parameters.
        backbone_vars_to_reload = [
            var for var in tf.global_variables()
            if functional_backbones.is_backbone_variable(
                var.name, only_if=is_not_requested_to_omit)
        ]
        backbone_saver = tf.train.Saver(
            var_list=backbone_vars_to_reload, max_to_keep=1)
        restore_or_log_informative_error(backbone_saver, self.sess,
                                         self.checkpoint_to_restore)
        logging.info(
            'Restored only vars %s from provided `checkpoint_to_restore`: %s',
            [var.name for var in backbone_vars_to_reload],
            self.checkpoint_to_restore)

      else:
        logging.info(
            'No checkpoints found; training from random initialization.')

    elif self.checkpoint_to_restore is not None:
      # For evaluation, we restore more than the backbone (embedding function)
      # variables from the provided checkpoint, so we use `self.saver`.
      restore_or_log_informative_error(self.saver, self.sess,
                                       self.checkpoint_to_restore)

      if self.dataset_classifier_to_restore:
        # For FLUTE evaluation, we also need to restore the dataset classifier
        # variables (if using the Blender initialization scheme).
        dataset_classifier_vars = [
            v for v in tf.global_variables()
            if 'set_encoder' in v.name or 'dataset_fc' in v.name
        ]
        dataset_classifier_saver = tf.train.Saver(
            var_list=dataset_classifier_vars, max_to_keep=1)
        restore_or_log_informative_error(dataset_classifier_saver, self.sess,
                                         self.dataset_classifier_to_restore)

    else:
      logging.info(
          'No checkpoints found; evaluating with a random initialization.')

  def get_batch_or_episodic_specification(self, split):
    if split == TRAIN_SPLIT:
      return self._create_train_specification()
    else:
      return self._create_eval_specification(split)

  def _create_train_specification(self):
    """Returns a `BatchSpecification` or `EpisodeSpecification` for training."""
    if (issubclass(self.train_learner_class, learners.EpisodicLearner) or
        self.eval_split == TRAIN_SPLIT):
      return learning_spec.EpisodeSpecification(learning_spec.Split.TRAIN,
                                                self.num_classes_train,
                                                self.num_support_train,
                                                self.num_query_train)
    elif issubclass(self.train_learner_class, learners.BatchLearner):
      return learning_spec.BatchSpecification(learning_spec.Split.TRAIN,
                                              self.batch_size)
    else:
      raise ValueError('The specified `learner_class` should be a subclass of '
                       '`learners.BatchLearner` or `learners.EpisodicLearner`, '
                       'but received {}.'.format(self.train_learner_class))

  def _create_eval_specification(self, split=TEST_SPLIT):
    """Create an `EpisodeSpecification` for episodic evaluation.

    Args:
      split: The split from which to generate the `EpisodeSpecification`.

    Returns:
      An `EpisodeSpecification`.

    Raises:
      ValueError: Invalid `split`.
    """
    if split not in (VALID_SPLIT, TEST_SPLIT):
      raise UnexpectedSplitError(
          split, expected_splits=(VALID_SPLIT, TEST_SPLIT))

    # If using DatasetLearner, we want a BatchSpecification during evaluation.
    if split == VALID_SPLIT and issubclass(self.eval_learner_class,
                                           baseline_learners.DatasetLearner):
      return learning_spec.BatchSpecification(learning_spec.Split.VALID,
                                              self.batch_size)

    split_enum = get_split_enum(split)
    return learning_spec.EpisodeSpecification(split_enum, self.num_classes_eval,
                                              self.num_support_eval,
                                              self.num_query_eval)

  def _restrict_dataset_list_for_split(self, split, splits_to_contribute,
                                       dataset_list):
    """Returns the restricted dataset_list for the given split.

    Args:
      split: A string, either TRAIN_SPLIT, VALID_SPLIT or TEST_SPLIT.
      splits_to_contribute: A list whose length is the number of datasets in the
        benchmark. Each element is a set of strings corresponding to the splits
        that the respective dataset will contribute to.
      dataset_list: A list which has one element per selected dataset (same
        length as splits_to_contribute), e.g. this can be one of the lists
        dataset_spec_list, has_dag_ontology, has_bilevel_ontology of the
        BenchmarkSpecification.
    """
    updated_list = []
    for dataset_num, dataset_splits in enumerate(splits_to_contribute):
      if split in dataset_splits:
        updated_list.append(dataset_list[dataset_num])
    return updated_list

  def get_num_to_take(self, dataset_name, split):
    """Return the number of examples to restrict to for a dataset/split pair."""
    num_to_take = -1  # By default, no restriction.
    if dataset_name in self.restrict_num_per_class:
      dataset_restrict_num_per_class = self.restrict_num_per_class[dataset_name]
      if split in dataset_restrict_num_per_class:
        num_to_take = dataset_restrict_num_per_class[split]
    return num_to_take

  def build_data(self, split):
    """Builds a `tf.Dataset` of episodes or batches from `split`."""
    learner_class = (
        self.train_learner_class
        if split == TRAIN_SPLIT else self.eval_learner_class)
    if (issubclass(learner_class, learners.BatchLearner) and
        split != self.eval_split):
      return self._build_batch(split)
    if issubclass(learner_class, baseline_learners.DatasetLearner):
      return self._build_batch(split)
    elif (issubclass(learner_class, learners.EpisodicLearner) or
          split == self.eval_split):
      if self._fixed_eval == 'vtab':
        return self._build_vtab_episode()
      elif self._fixed_eval == 'dumped':
        return self._build_dumped_episode(split)
      else:
        return self._build_episode(split)
    else:
      raise ValueError('The `Learner` for `split` should be a subclass of '
                       '`learners.BatchLearner` or `learners.EpisodicLearner`, '
                       'but received {}.'.format(learner_class))

  def _build_vtab_episode(self):
    """Build a `tf.Dataset` of vtab episodes."""
    # There is only one dataset.
    # ['vtab_cifar'] -> 'cifar'
    dataset_name = re.search('^vtab_(.+)', self.eval_dataset_list[0]).group(1)
    # Replace back comma.
    dataset_name = dataset_name.replace(':', ',')
    return_vals = read_episodes.read_vtab_as_episode(
        dataset_name,
        self.data_config.image_height,
        query_size_limit=self.data_config.vtab_query_size_limit)
    support_ds, query_ds, n_eval, n_classes = return_vals
    self.vtab_test_classes = n_classes
    logging.info('Using VTAB episode for eval. Dataset: %s, n_eval: %d',
                 dataset_name, n_eval)
    episodes = tf.data.Dataset.zip((support_ds.repeat(), query_ds.repeat()))
    self.num_eval_episodes = n_eval

    def create_episode_struct(support_data, query_data):
      return providers.Episode(
          support_images=support_data['image'],
          query_images=query_data['image'],
          support_labels=support_data['label'],
          query_labels=query_data['label'],
          support_class_ids=support_data['label'],
          query_class_ids=query_data['label'])

    return episodes.map(create_episode_struct)

  def _build_dumped_episode(self, split):
    """Builds a `tf.Dataset` of episodes through reading dumped episodes."""
    dataset_name = re.search('^dumped_(.+)', self.eval_dataset_list[0]).group(1)
    if ('mnist' in dataset_name or 'cifar' in dataset_name or
        'unshuffled_traffic_sign' in dataset_name):
      folder_path = os.path.join(
          self.data_config.eval_extra_dumped_episodes_dir, dataset_name)
    else:
      folder_path = os.path.join(self.data_config.eval_dumped_episodes_dir,
                                 'valid' if split == VALID_SPLIT else 'test',
                                 dataset_name)
    dumped_episode_ds, n_eval = read_episodes.read_episodes_from_records(
        folder_path)
    # If we request less than the available number of episodes, we use that
    # number instead.
    n_eval = min(self.num_eval_episodes, n_eval)
    logging.info('Using dumped episode for eval. Dataset: %s, n_eval: %d',
                 dataset_name, n_eval)
    self.num_eval_episodes = n_eval
    map_fn = functools.partial(
        pipeline.process_dumped_episode,
        image_size=self.data_config.image_height)
    dataset = dumped_episode_ds.map(map_fn)

    # Overlap episode processing and training.
    data_pipeline = dataset.prefetch(1)
    data_pipeline = apply_dataset_options(data_pipeline)

    def create_episode_struct(support_images, support_labels, support_class_ids,
                              query_images, query_labels, query_class_ids):
      return providers.Episode(
          support_images=support_images,
          query_images=query_images,
          support_labels=support_labels,
          query_labels=query_labels,
          support_class_ids=support_class_ids,
          query_class_ids=query_class_ids)

    return data_pipeline.map(create_episode_struct)


  def _build_episode(self, split):
    """Builds a `tf.Dataset` containing Episodes for "split".

    Args:
      split: A string, either TRAIN_SPLIT, VALID_SPLIT, or TEST_SPLIT.

    Returns:
      An `tf.Dataset` with Episodes.

    Raises:
      UnexpectedSplitError: If split not as expected for this episode build.
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    num_prefetch = self.data_config.num_prefetch
    (_, image_shape, dataset_spec_list, has_dag_ontology, has_bilevel_ontology,
     splits_to_contribute) = self.benchmark_spec

    # Choose only the datasets that are chosen to contribute to the given split.
    dataset_spec_list = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, dataset_spec_list)
    has_dag_ontology = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, has_dag_ontology)
    has_bilevel_ontology = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, has_bilevel_ontology)

    episode_spec = self.split_episode_or_batch_specs[split]
    dataset_split = episode_spec[0]
    # TODO(lamblinp): Support non-square shapes if necessary. For now, all
    # images are resized to square, even if it changes the aspect ratio.
    image_size = image_shape[0]
    if image_shape[1] != image_size:
      raise ValueError(
          'Expected a square image shape, not {}'.format(image_shape))

    if split == TRAIN_SPLIT:
      episode_descr_config = self.train_episode_config
    elif split in (VALID_SPLIT, TEST_SPLIT):
      episode_descr_config = self.eval_episode_config
    else:
      raise UnexpectedSplitError(split)

    # Decide how many examples per class to restrict to for each dataset for the
    # given split (by default there is no restriction).
    num_per_class = []  # A list whose length is the number of datasets.
    for dataset_spec in dataset_spec_list:
      num_per_class.append(self.get_num_to_take(dataset_spec.name, split))

    if split == TRAIN_SPLIT:
      # The learner for the training split should only be in training mode if
      # the evaluation split is not the training split.
      gin_scope_name = ('train'
                        if self.eval_split != TRAIN_SPLIT else 'evaluation')
    else:
      gin_scope_name = 'evaluation'

    ignore_hierarchy_prob = episode_descr_config.ignore_hierarchy_probability
    simclr_episode_fraction = episode_descr_config.simclr_episode_fraction
    if simclr_episode_fraction > 0:
      assert not self.enable_tf_optimizations, (
          'Must set enable_tf_optimizations=False or SimCLR will fail; see '
          'https://github.com/tensorflow/tensorflow/issues/22145')
    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:
      use_dag_ontology = has_dag_ontology[0]
      if self.eval_finegrainedness or self.eval_imbalance_dataset:
        use_dag_ontology = False

      with gin.config_scope(gin_scope_name):
        data_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec_list[0],
            use_dag_ontology=use_dag_ontology,
            use_bilevel_ontology=has_bilevel_ontology[0],
            split=dataset_split,
            episode_descr_config=episode_descr_config,
            shuffle_buffer_size=shuffle_buffer_size,
            read_buffer_size_bytes=read_buffer_size_bytes,
            num_prefetch=num_prefetch,
            image_size=image_size,
            num_to_take=num_per_class[0],
            simclr_episode_fraction=simclr_episode_fraction,
            ignore_hierarchy_probability=ignore_hierarchy_prob)

    else:
      if ignore_hierarchy_prob > 0.0:
        raise ValueError(
            'ignore_hierarchy_probability not supported with multisource pipelines'
        )
      with gin.config_scope(gin_scope_name):
        data_pipeline = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list,
            use_dag_ontology_list=has_dag_ontology,
            use_bilevel_ontology_list=has_bilevel_ontology,
            split=dataset_split,
            episode_descr_config=episode_descr_config,
            shuffle_buffer_size=shuffle_buffer_size,
            read_buffer_size_bytes=read_buffer_size_bytes,
            num_prefetch=num_prefetch,
            image_size=image_size,
            num_to_take=num_per_class,
            simclr_episode_fraction=simclr_episode_fraction)

    data_pipeline = apply_dataset_options(data_pipeline)

    def create_episode_struct(data, dataset_index):
      (support_images, support_labels, support_class_ids, query_images,
       query_labels, query_class_ids) = data
      episode = providers.Episode(
          support_images=support_images,
          query_images=query_images,
          support_labels=support_labels,
          query_labels=query_labels,
          support_class_ids=support_class_ids,
          query_class_ids=query_class_ids)
      if self.needs_source[split]:
        return (episode, dataset_index)
      else:
        return episode

    return data_pipeline.map(create_episode_struct)

  def _build_batch(self, split):
    """Builds a `tf.Dataset` of Batch objects containing data for "split".

    Args:
      split: A string, either TRAIN_SPLIT, VALID_SPLIT, or TEST_SPLIT.

    Returns:
      A `tf.Dataset` containing Batches
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    num_prefetch = self.data_config.num_prefetch
    (_, image_shape, dataset_spec_list, _, _,
     splits_to_contribute) = self.benchmark_spec

    # Choose only the datasets that are chosen to contribute to the given split.
    dataset_spec_list = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, dataset_spec_list)

    # Decide how many examples per class to restrict to for each dataset for the
    # given split (by default there is no restriction).
    num_per_class = []  # A list whose length is the number of datasets.
    for dataset_spec in dataset_spec_list:
      num_per_class.append(self.get_num_to_take(dataset_spec.name, split))

    dataset_split, batch_size = self.split_episode_or_batch_specs[split]
    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:
      data_pipeline = pipeline.make_one_source_batch_pipeline(
          dataset_spec_list[0],
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_shape[0],
          num_to_take=num_per_class[0])
    else:
      data_pipeline = pipeline.make_multisource_batch_pipeline(
          dataset_spec_list,
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_shape[0],
          num_to_take=num_per_class)

    data_pipeline = apply_dataset_options(data_pipeline)

    def create_batch_structure(data_pipeline, imagenet_pipeline):
      data, dataset_index = data_pipeline
      if self.sample_half_from_imagenet:
        imagenet_data, imagenet_index = imagenet_pipeline
        data, dataset_index = tf.cond(
            tf.greater_equal(tf.random.uniform([], minval=0., maxval=1.), 0.5),
            true_fn=lambda: (data, dataset_index),
            false_fn=lambda: (imagenet_data, imagenet_index))
      (images, class_ids) = data

      # The number of available classes for each dataset
      all_n_classes = [
          len(dataset_spec.get_classes(get_split_enum(split)))
          for dataset_spec in dataset_spec_list
      ]
      if len(dataset_spec_list) == 1:
        n_classes = all_n_classes[0]
      elif gin.query_parameter('BatchSplitReaderGetReader.add_dataset_offset'):
        # The total number of classes is the sum for all datasets
        n_classes = sum(all_n_classes)
      else:
        # The number of classes is the one of the current dataset
        n_classes = tf.convert_to_tensor(all_n_classes)[dataset_index]
      batch = providers.Batch(
          images=images, labels=class_ids, n_classes=n_classes)
      if self.needs_source[split]:
        return (batch, dataset_index)
      else:
        return batch

    if self.sample_half_from_imagenet:
      data_pipeline_imagenet = pipeline.make_one_source_batch_pipeline(
          dataset_spec_list[0],
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_shape[0],
          num_to_take=num_per_class[0])
      data_pipeline = tf.data.Dataset.zip(
          (data_pipeline, data_pipeline_imagenet))
    else:
      # Zip a dummy "ImageNet pipeline" that won't be used.
      data_pipeline = tf.data.Dataset.zip(
          (data_pipeline, tf.data.Dataset.from_tensors(0).repeat()))
    return data_pipeline.map(create_batch_structure)

  def get_run_fn_with_train_op(self, run_fn, regularizer_fn, global_step):
    """Returns the operation that performs a training update."""

    def run_fn_with_train_op(data):
      """Run and train the model."""
      res = run_fn(data)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # note: every worker computes the same loss.  This is because the
      # reduce_mean needs to be computed globally.
      loss = distribute_utils.aggregate(res['loss'])
      loss = tf.reduce_mean(loss)
      loss += regularizer_fn()

      replica_ctx = tf.distribute.get_replica_context()
      if replica_ctx:
        loss /= replica_ctx.num_replicas_in_sync
      # TODO(doersch): there's probably a better way to do EMA updates. EMAs
      # are MirroredVariables, which means the assign needs to happen on every
      # replica.  I think what's happening is that the update from replica 1
      # is getting run on every replica for a different copy of the mirrored
      # variable.  Only running the assign ops from the first replica doesn't
      # work.
      with tf.control_dependencies([tf.group(update_ops)]):

        if self.normalized_gradient_descent:
          if self.meta_batch_size != 1:
            raise NotImplementedError('Setting `meta_batch_size` != 1 is not '
                                      'yet supported when using normalized '
                                      'gradient descent.')
          opt_vars = tf.trainable_variables()
          if self.distribute:
            grads = self.optimizer.get_gradients(loss, opt_vars)
            grads_and_vars = list(zip(grads, opt_vars))
          else:
            grads_and_vars = self.optimizer.compute_gradients(loss, opt_vars)
          global_norm = 0
          # We reverse the order of grads_and_vars because they're computed in
          # reverse order; this way, the network can begin aggregating the
          # global norm before the backwards pass is complete.
          for g, v in grads_and_vars[::-1]:
            if replica_ctx:
              g = replica_ctx.all_reduce('sum', g)
            sumsq = tf.reduce_sum(tf.square(g))
            global_norm += sumsq
          nrm = tf.sqrt(tf.maximum(global_norm, 1e-5))

          grads_and_vars2 = []
          for g, v in grads_and_vars:
            grads_and_vars2.append((g / nrm, v))

          train_op = self.optimizer.apply_gradients(grads_and_vars2)

          with tf.control_dependencies([train_op]):
            train_op = tf.assign(global_step, global_step + 1)
        else:
          if self.meta_batch_size > 1:
            grads_and_vars = self.optimizer.compute_gradients(
                loss, var_list=tf.trainable_variables())
            avg_grads_and_vars = []
            self._grad_placeholders = []
            self._grad_op = []
            for grad, var in grads_and_vars:
              grad_ph = tf.placeholder(grad.dtype, grad.shape)
              self._grad_placeholders.append(grad_ph)
              avg_grads_and_vars.append((grad_ph, var))
              self._grad_op.append(grad)
            train_op = self.optimizer.apply_gradients(avg_grads_and_vars,
                                                      global_step)
          else:
            train_op = self.optimizer.minimize(
                loss,
                global_step=global_step,
                var_list=tf.trainable_variables())

      res['train_op'] = train_op
      return res

    return run_fn_with_train_op

  def get_updated_global_step(self):
    with tf.control_dependencies([self.train_op]):
      global_step = tf.identity(tf.train.get_global_step())
    return global_step

  def train(self):
    """The training loop."""
    global_step = self.sess.run(tf.train.get_global_step())
    logging.info('Starting training from global_step: %d', global_step)
    updated_global_step = self.get_updated_global_step()

    should_save = self.checkpoint_dir is not None
    if should_save and global_step == 0:
      # Save the initialization weights.
      save_path = self.saver.save(
          self.sess, os.path.join(self.checkpoint_dir, 'model_0.ckpt'))
      logging.info('Model initialization saved: %s', save_path)

    # Compute the initial validation performance before starting the training,
    # unless train() has already been called on this object.
    if np.isnan([self.valid_acc, self.valid_ci]).any():
      self.maybe_evaluate(global_step)

    while global_step < self.num_updates:
      if self.num_gpus > self.meta_batch_size:
        tf.logging.info('Requested more GPUs (`num_gpus` = %d) than the '
                        'meta-batch size (`meta_batch_size` = %d), so %d '
                        'GPUs will be unused.' %
                        (self.num_gpus, self.meta_batch_size,
                         self.num_gpus - self.meta_batch_size))
        self.num_gpus = self.meta_batch_size

      # Perform the next update.
      if self.meta_batch_size > 1:
        train_loss, train_acc, train_grads = [], [], []
        n_per_gpu = int(self.meta_batch_size / self.num_gpus)
        remainder = self.meta_batch_size % self.num_gpus
        devices = []
        for i in range(self.num_gpus):
          extra = remainder if not i else 0
          devices.extend(['/gpu:{}'.format(i)] * (n_per_gpu + extra))
        assert len(devices) == self.meta_batch_size, (
            'The `devices` list must contain the same '
            'number of elements as '
            '`self.meta_batch_size`.')
        for _, d in zip(range(self.meta_batch_size), devices):
          with tf.device(d):
            train_loss_, train_acc_, train_grads_ = self.sess.run([
                self.losses[TRAIN_SPLIT], self.accuracies[TRAIN_SPLIT],
                self._grad_op
            ])
          train_loss.append(train_loss_)
          train_acc.append(train_acc_)
          train_grads.append(train_grads_)
        # Average the gradients and perform a single model update.
        feed_dict = {}
        for i, ph in enumerate(self._grad_placeholders):
          feed_dict[ph] = np.stack([g[i] for g in train_grads],
                                   axis=0).mean(axis=0)
        _, global_step = self.sess.run([self.train_op, updated_global_step],
                                       feed_dict=feed_dict)
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
      else:
        (_, train_loss, train_acc, global_step) = self.sess.run([
            self.train_op, self.losses[TRAIN_SPLIT],
            self.accuracies[TRAIN_SPLIT], updated_global_step
        ])
      train_acc = np.mean(train_acc)

      # Maybe validate, depending on the global step's value.
      self.maybe_evaluate(global_step)

      # Log training progress.
      if not global_step % self.log_every:
        message = (
            'Update %d. Train loss: %f, Train accuracy: %f, '
            'Valid accuracy %f +/- %f.\n' %
            (global_step, train_loss, train_acc, self.valid_acc, self.valid_ci))
        logging.info(message)

        # Update summaries.
        if self.summary_writer:
          summaries = self.sess.run(self.standard_summaries)
          self.summary_writer.add_summary(summaries, global_step)

      if should_save and global_step % self.checkpoint_every == 0:
        save_path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, 'model_%d.ckpt' % global_step))
        logging.info('Model checkpoint saved: %s', save_path)

  def maybe_evaluate(self, global_step):
    """Maybe perform evaluation, depending on the value of global_step."""
    if not global_step % self.validate_every:
      # Get the validation accuracy and confidence interval.
      (valid_acc, valid_ci, valid_acc_summary,
       valid_ci_summary) = self.evaluate(
           VALID_SPLIT, step=global_step)
      # Validation summaries are updated every time validation happens which is
      # every validate_every steps instead of log_every steps.
      if self.summary_writer:
        self.summary_writer.add_summary(valid_acc_summary, global_step)
        self.summary_writer.add_summary(valid_ci_summary, global_step)
      self.valid_acc = valid_acc
      self.valid_ci = valid_ci

  # TODO(evcu) Improve this so that if the eval_only loads a global_step, it is
  # used at logging instead of value 0.
  def evaluate(self, split, step=0):
    """Returns performance metrics across num_eval_trials episodes / batches."""
    num_eval_trials = self.num_eval_episodes
    logging.info('Performing evaluation of the %s split using %d episodes...',
                 split, num_eval_trials)
    accuracies = []
    total_samples = 0
    for eval_trial_num in range(num_eval_trials):
      # Following is used to normalize accuracies.
      acc, summaries = self.sess.run(
          [self.accuracies[split], self.evaluation_summaries])
      # Write complete summaries during evaluation, but not training.
      # Otherwise, validation summaries become too big.
      if not self.is_training and self.summary_writer:
        self.summary_writer.add_summary(summaries, eval_trial_num)
      if self._fixed_eval == 'vtab':
        accuracies.append(np.sum(acc))
        total_samples += np.size(acc)
        continue
      accuracies.append(np.mean(acc))
      total_samples += 1

    logging.info('Finished evaluation.')

    mean_acc = np.sum(accuracies) / total_samples
    ci_acc = np.std(accuracies) * 1.96 / np.sqrt(len(accuracies))  # confidence

    # VTAB evaluation has 1 episode.
    if self._fixed_eval == 'vtab':
      ci_acc = 0

    if not self.is_training:
      # Logging during training is handled by self.train() instead.
      logging.info('Accuracy on the meta-%s split: %f, +/- %f.\n', split,
                   mean_acc, ci_acc)

    with tf.name_scope('trainer_metrics'):
      with tf.name_scope(split):
        mean_acc_summary = tf.Summary()
        mean_acc_summary.value.add(tag='mean acc', simple_value=mean_acc)
        ci_acc_summary = tf.Summary()
        ci_acc_summary.value.add(tag='acc CI', simple_value=ci_acc)

    return mean_acc, ci_acc, mean_acc_summary, ci_acc_summary

  def add_eval_summaries(self):
    """Returns summaries of way / shot / classes/ logits / targets."""
    evaluation_summaries = [
        tf.summary.scalar('global_step', tf.train.get_global_step())
    ]
    for split in self.required_splits:
      evaluation_summaries.extend(self._add_eval_summaries_split(split))
    return evaluation_summaries

  def _add_eval_summaries_split(self, split):
    """Returns split's summaries of way / shot / classes / logits / targets."""
    split_eval_summaries = []
    episode_info = copy.copy(self.episode_info[split])
    episode_info['query_logits'] = self.predictions[split]
    summary_labels = [
        'way', 'shots', 'class_ids', 'query_logits', 'query_targets'
    ]
    if self.eval_imbalance_dataset:
      summary_labels += ['class_props']
    for label in summary_labels:
      if episode_info[label] is not None:
        if episode_info[label].shape:
          summary_fn = tf.summary.tensor_summary
        else:
          summary_fn = tf.summary.scalar
        summary = summary_fn('%s_%s' % (split, label), episode_info[label])
        split_eval_summaries.append(summary)
    return split_eval_summaries

  def _get_logit_dim(self,
                     split,
                     is_batch_learner,
                     is_training,
                     return_list=False):
    """Returns the total number of logits needed.

    Args:
      split: string, one of TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT.
      is_batch_learner: bool, if True the logit count is obtained from dataset
        spec. If False, `max_ways` is used for episodic dataset.
      is_training: bool, used to decide number of logits.
      return_list: bool, whether to return a list of numbers of classes for the
        different datasets, instead of returning only the overall sum. Only
        applies if is_batch_learner is True.

    Returns:
      int, total number of logits needed.
    """
    if self._fixed_eval == 'vtab':
      return self.vtab_test_classes
    if is_batch_learner:
      # Get the total number of classes in this split, across all datasets
      # contributing to this split.
      total_classes = []
      for dataset_spec, dataset_splits in zip(
          self.benchmark_spec.dataset_spec_list,
          self.benchmark_spec.splits_to_contribute):
        if any(
            get_split_enum(ds_split) == split for ds_split in dataset_splits):
          total_classes.append(len(dataset_spec.get_classes(split)))
      if not return_list:
        total_classes = sum(total_classes)
    else:
      if return_list:
        raise ValueError('Option `return_list` is not applicable to episodic '
                         'learners.')
      total_classes = (
          self.train_episode_config.max_ways
          if is_training else self.eval_episode_config.max_ways)
    return total_classes
