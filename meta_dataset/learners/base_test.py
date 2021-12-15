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
"""Abstract tests for subclasses inheriting from meta_dataset.learners.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from meta_dataset.learners.experimental import base as experimental_learner_base
from meta_dataset.models import functional_backbones
from meta_dataset.models.experimental import reparameterizable_backbones
import numpy as np
import tensorflow.compat.v1 as tf

IMAGE_HEIGHT = IMAGE_WIDTH = 84
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
SUPPORT_SIZE, QUERY_SIZE = 5, 15
NUM_CLASSES = 5

# Initialization arguments shared between non-experimental and experimental
# learners.
VALID_ABSTRACT_LEARNER_INIT_ARGS = {
    'is_training': True,
    'logit_dim': NUM_CLASSES,
    'transductive_batch_norm': True,
    'backprop_through_moments': True,
    'input_shape': IMAGE_SHAPE,
}

VALID_LEARNER_INIT_ARGS = {
    **VALID_ABSTRACT_LEARNER_INIT_ARGS, 'embedding_fn':
        functools.partial(
            functional_backbones.four_layer_convnet, weight_decay=0.01)
}

# TODO(eringrant): Right now some `Learner` (`MatchingNetworkLearner`)
# implementations NAN out if there are no labels of a class index in an episode.
# Decide on expected behavior for these cases and implement a test for that
# behavior.

# TODO(eringrant): Test feature (embeddding function-less) representations.


class MockEmbedding(reparameterizable_backbones.ConvNet):

  def __init__(self, keep_spatial_dims=False):
    super().__init__(
        output_dim=None,
        keep_spatial_dims=keep_spatial_dims,
        num_filters_per_layer=(64, 64, 64, 64),
    )


class MockEpisode(
    collections.namedtuple(
        'MockEpisode', 'support_images, query_images, '
        'support_labels, query_labels')):

  @property
  def way(self):
    return NUM_CLASSES

  @property
  def onehot_support_labels(self):
    return tf.one_hot(self.support_labels, NUM_CLASSES)

  @property
  def onehot_query_labels(self):
    return tf.one_hot(self.query_labels, NUM_CLASSES)

  @property
  def labels(self):
    return self.query_labels

  @property
  def onehot_labels(self):
    return self.onehot_query_labels


class MockBatch(collections.namedtuple('MockBatch', 'images, labels')):

  @property
  def way(self):
    return NUM_CLASSES

  @property
  def onehot_labels(self):
    return tf.one_hot(self.labels, NUM_CLASSES)


class TestLearner(tf.test.TestCase):
  convergence_test_iterations = 500

  @property
  def learner_cls(self):
    raise NotImplementedError('The test subclass must provide a `Learner`.')

  @property
  def learner_kwargs(self):
    raise NotImplementedError(
        'The test subclass must provide keyword arguments to initialize a '
        '`Learner`.')

  def set_up_learner(self):
    learner_kwargs = self.learner_kwargs
    if issubclass(self.learner_cls,
                  experimental_learner_base.ExperimentalLearner):
      learner_kwargs['embedding_fn'] = MockEmbedding()
    data = self.random_data()
    learner = self.learner_cls(**learner_kwargs)
    learner.build()
    return data, learner

  def testForwardPass(self):
    """Assert that the learner obeys the API for `forward_pass`."""
    data, learner = self.set_up_learner()
    outputs = learner.forward_pass(data)
    self.assertEqual(len(outputs.get_shape().as_list()), 2)
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(outputs)

  def testComputeLoss(self):
    """Assert that the learner obeys the API for `compute_loss`."""
    data, learner = self.set_up_learner()
    loss = learner.compute_loss(
        data.onehot_labels,
        tf.cast(data.onehot_labels, tf.float32),
    )
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(loss)

  def testComputeAccuracy(self):
    """Assert that the learner obeys the API for `compute_accuracy`."""
    data, learner = self.set_up_learner()
    accuracy = learner.compute_accuracy(
        data.onehot_labels,
        tf.cast(data.onehot_labels, tf.float32),
    )
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(accuracy)

  def testLearnerInitRandomAccuracy(self):
    """Assert that the learner's performance wrt accuracy is initially random."""
    data, learner = self.set_up_learner()
    outputs = learner.forward_pass(data)
    accuracy = learner.compute_accuracy(data.onehot_labels, outputs)
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      accuracy_value = self.evaluate(accuracy).mean()
      epsilon = 0.08  # Allow 8% deviation from random.
      self.assertLess(accuracy_value, 1. / data.way + epsilon)

  def testLearnerImprovement(self):
    """Assert that the learner's objective monotonically improves."""
    data, learner = self.set_up_learner()
    # Small learning rate for improvement check.
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    outputs = learner.forward_pass(data)
    loss = learner.compute_loss(data.onehot_labels, outputs)
    train_op = optimizer.minimize(loss)
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      loss_value_prev = np.inf
      # Allow three steps for gradient descent to stabilize.
      for _ in range(3):
        _, loss_value = self.evaluate((train_op, loss))
      self.assertLess(loss_value.mean(), loss_value_prev)

  def testLearnerConvergence(self):
    """Assert that the unregularized learner overfits a single batch."""
    data, learner = self.set_up_learner()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    outputs = learner.forward_pass(data)
    loss = learner.compute_loss(data.onehot_labels, outputs)
    train_op = optimizer.minimize(loss)
    with self.session():
      self.evaluate(tf.compat.v1.local_variables_initializer())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      for _ in range(self.convergence_test_iterations):
        _, loss_value = self.evaluate((train_op, loss))
      # TODO(eringrant): Parameterize this convergence check value per
      # `Learner`; 10.0 is too high a loss value for most `Learner`s.
      self.assertLess(loss_value.mean(), 10.0)


class TestBatchLearner(TestLearner):

  def random_data(self):
    return MockBatch(
        tf.cast(
            np.random.uniform(size=(NUM_CLASSES * QUERY_SIZE, *IMAGE_SHAPE)),
            tf.float32),
        tf.cast(
            np.random.permutation(list(np.arange(NUM_CLASSES)) * QUERY_SIZE),
            tf.int32),
    )


class TestEpisodicLearner(TestLearner):

  def random_data(self):

    return MockEpisode(
        tf.cast(
            np.random.uniform(size=(NUM_CLASSES * SUPPORT_SIZE, *IMAGE_SHAPE)),
            tf.float32),
        tf.cast(
            np.random.uniform(size=(NUM_CLASSES * QUERY_SIZE, *IMAGE_SHAPE)),
            tf.float32),
        tf.cast(
            np.random.permutation(list(np.arange(NUM_CLASSES)) * SUPPORT_SIZE),
            tf.int32),
        tf.cast(
            np.random.permutation(list(np.arange(NUM_CLASSES)) * QUERY_SIZE),
            tf.int32),
    )
