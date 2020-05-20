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
"""Abstract learners."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import logging
import gin.tf
from meta_dataset.models import functional_backbones
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

# TODO(eringrant): Create a single entry point for importing `Learner`s but
# keep the `Learner`s in different files; the import statement should remain
# something like `from meta_dataset.learners import PrototypicalNetwork`.


@gin.configurable
class Learner(object):
  """A Learner."""

  def __init__(
      self,
      is_training,
      logit_dim,
      transductive_batch_norm,
      backprop_through_moments,
      embedding_fn,
      weight_decay,
  ):
    """Initializes a Learner.

    Note that Gin configuration of subclasses of `Learner` will override any
    corresponding Gin configurations of `Learner`, since parameters are passed
    to the `Learner` base class's constructor (See
    https://github.com/google/gin-config/blob/master/README.md) for more
    details).

    Args:
      is_training: Whether the learning is in training mode.
      logit_dim: An integer; the maximum dimensionality of output predictions.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      embedding_fn: A string; the name of the function that embeds images.
      weight_decay: coefficient for L2 regularization.
    """
    self.is_training = is_training
    self.logit_dim = logit_dim
    self.transductive_batch_norm = transductive_batch_norm
    self.backprop_through_moments = backprop_through_moments
    self.embedding_fn = functional_backbones.NAME_TO_EMBEDDING_NETWORK[
        embedding_fn]
    self.weight_decay = weight_decay

    if self.transductive_batch_norm:
      logging.info('Using transductive batch norm!')

  def compute_loss(self, onehot_labels, predictions):
    """Computes the CE loss of `predictions` with respect to `onehot_labels`.

    Args:
      onehot_labels: A `tf.Tensor` containing the the class labels; each vector
        along the class dimension should hold a valid probability distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` representing the average loss.
    """
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=predictions)
    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + self.weight_decay * regularization
    return loss

  def compute_accuracy(self, labels, predictions):
    """Computes the accuracy of `predictions` with respect to `labels`.

    Args:
      labels: A `tf.Tensor` containing the the class labels; each vector along
        the class dimension should hold a valid probability distribution.
      predictions: A `tf.Tensor` containing the the class predictions,
        interpreted as unnormalized log probabilities.

    Returns:
       A `tf.Tensor` representing the average accuracy.
    """
    correct = tf.equal(labels, tf.to_int32(tf.argmax(predictions, -1)))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

  def forward_pass(self, data):
    """Returns the (query if episodic) logits."""
    raise NotImplementedError('Abstract method.')


class EpisodicLearner(Learner):
  """An episodic learner."""

  pass


class BatchLearner(Learner):
  """A batch learner."""

  pass


def linear_classifier_forward_pass(embeddings, w_fc, b_fc, cosine_classifier,
                                   cosine_logits_multiplier, use_weight_norm):
  """Passes embeddings through the linear layer defined by w_fc and b_fc.

  Args:
    embeddings: A Tensor of size [batch size, embedding dim].
    w_fc: A Tensor of size [embedding dim, num outputs].
    b_fc: Either None, or a Tensor of size [num outputs] or []. If
      cosine_classifier is False, it can not be None.
    cosine_classifier: A bool. If true, a cosine classifier is used which does
      not require the bias b_fc.
    cosine_logits_multiplier: A float. Only used if cosine_classifier is True,
      and multiplies the resulting logits.
    use_weight_norm: A bool. Whether weight norm was used. If so, then if using
      cosine classifier, normalize only the embeddings but not the weights.

  Returns:
    logits: A Tensor of size [batch size, num outputs].
  """
  if cosine_classifier:
    # Each column of the weight matrix may be interpreted as a class
    # representation (of the same dimenionality as the embedding space). The
    # logit for an embedding vector belonging to that class is the cosine
    # similarity between that embedding and that class representation.
    embeddings = tf.nn.l2_normalize(embeddings, axis=1, epsilon=1e-3)
    if not use_weight_norm:
      # Only normalize the weights if weight norm was not used.
      w_fc = tf.nn.l2_normalize(w_fc, axis=0, epsilon=1e-3)
    logits = tf.matmul(embeddings, w_fc)
    # Scale the logits as passing numbers in [-1, 1] to softmax is not very
    # expressive.
    logits *= cosine_logits_multiplier
  else:
    assert b_fc is not None
    logits = tf.matmul(embeddings, w_fc) + b_fc
  return logits


def linear_classifier_logits(embeddings, num_classes, cosine_classifier,
                             cosine_logits_multiplier, use_weight_norm):
  """Forward pass through a linear classifier, possibly a cosine classifier."""

  # A variable to keep track of whether the initialization has already happened.
  data_dependent_init_done = tf.get_variable(
      'data_dependent_init_done',
      initializer=0,
      dtype=tf.int32,
      trainable=False)

  embedding_dims = embeddings.get_shape().as_list()[-1]

  if use_weight_norm:
    w_fc = tf.get_variable(
        'w_fc', [embedding_dims, num_classes],
        initializer=tf.random_normal_initializer(0, 0.05),
        trainable=True)
    # This init is temporary as it needs to be done in a data-dependent way.
    # It will be overwritten during the first forward pass through this layer.
    g = tf.get_variable(
        'g',
        dtype=tf.float32,
        initializer=tf.ones([num_classes]),
        trainable=True)
    b_fc = None
    if not cosine_classifier:
      # Also initialize a bias.
      b_fc = tf.get_variable(
          'b_fc', initializer=tf.zeros([num_classes]), trainable=True)

    def _do_data_dependent_init():
      """Returns ops for the data-dependent init of g and maybe b_fc."""
      w_fc_normalized = tf.nn.l2_normalize(w_fc.read_value(), [0])
      output_init = tf.matmul(embeddings, w_fc_normalized)
      mean_init, var_init = tf.nn.moments(output_init, [0])
      # Data-dependent init values.
      g_init_value = 1. / tf.sqrt(var_init + 1e-10)
      ops = [tf.assign(g, g_init_value)]
      if not cosine_classifier:
        # Also initialize a bias in a data-dependent way.
        b_fc_init_value = -mean_init * g_init_value
        ops.append(tf.assign(b_fc, b_fc_init_value))
      # Mark that the data-dependent initialization is done to prevent it from
      # happening again in the future.
      ops.append(tf.assign(data_dependent_init_done, 1))
      return tf.group(*ops)

    # Possibly perform data-dependent init (if it hasn't been done already).
    init_op = tf.cond(
        tf.equal(data_dependent_init_done, 0), _do_data_dependent_init,
        tf.no_op)

    with tf.control_dependencies([init_op]):
      # Apply weight normalization.
      w_fc *= g / tf.sqrt(tf.reduce_sum(tf.square(w_fc), [0]))
      # Forward pass through the layer defined by w_fc and b_fc.
      logits = linear_classifier_forward_pass(embeddings, w_fc, b_fc,
                                              cosine_classifier,
                                              cosine_logits_multiplier, True)

  else:
    # No weight norm.
    w_fc = functional_backbones.weight_variable([embedding_dims, num_classes])
    b_fc = None
    if not cosine_classifier:
      # Also initialize a bias.
      b_fc = functional_backbones.bias_variable([num_classes])
    # Forward pass through the layer defined by w_fc and b_fc.
    logits = linear_classifier_forward_pass(embeddings, w_fc, b_fc,
                                            cosine_classifier,
                                            cosine_logits_multiplier, False)
  return logits


# TODO(tylerzhu): Consider adding an episodic kNN learner as well so we can
# create a baseline leaner by composing a batch learner and the evaluation
# process of an episodic kNN learner.
@gin.configurable
class BaselineLearner(BatchLearner):
  """A Baseline Network."""

  # TODO(eringrant): Remove this attribute when the `BaselineFinetuneLearner`
  # subclass is refactored to obey the interface of `Learner.compute_logits`.
  obeys_compute_logits_interface = True

  def __init__(self, knn_in_fc, knn_distance, cosine_classifier,
               cosine_logits_multiplier, use_weight_norm, **kwargs):
    """Initializes a baseline learner.

    Args:
      knn_in_fc: Whether kNN is performed in the space of fc activations or
        embeddings. If True, the logits from the last fc layer are used as the
        embedding on which kNN lookup is performed. Otherwise, the penultimate
        layer is what the kNN lookup is performed on.
      knn_distance: The distance measurement used by kNN lookup. 'l2', 'cosine'
      cosine_classifier: A bool. Whether to use a cosine classifier at training
        time when performing the all-way classification task to train the
        backbone.
      cosine_logits_multiplier: A float. A scalar that will multiply the logits
        computed by the cosine classifier (if applicable) before passing them
        into the softmax.
      use_weight_norm: A bool. Whether to apply weight normalization to the
        linear classifier layer.
      **kwargs: Keyword arguments common to all BatchLearners.
    """
    self.knn_in_fc = knn_in_fc
    self.distance = knn_distance
    self.cosine_classifier = cosine_classifier
    self.cosine_logits_multiplier = cosine_logits_multiplier
    self.use_weight_norm = use_weight_norm
    logging.info(
        'BaselineLearner: '
        'distance %s, '
        'cosine_classifier: %s'
        'knn_in_fc %s', knn_distance, cosine_classifier, knn_in_fc)

    super(BaselineLearner, self).__init__(**kwargs)

  def forward_pass(self, data):
    if self.is_training:
      images = data.images
      embeddings_params_moments = self.embedding_fn(images, self.is_training)
      support_embeddings = embeddings_params_moments['embeddings']
      support_logits = self.forward_pass_fc(support_embeddings)
      return support_logits
    else:
      support_embeddings_params_moments = self.embedding_fn(
          data.support_images, self.is_training)
      support_embeddings = support_embeddings_params_moments['embeddings']
      support_set_moments = None
      if not self.transductive_batch_norm:
        support_set_moments = support_embeddings_params_moments['moments']
      query_embeddings = self.embedding_fn(
          data.query_images,
          self.is_training,
          moments=support_set_moments,
          backprop_through_moments=self.backprop_through_moments)
      query_embeddings = query_embeddings['embeddings']

      # TODO(eringrant): The `BaselineFinetuneLearner` subclass is not yet
      # refactored to obey the interface of `Learner.compute_logits`.
      if self.obeys_compute_logits_interface:
        query_logits = self.compute_logits(support_embeddings, query_embeddings,
                                           data.onehot_support_labels)
      else:
        query_logits = self.compute_logits(data)  # pylint: disable=no-value-for-parameter

      return query_logits

  def forward_pass_fc(self, embeddings):
    """Passes the provided embeddings through the fc layer to get the logits.

    Args:
      embeddings: A Tensor of the penultimate layer activations as computed by
        BaselineLearner.forward_pass.

    Returns:
      The fc layer activations.
    """
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      # Always maps to a space whose dimensionality is the number of classes
      # at meta-training time.
      logits = linear_classifier_logits(embeddings, self.logit_dim,
                                        self.cosine_classifier,
                                        self.cosine_logits_multiplier,
                                        self.use_weight_norm)
      return logits

  def compute_logits(self, support_embeddings, query_embeddings,
                     onehot_support_labels):
    """Computes the class logits for the episode.

    Args:
      support_embeddings: A Tensor of size [num_support_images, embedding dim].
      query_embeddings: A Tensor of size [num_query_images, embedding dim].
      onehot_support_labels: A Tensor of size [batch size, way].

    Returns:
      The query set logits as a [num_query_images, way] matrix.

    Raises:
      ValueError: Distance must be one of l2 or cosine.
    """

    if self.knn_in_fc:
      # Recompute the support and query embeddings that were originally computed
      # in self.forward_pass() to be the fc layer activations.
      support_embeddings = self.forward_pass_fc(support_embeddings)
      query_embeddings = self.forward_pass_fc(query_embeddings)

    # ------------------------ K-NN look up -------------------------------
    # For each testing example in an episode, we use its embedding
    # vector to look for the closest neighbor in all the training examples'
    # embeddings from the same episode and then assign the training example's
    # class label to the testing example as the predicted class label for it.
    if self.distance == 'l2':
      #  [1, num_support, embed_dims]
      support_embeddings = tf.expand_dims(support_embeddings, axis=0)
      #  [num_query, 1, embed_dims]
      query_embeddings = tf.expand_dims(query_embeddings, axis=1)
      #  [num_query, num_support]
      distance = tf.norm(query_embeddings - support_embeddings, axis=2)
    elif self.distance == 'cosine':
      support_embeddings = tf.nn.l2_normalize(support_embeddings, axis=1)
      query_embeddings = tf.nn.l2_normalize(query_embeddings, axis=1)
      distance = -1 * tf.matmul(
          query_embeddings, support_embeddings, transpose_b=True)
    else:
      raise ValueError('Distance must be one of l2 or cosine.')
    #  [num_query]
    _, indices = tf.nn.top_k(-distance, k=1)
    indices = tf.squeeze(indices, axis=1)
    #  [num_query, num_classes]
    query_logits = tf.gather(onehot_support_labels, indices)
    return query_logits
