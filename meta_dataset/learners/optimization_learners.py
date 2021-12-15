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
"""Optimization-based learners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging
import gin.tf
from meta_dataset.learners import base as learner_base
from meta_dataset.learners import baseline_learners
from meta_dataset.learners import metric_learners
from meta_dataset.models import functional_backbones
from meta_dataset.models import functional_classifiers
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


def get_embeddings_vars_copy_ops(embedding_vars_dict, make_copies):
  """Gets copies of the embedding variables or returns those variables.

  This is useful at meta-test time for MAML and the finetuning baseline. In
  particular, at meta-test time, we don't want to make permanent updates to
  the model's variables, but only modifications that persist in the given
  episode. This can be achieved by creating copies of each variable and
  modifying and using these copies instead of the variables themselves.

  Args:
    embedding_vars_dict: A dict mapping each variable name to the corresponding
      Variable.
    make_copies: A bool. Whether to copy the given variables. If not, those
      variables themselves will be returned. Typically, this is True at meta-
      test time and False at meta-training time.

  Returns:
    embedding_vars_keys: A list of variable names.
    embeddings_vars: A corresponding list of Variables.
    embedding_vars_copy_ops: A (possibly empty) list of operations, each of
      which assigns the value of one of the provided Variables to a new
      Variable which is its copy.
  """
  embedding_vars_keys = []
  embedding_vars = []
  embedding_vars_copy_ops = []
  for name, var in six.iteritems(embedding_vars_dict):
    embedding_vars_keys.append(name)
    if make_copies:
      with tf.variable_scope('weight_copy'):
        shape = var.shape.as_list()
        var_copy = tf.Variable(
            tf.zeros(shape), collections=[tf.GraphKeys.LOCAL_VARIABLES])
        var_copy_op = tf.assign(var_copy, var)
        embedding_vars_copy_ops.append(var_copy_op)
      embedding_vars.append(var_copy)
    else:
      embedding_vars.append(var)
  return embedding_vars_keys, embedding_vars, embedding_vars_copy_ops


def get_fc_vars_copy_ops(fc_weights, fc_bias, make_copies):
  """Gets copies of the classifier layer variables or returns those variables.

  At meta-test time, a copy is created for the given Variables, and these copies
  copies will be used in place of the original ones.

  Args:
    fc_weights: A Variable for the weights of the fc layer.
    fc_bias: A Variable for the bias of the fc layer.
    make_copies: A bool. Whether to copy the given variables. If not, those
      variables themselves are returned.

  Returns:
    fc_weights: A Variable for the weights of the fc layer. Might be the same as
      the input fc_weights or a copy of it.
    fc_bias: Analogously, a Variable for the bias of the fc layer.
    fc_vars_copy_ops: A (possibly empty) list of operations for assigning the
      value of each of fc_weights and fc_bias to a respective copy variable.
  """
  fc_vars_copy_ops = []
  if make_copies:
    with tf.variable_scope('weight_copy'):
      # fc_weights copy
      fc_weights_copy = tf.Variable(
          tf.zeros(fc_weights.shape.as_list()),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      fc_weights_copy_op = tf.assign(fc_weights_copy, fc_weights)
      fc_vars_copy_ops.append(fc_weights_copy_op)

      # fc_bias copy
      fc_bias_copy = tf.Variable(
          tf.zeros(fc_bias.shape.as_list()),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      fc_bias_copy_op = tf.assign(fc_bias_copy, fc_bias)
      fc_vars_copy_ops.append(fc_bias_copy_op)

      fc_weights = fc_weights_copy
      fc_bias = fc_bias_copy
  return fc_weights, fc_bias, fc_vars_copy_ops


def gradient_descent_step(loss,
                          variables,
                          stop_grads,
                          allow_grads_to_batch_norm_vars,
                          learning_rate,
                          get_update_ops=True):
  """Returns the updated vars after one step of gradient descent."""
  grads = tf.gradients(loss, variables)

  if stop_grads:
    grads = [tf.stop_gradient(dv) for dv in grads]

  def _apply_grads(variables, grads):
    """Applies gradients using SGD on a list of variables."""
    v_new, update_ops = [], []
    for (v, dv) in zip(variables, grads):
      if (not allow_grads_to_batch_norm_vars and
          ('offset' in v.name or 'scale' in v.name)):
        updated_value = v  # no update.
      else:
        updated_value = v - learning_rate * dv  # gradient descent update.
        if get_update_ops:
          update_ops.append(tf.assign(v, updated_value))
      v_new.append(updated_value)
    return v_new, update_ops

  updated_vars, update_ops = _apply_grads(variables, grads)
  return {'updated_vars': updated_vars, 'update_ops': update_ops}


@gin.configurable
class BaselineFinetuneLearner(baseline_learners.BaselineLearner):
  """A Baseline Network with test-time finetuning."""

  # TODO(eringrant): Remove this attribute when the `BaselineFinetuneLearner`
  # subclass is refactored to obey the interface of `Learner.compute_logits`.
  obeys_compute_logits_interface = False

  def __init__(self,
               num_finetune_steps,
               finetune_lr,
               debug_log=False,
               finetune_all_layers=False,
               finetune_with_adam=False,
               **kwargs):
    """Initializes a baseline learner.

    Args:
      num_finetune_steps: number of finetune steps.
      finetune_lr: the learning rate used for finetuning.
      debug_log: If True, print out debug logs.
      finetune_all_layers: Whether to finetune all embedding variables. If
        False, only trains a linear classifier on top of the embedding.
      finetune_with_adam: Whether to use Adam for the within-episode finetuning.
        If False, gradient descent is used instead.
      **kwargs: Keyword arguments common to all `BaselineLearner`s (including
        `knn_in_fc` and `knn_distance`, which are not used by
        `BaselineFinetuneLearner` but are used by the parent class).
    """
    self.num_finetune_steps = num_finetune_steps
    self.finetune_lr = finetune_lr
    self.debug_log = debug_log
    self.finetune_all_layers = finetune_all_layers
    self.finetune_with_adam = finetune_with_adam
    if finetune_with_adam:
      self.finetune_opt = tf.train.AdamOptimizer(self.finetune_lr)
    super(BaselineFinetuneLearner, self).__init__(**kwargs)

  def compute_logits(self, data):
    """Computes the class logits for the episode.

    Args:
      data: A `meta_dataset.providers.Episode`.

    Returns:
      The query set logits as a [num_query_images, way] matrix.

    Raises:
      ValueError: Distance must be one of l2 or cosine.
    """
    # ------------------------ Finetuning -------------------------------
    # Possibly make copies of embedding variables, if they will get modified.
    # This is for making temporary-only updates to the embedding network
    # which will not persist after the end of the episode.
    make_copies = self.finetune_all_layers

    # TODO(eringrant): Reduce the number of times the embedding function graph
    # is built with the same input.
    support_embeddings_params_moments = self.embedding_fn(
        data.support_images, self.is_training)
    support_embeddings = support_embeddings_params_moments['embeddings']
    support_embeddings_var_dict = support_embeddings_params_moments['params']

    (embedding_vars_keys, embedding_vars,
     embedding_vars_copy_ops) = get_embeddings_vars_copy_ops(
         support_embeddings_var_dict, make_copies)
    embedding_vars_copy_op = tf.group(*embedding_vars_copy_ops)

    # Compute the initial training loss (only for printing purposes). This
    # line is also needed for adding the fc variables to the graph so that the
    # tf.all_variables() line below detects them.
    logits = self._fc_layer(support_embeddings)[:, 0:data.way]
    finetune_loss = self.compute_loss(
        onehot_labels=data.onehot_support_labels,
        predictions=logits,
    )

    # Decide which variables to finetune.
    fc_vars, vars_to_finetune = [], []
    for var in tf.trainable_variables():
      if 'fc_finetune' in var.name:
        fc_vars.append(var)
        vars_to_finetune.append(var)
    if self.finetune_all_layers:
      vars_to_finetune.extend(embedding_vars)
    logging.info('Finetuning will optimize variables: %s', vars_to_finetune)

    for i in range(self.num_finetune_steps):
      if i == 0:
        # Randomly initialize the fc layer.
        fc_reset = tf.variables_initializer(var_list=fc_vars)
        # Adam related variables are created when minimize() is called.
        # We create an unused op here to put all adam varariables under
        # the 'adam_opt' namescope and create a reset op to reinitialize
        # these variables before the first finetune step.
        adam_reset = tf.no_op()
        if self.finetune_with_adam:
          with tf.variable_scope('adam_opt'):
            unused_op = self.finetune_opt.minimize(
                finetune_loss, var_list=vars_to_finetune)
          adam_reset = tf.variables_initializer(self.finetune_opt.variables())
        with tf.control_dependencies(
            [fc_reset, adam_reset, finetune_loss, embedding_vars_copy_op] +
            vars_to_finetune):
          print_op = tf.no_op()
          if self.debug_log:
            print_op = tf.print([
                'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                finetune_loss
            ])

          with tf.control_dependencies([print_op]):
            # Get the operation for finetuning.
            # (The logits and loss are returned just for printing).
            logits, finetune_loss, finetune_op = self._get_finetune_op(
                data, embedding_vars_keys, embedding_vars, vars_to_finetune,
                support_embeddings if not self.finetune_all_layers else None)

            if self.debug_log:
              # Test logits are computed only for printing logs.
              query_embeddings = self.embedding_fn(
                  data.query_images,
                  self.is_training,
                  params=collections.OrderedDict(
                      zip(embedding_vars_keys, embedding_vars)),
                  reuse=True)['embeddings']
              query_logits = (self._fc_layer(query_embeddings)[:, 0:data.way])

      else:
        with tf.control_dependencies([finetune_op, finetune_loss] +
                                     vars_to_finetune):
          print_op = tf.no_op()
          if self.debug_log:
            print_op = tf.print([
                'step: %d' % i,
                vars_to_finetune[0][0, 0],
                'loss:',
                finetune_loss,
                'accuracy:',
                self.compute_accuracy(
                    labels=data.onehot_support_labels, predictions=logits),
                'query accuracy:',
                self.compute_accuracy(
                    labels=data.onehot_query_labels, predictions=query_logits),
            ])

          with tf.control_dependencies([print_op]):
            # Get the operation for finetuning.
            # (The logits and loss are returned just for printing).
            logits, finetune_loss, finetune_op = self._get_finetune_op(
                data, embedding_vars_keys, embedding_vars, vars_to_finetune,
                support_embeddings if not self.finetune_all_layers else None)

            if self.debug_log:
              # Test logits are computed only for printing logs.
              query_embeddings = self.embedding_fn(
                  data.query_images,
                  self.is_training,
                  params=collections.OrderedDict(
                      zip(embedding_vars_keys, embedding_vars)),
                  reuse=True)['embeddings']
              query_logits = (self._fc_layer(query_embeddings)[:, 0:data.way])

    # Finetuning is now over, compute the query performance using the updated
    # fc layer, and possibly the updated embedding network.
    with tf.control_dependencies([finetune_op] + vars_to_finetune):
      query_embeddings = self.embedding_fn(
          data.query_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, embedding_vars)),
          reuse=True)['embeddings']
      query_logits = self._fc_layer(query_embeddings)[:, 0:data.way]

      if self.debug_log:
        # The train logits are computed only for printing.
        support_embeddings = self.embedding_fn(
            data.support_images,
            self.is_training,
            params=collections.OrderedDict(
                zip(embedding_vars_keys, embedding_vars)),
            reuse=True)['embeddings']
        logits = self._fc_layer(support_embeddings)[:, 0:data.way]

      print_op = tf.no_op()
      if self.debug_log:
        print_op = tf.print([
            'accuracy:',
            self.compute_accuracy(
                labels=data.onehot_support_labels, predictions=logits),
            'query accuracy:',
            self.compute_accuracy(
                labels=data.onehot_query_labels, predictions=query_logits),
        ])
      with tf.control_dependencies([print_op]):
        query_logits = self._fc_layer(query_embeddings)[:, 0:data.way]

    return query_logits

  def _get_finetune_op(self,
                       data,
                       embedding_vars_keys,
                       embedding_vars,
                       vars_to_finetune,
                       support_embeddings=None):
    """Returns the operation for performing a finetuning step."""
    if support_embeddings is None:
      support_embeddings = self.embedding_fn(
          data.support_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, embedding_vars)),
          reuse=True)['embeddings']
    logits = self._fc_layer(support_embeddings)[:, 0:data.way]
    finetune_loss = self.compute_loss(
        onehot_labels=data.onehot_support_labels,
        predictions=logits,
    )
    # Perform one step of finetuning.
    if self.finetune_with_adam:
      finetune_op = self.finetune_opt.minimize(
          finetune_loss, var_list=vars_to_finetune)
    else:
      # Apply vanilla gradient descent instead of Adam.
      update_ops = gradient_descent_step(finetune_loss, vars_to_finetune, True,
                                         False, self.finetune_lr)['update_ops']
      finetune_op = tf.group(*update_ops)
    return logits, finetune_loss, finetune_op

  def _fc_layer(self, embedding):
    """The fully connected layer to be finetuned."""
    with tf.variable_scope('fc_finetune', reuse=tf.AUTO_REUSE):
      logits = functional_classifiers.linear_classifier(
          embedding, self.logit_dim, self.cosine_classifier,
          self.cosine_logits_multiplier, self.use_weight_norm)
    return logits


@gin.configurable
class OptimizationLearner(learner_base.EpisodicLearner):
  pass


@gin.configurable
class MAMLLearner(OptimizationLearner):
  """Model-Agnostic Meta Learner."""

  def __init__(self,
               num_update_steps,
               additional_evaluation_update_steps,
               first_order,
               alpha,
               adapt_batch_norm,
               zero_fc_layer,
               proto_maml_fc_layer_init,
               classifier_weight_decay,
               debug=False,
               **kwargs):
    """Initializes a baseline learner.

    Args:
      num_update_steps: The number of inner-loop steps to take.
      additional_evaluation_update_steps: The number of additional inner-loop
        steps to take on meta test and meta validation set.
      first_order: If True, ignore second-order gradients (faster).
      alpha: The inner-loop learning rate.
      adapt_batch_norm: If True, adapt batch norm parameters in the inner loop.
      zero_fc_layer: Whether to use zero fc layer initialization.
      proto_maml_fc_layer_init: Whether to use ProtoNets equivalent fc layer
        initialization.
      classifier_weight_decay: Scalar weight decay coefficient for
        regularization of the linear classifier layer.
      debug: If True, print out debug logs.
      **kwargs: Keyword arguments common to all `OptimizationLearner`.
    """
    self.alpha = alpha
    self.num_update_steps = num_update_steps
    self.additional_evaluation_update_steps = additional_evaluation_update_steps
    self.first_order = first_order
    self.adapt_batch_norm = adapt_batch_norm
    self.debug_log = debug
    self.zero_fc_layer = zero_fc_layer
    self.proto_maml_fc_layer_init = proto_maml_fc_layer_init
    self.classifier_weight_decay = classifier_weight_decay
    super(MAMLLearner, self).__init__(**kwargs)

  def proto_maml_fc_weights(self, prototypes, zero_pad_to_max_way=False):
    """Computes the Prototypical MAML fc layer's weights.

    Args:
      prototypes: Tensor of shape [num_classes, embedding_size]
      zero_pad_to_max_way: Whether to zero padd to max num way.

    Returns:
      fc_weights: Tensor of shape [embedding_size, num_classes] or
        [embedding_size, self.logit_dim] when zero_pad_to_max_way is True.
    """
    fc_weights = 2 * prototypes
    fc_weights = tf.transpose(fc_weights)
    if zero_pad_to_max_way:
      paddings = [[0, 0], [0, self.logit_dim - tf.shape(fc_weights)[1]]]
      fc_weights = tf.pad(fc_weights, paddings, 'CONSTANT', constant_values=0)
    return fc_weights

  def proto_maml_fc_bias(self, prototypes, zero_pad_to_max_way=False):
    """Computes the Prototypical MAML fc layer's bias.

    Args:
      prototypes: Tensor of shape [num_classes, embedding_size]
      zero_pad_to_max_way: Whether to zero padd to max num way.

    Returns:
      fc_bias: Tensor of shape [num_classes] or [self.logit_dim]
        when zero_pad_to_max_way is True.
    """
    fc_bias = -tf.square(tf.norm(prototypes, axis=1))
    if zero_pad_to_max_way:
      paddings = [[0, self.logit_dim - tf.shape(fc_bias)[0]]]
      fc_bias = tf.pad(fc_bias, paddings, 'CONSTANT', constant_values=0)
    return fc_bias

  def forward_pass(self, data):
    """Computes the test logits of MAML.

    Args:
      data: A `meta_dataset.providers.Episode` containing the data for the
        episode.

    Returns:
      The output logits for the query data in this episode.
    """
    # Have to use one-hot labels since sparse softmax doesn't allow
    # second derivatives.
    support_embeddings_ = self.embedding_fn(
        data.support_images, self.is_training, reuse=tf.AUTO_REUSE)
    support_embeddings = support_embeddings_['embeddings']
    embedding_vars_dict = support_embeddings_['params']

    # TODO(eringrant): Refactor to make use of
    # `functional_backbones.linear_classifier`, which allows Gin-configuration.
    with tf.variable_scope('linear_classifier', reuse=tf.AUTO_REUSE):
      embedding_depth = support_embeddings.shape.as_list()[-1]
      fc_weights = functional_backbones.weight_variable(
          [embedding_depth, self.logit_dim],
          weight_decay=self.classifier_weight_decay)
      fc_bias = functional_backbones.bias_variable([self.logit_dim])

    # A list of variable names, a list of corresponding Variables, and a list
    # of operations (possibly empty) that creates a copy of each Variable.
    (embedding_vars_keys, embedding_vars,
     embedding_vars_copy_ops) = get_embeddings_vars_copy_ops(
         embedding_vars_dict, make_copies=not self.is_training)

    # A Variable for the weights of the fc layer, a Variable for the bias of the
    # fc layer, and a list of operations (possibly empty) that copies them.
    (fc_weights, fc_bias, fc_vars_copy_ops) = get_fc_vars_copy_ops(
        fc_weights, fc_bias, make_copies=not self.is_training)

    fc_vars = [fc_weights, fc_bias]
    num_embedding_vars = len(embedding_vars)
    num_fc_vars = len(fc_vars)

    def _cond(step, *args):
      del args
      num_steps = self.num_update_steps
      if not self.is_training:
        num_steps += self.additional_evaluation_update_steps
      return step < num_steps

    def _body(step, *args):
      """The inner update loop body."""
      updated_embedding_vars = args[0:num_embedding_vars]
      updated_fc_vars = args[num_embedding_vars:num_embedding_vars +
                             num_fc_vars]
      support_embeddings = self.embedding_fn(
          data.support_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          reuse=True)['embeddings']

      updated_fc_weights, updated_fc_bias = updated_fc_vars
      support_logits = tf.matmul(support_embeddings,
                                 updated_fc_weights) + updated_fc_bias

      support_logits = support_logits[:, 0:data.way]
      loss = tf.losses.softmax_cross_entropy(data.onehot_support_labels,
                                             support_logits)

      print_op = tf.no_op()
      if self.debug_log:
        print_op = tf.print(['step: ', step, updated_fc_bias[0], 'loss:', loss])

      with tf.control_dependencies([print_op]):
        updated_embedding_vars = gradient_descent_step(
            loss, updated_embedding_vars, self.first_order,
            self.adapt_batch_norm, self.alpha, False)['updated_vars']
        updated_fc_vars = gradient_descent_step(loss, updated_fc_vars,
                                                self.first_order,
                                                self.adapt_batch_norm,
                                                self.alpha,
                                                False)['updated_vars']

        step = step + 1
      return tuple([step] + list(updated_embedding_vars) +
                   list(updated_fc_vars))

    # MAML meta updates using query set examples from an episode.
    if self.zero_fc_layer:
      # To account for variable class sizes, we initialize the output
      # weights to zero. See if truncated normal initialization will help.
      zero_weights_op = tf.assign(fc_weights, tf.zeros_like(fc_weights))
      zero_bias_op = tf.assign(fc_bias, tf.zeros_like(fc_bias))
      fc_vars_init_ops = [zero_weights_op, zero_bias_op]
    else:
      fc_vars_init_ops = fc_vars_copy_ops

    if self.proto_maml_fc_layer_init:
      support_embeddings = self.embedding_fn(
          data.support_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, embedding_vars)),
          reuse=True)['embeddings']

      prototypes = metric_learners.compute_prototypes(
          support_embeddings, data.onehot_support_labels)
      pmaml_fc_weights = self.proto_maml_fc_weights(
          prototypes, zero_pad_to_max_way=True)
      pmaml_fc_bias = self.proto_maml_fc_bias(
          prototypes, zero_pad_to_max_way=True)
      fc_vars = [pmaml_fc_weights, pmaml_fc_bias]

    # These control dependencies assign the value of each variable to a new copy
    # variable that corresponds to it. This is required at test time for
    # initilizing the copies as they are used in place of the original vars.
    with tf.control_dependencies(fc_vars_init_ops + embedding_vars_copy_ops):
      # Make step a local variable as we don't want to save and restore it.
      step = tf.Variable(
          0,
          trainable=False,
          name='inner_step_counter',
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      loop_vars = [step] + embedding_vars + fc_vars
      step_and_all_updated_vars = tf.while_loop(
          _cond, _body, loop_vars, swap_memory=True)
      step = step_and_all_updated_vars[0]
      all_updated_vars = step_and_all_updated_vars[1:]
      updated_embedding_vars = all_updated_vars[0:num_embedding_vars]
      updated_fc_weights, updated_fc_bias = all_updated_vars[
          num_embedding_vars:num_embedding_vars + num_fc_vars]

    # Forward pass the training images with the updated weights in order to
    # compute the means and variances, to use for the query's batch norm.
    support_set_moments = None
    if not self.transductive_batch_norm:
      support_set_moments = self.embedding_fn(
          data.support_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(embedding_vars_keys, updated_embedding_vars)),
          reuse=True)['moments']

    query_embeddings = self.embedding_fn(
        data.query_images,
        self.is_training,
        params=collections.OrderedDict(
            zip(embedding_vars_keys, updated_embedding_vars)),
        moments=support_set_moments,  # Use support set stats for batch norm.
        reuse=True,
        backprop_through_moments=self.backprop_through_moments)['embeddings']

    query_logits = (tf.matmul(query_embeddings, updated_fc_weights) +
                    updated_fc_bias)[:, 0:data.way]

    return query_logits


@gin.configurable
class FLUTEFiLMLearner(learner_base.EpisodicLearner):
  """A Learner that trains a new set of FiLM params for each evaluation task."""

  def __init__(self, num_steps, lr, film_init, debug_log=False, **kwargs):
    """Initializes a FLUTEFiLMLearner.

    Args:
      num_steps: The number of steps with which the FiLM parameters will be
        fine-tuned.
      lr: The learning rate for the fine-tuning of the FiLM parameters.
      film_init: The method with which to initialize the FiLM parameters of the
        new task. The valid options are 'scratch', 'imagenet', 'blender', or
        'hard blender'. The last two use a dataset classifier to assess the
        compatibility between the given support set and each training dataset,
        and then either take a convex combination of the trained sets of FiLM
        parameters accordingly ('blender'), or use only the FiLM parameters of
        the most likely dataset ('hard belnder').
      debug_log: Whether to print additional information for debugging.
      **kwargs: Additional kwargs for the parent Learner class.
    """
    if film_init not in ['scratch', 'imagenet', 'blender', 'blender_hard']:
      raise ValueError('Unknown FiLM parameter init scheme.')
    tf.logging.info(
        'Initializing a FiLM Learner with {} steps and lr {}'.format(
            num_steps, lr))
    self.num_steps = num_steps
    self.film_init = film_init
    self.debug_log = debug_log
    self.lr = lr
    self.opt = tf.train.AdamOptimizer(lr)
    super(FLUTEFiLMLearner, self).__init__(**kwargs)
    delattr(self, 'logit_dim')

  def forward_pass(self, data):
    """Computes the query logits for the given episode `data`."""

    if self.film_init == 'scratch':
      self.film_selector = None
    elif self.film_init == 'imagenet':
      # Note: this makes the assumption that the first set of learned FiLM
      # parameters corresponds to the ImageNet dataset. Otherwise, the
      # following line should be changed appropriately.
      self.film_selector = 0
    elif self.film_init in ['blender', 'blender_hard']:
      dataset_logits = functional_backbones.dataset_classifier(
          data.support_images)
      if self.film_init == 'blender_hard':
        # Select only the argmax entry.
        self.film_selector = tf.one_hot(
            tf.math.argmax(dataset_logits, axis=-1),
            depth=tf.shape(dataset_logits)[1])
      else:
        # Take a convex combination.
        self.film_selector = tf.nn.softmax(dataset_logits, axis=-1)

    if self.num_steps:
      # Initial forward pass, required for the `unused_op` below and for placing
      # variables in tf.trainable_variables() for the below block to pick up.
      loss = self._compute_losses(data, compute_on_query=False)['loss']

      # Pick out the variables to optimize.
      self.opt_vars = []
      for var in tf.trainable_variables():
        if '_for_film_learner' in var.name:
          self.opt_vars.append(var)
      tf.logging.info('FiLMLearner will optimize vars: {}'.format(
          self.opt_vars))

    for i in range(self.num_steps):
      if i == 0:
        # Re-initialize the variables to optimize for the new episode, to ensure
        # the FiLM parameters aren't re-used across tasks of a given dataset.
        vars_reset = tf.variables_initializer(var_list=self.opt_vars)
        # Adam related variables are created when minimize() is called.
        # We create an unused op here to put all adam varariables under
        # the 'adam_opt' namescope and create a reset op to reinitialize
        # these variables before the first finetune step.
        with tf.variable_scope('adam_opt', reuse=tf.AUTO_REUSE):
          unused_op = self.opt.minimize(loss, var_list=self.opt_vars)
        adam_reset = tf.variables_initializer(self.opt.variables())

        with tf.control_dependencies([vars_reset, adam_reset, loss] +
                                     self.opt_vars):
          print_op = tf.no_op()
          if self.debug_log:
            print_op = tf.print(
                ['step: %d' % i, self.opt_vars[0][0], 'loss:', loss],
                summarize=-1)

          with tf.control_dependencies([print_op]):
            # Get the train op.
            results = self._get_train_op(data)
            (train_op, loss, query_loss, acc,
             query_acc) = (results['train_op'], results['loss'],
                           results['query_loss'], results['acc'],
                           results['query_acc'])

      else:
        with tf.control_dependencies([train_op, loss, acc] + self.opt_vars +
                                     [query_loss, query_acc] *
                                     int(self.debug_log)):

          print_op = tf.no_op()
          if self.debug_log:
            print_list = [
                '################',
                'step: %d' % i,
                self.opt_vars[0][0],
                'support loss:',
                loss,
                'query loss:',
                query_loss,
                'support acc:',
                acc,
                'query acc:',
                query_acc,
            ]
            print_op = tf.print(print_list)

          with tf.control_dependencies([print_op]):
            # Get the train op (the loss is returned just for printing).
            results = self._get_train_op(data)
            (train_op, loss, query_loss, acc,
             query_acc) = (results['train_op'], results['loss'],
                           results['query_loss'], results['acc'],
                           results['query_acc'])

    # Training is now over, compute the final query logits.
    dependency_list = [] if not self.num_steps else [train_op] + self.opt_vars
    with tf.control_dependencies(dependency_list):
      results = self._compute_losses(data, compute_on_query=True)
      (loss, query_loss, query_logits, acc,
       query_acc) = (results['loss'], results['query_loss'],
                     results['query_logits'], results['acc'],
                     results['query_acc'])

      print_op = tf.no_op()
      if self.debug_log:
        print_op = tf.print([
            'Done training',
            'support loss:',
            loss,
            'query loss:',
            query_loss,
            'support acc:',
            acc,
            'query acc:',
            query_acc,
        ])
      with tf.control_dependencies([print_op]):
        query_logits = tf.identity(query_logits)

    return query_logits

  def _forward_pass(self, images, params=None, moments=None):
    """Returns the result of the forward pass through the embedding network."""
    return self.embedding_fn(
        images,
        self.is_training,
        params=params,
        moments=moments,
        film_selector=self.film_selector)

  def _compute_prototype_loss(self,
                              embeddings,
                              labels,
                              labels_one_hot,
                              prototypes=None):
    """Computes the loss and accuracy on an episode."""
    labels_dense = labels
    if prototypes is None:
      # Compute protos.
      labels = tf.cast(labels_one_hot, tf.float32)
      # [num examples, 1, embedding size].
      embeddings_ = tf.expand_dims(embeddings, 1)
      # [num examples, num classes, 1].
      labels = tf.expand_dims(labels, 2)
      # Sums each class' embeddings. [num classes, embedding size].
      class_sums = tf.reduce_sum(labels * embeddings_, 0)
      # The prototype of each class is the averaged embedding of its examples.
      class_num_images = tf.reduce_sum(labels, 0)  # [way].
      prototypes = class_sums / class_num_images  # [way, embedding size].

    # Compute logits.
    embeddings = tf.nn.l2_normalize(embeddings, 1, epsilon=1e-3)
    prototypes = tf.nn.l2_normalize(prototypes, 1, epsilon=1e-3)
    logits = tf.matmul(embeddings, prototypes, transpose_b=True)

    loss = self.compute_loss(labels_one_hot, logits)
    acc = tf.reduce_mean(self.compute_accuracy(labels_dense, logits))
    return loss, acc, prototypes, logits

  def _compute_losses(self, data, compute_on_query=False):
    """Computes the nearest-centroid loss and accuracy."""
    support_dict = self._forward_pass(data.support_images)
    support_embeddings = support_dict['embeddings']
    loss, acc, prototypes, _ = self._compute_prototype_loss(
        support_embeddings, data.support_labels, data.onehot_support_labels)

    query_logits, query_acc, query_loss = None, None, None
    if compute_on_query:
      query_embeddings = self._forward_pass(
          data.query_images,
          params=support_dict['params'],
          moments=support_dict['moments'])['embeddings']
      query_loss, query_acc, _, query_logits = self._compute_prototype_loss(
          query_embeddings,
          data.query_labels,
          data.onehot_query_labels,
          prototypes=prototypes)
    return {
        'loss': loss,
        'query_loss': query_loss,
        'query_logits': query_logits,
        'acc': acc,
        'query_acc': query_acc,
    }

  def _get_train_op(self, data):
    """Returns the loss and the operation for performing a train step."""
    results = self._compute_losses(data, compute_on_query=True)

    grads_and_vars = self.opt.compute_gradients(
        results['loss'], var_list=self.opt_vars)
    train_op = self.opt.apply_gradients(grads_and_vars)
    results['train_op'] = train_op

    return results

