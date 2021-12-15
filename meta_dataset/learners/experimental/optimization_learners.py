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
"""Optimization-based learners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import gin.tf

from meta_dataset.learners.experimental import base as learner_base
from meta_dataset.models.experimental import reparameterizable_backbones
from meta_dataset.models.experimental import reparameterizable_base
from meta_dataset.models.experimental import reparameterizable_distributions
from six.moves import zip
import tensorflow as tf


@gin.configurable
def sgd(learning_rate):
  """Construct optimizer triple for stochastic gradient descent (SGD).

  Inspired by the optimizer definitions in JAX
  (https://github.com/google/jax/blob/main/jax/experimental/optimizers.py),
  this implementation of SGD is fully functional (i.e., it maintains no hidden
  state) and so is compatible for use with an optimization-based meta-learner.

  Args:
    learning_rate: A positive scalar.

  Returns:
    An (init, update, get_params) function triple.
  """

  def init(x0):
    return x0

  def update(i, grad, state):
    del i
    x = state
    return x - learning_rate * grad

  def get_params(state):
    x = state
    return x

  return init, update, get_params


@gin.configurable
def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Inspired by the optimizer definitions in JAX
  (https://github.com/google/jax/blob/main/jax/experimental/optimizers.py),
  this implementation of Adam is fully functional (i.e., it maintains no hidden
  state) and so is compatible for use with an optimization-based meta-learner.

  Args:
    learning_rate: A positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init, update, get_params) function triple.
  """

  def init(x0):
    m0 = tf.zeros_like(x0)
    v0 = tf.zeros_like(x0)
    return x0, m0, v0

  def update(i, grad, state):
    i = tf.cast(i, dtype=tf.float32)
    x, m, v = state
    m = (1. - b1) * grad + b1 * m  # First  moment estimate.
    v = (1. - b2) * (grad**2.) + b2 * v  # Second moment estimate.
    mhat = m / (1. - b1**(i + 1.))  # Bias correction.
    vhat = v / (1. - b2**(i + 1.))
    x = x - learning_rate * mhat / (tf.sqrt(vhat) + eps)
    return x, m, v

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params


def optimizer_update(iterate_collection, iteration_idx, objective_fn, update_fn,
                     get_params_fn, first_order, clip_grad_norm):
  """Returns the next iterate in the optimization of objective_fn wrt variables.

  Args:
    iterate_collection: A (potentially structured) container of tf.Tensors
      corresponding to the state of the current iterate.
    iteration_idx: An int Tensor; the iteration number.
    objective_fn: Callable that takes in variables and produces the value of the
      objective function.
    update_fn: Callable that takes in the gradient of the objective function and
      the current iterate and produces the next iterate.
    get_params_fn: Callable that takes in the gradient of the objective function
      and the current iterate and produces the next iterate.
    first_order: If True, prevent the computation of higher order gradients.
    clip_grad_norm: If not None, gradient dimensions are independently clipped
      to lie in the interval [-clip_grad_norm, clip_grad_norm].
  """
  variables = [get_params_fn(iterate) for iterate in iterate_collection]

  if tf.executing_eagerly():
    with tf.GradientTape(persistent=True) as g:
      g.watch(variables)
      loss = objective_fn(variables, iteration_idx)
    grads = g.gradient(loss, variables)
  else:
    loss = objective_fn(variables, iteration_idx)
    grads = tf.gradients(ys=loss, xs=variables)

  if clip_grad_norm:
    grads = [
        tf.clip_by_value(grad, -1 * clip_grad_norm, clip_grad_norm)
        for grad in grads
    ]

  if first_order:
    grads = [tf.stop_gradient(dv) for dv in grads]

  return [
      update_fn(i=iteration_idx, grad=dv, state=s)
      for (s, dv) in zip(iterate_collection, grads)
  ]


def em_loop(
    num_updates,
    e_step,
    m_step,
    variables,
):
  """Expectation-maximization of objective_fn wrt variables for num_updates."""

  def _body(step, preupdate_vars):
    train_predictions_, responsibilities_ = e_step(preupdate_vars)
    updated_vars = m_step(preupdate_vars, train_predictions_, responsibilities_)
    return step + 1, updated_vars

  def _cond(step, *args):
    del args
    return step < num_updates

  step = tf.Variable(0, trainable=False, name='inner_step_counter')
  loop_vars = (step, variables)
  step, updated_vars = tf.while_loop(
      cond=_cond, body=_body, loop_vars=loop_vars, swap_memory=True)

  return updated_vars


@gin.configurable
def optimizer_loop(
    num_updates,
    objective_fn,
    update_fn,
    variables,
    first_order,
    clip_grad_norm,
):
  """Optimization of `objective_fn` for `num_updates` of `variables`."""

  # Optimizer specifics.
  init, update, get_params = update_fn()

  def _body(step, preupdate_vars):
    """Optimization loop body."""
    updated_vars = optimizer_update(
        iterate_collection=preupdate_vars,
        iteration_idx=step,
        objective_fn=objective_fn,
        update_fn=update,
        get_params_fn=get_params,
        first_order=first_order,
        clip_grad_norm=clip_grad_norm,
    )

    return step + 1, updated_vars

  def _cond(step, *args):
    """Optimization truncation condition."""
    del args
    return step < num_updates

  step = tf.Variable(0, trainable=False, name='inner_step_counter')
  loop_vars = (step, [init(var) for var in variables])
  step, updated_vars = tf.while_loop(
      cond=_cond, body=_body, loop_vars=loop_vars, swap_memory=True)

  return [get_params(v) for v in updated_vars]


ForwardPass = collections.namedtuple('ForwardPass', (
    'embeddings',
    'predictions',
    'inner_objective_value',
    'outer_objective_value',
    'accuracy',
))

Adaptation = collections.namedtuple('Adaptation', (
    'pre_adaptation_support_results',
    'post_adaptation_support_results',
    'pre_adaptation_query_results',
    'post_adaptation_query_results',
    'objective_fn',
    'support_module_objective_fn',
    'query_module_objective_fn',
    'forward_pass_fn',
    'init_loop_variables_mapping',
    'final_loop_variables_mapping',
))


@gin.configurable
class ExperimentalOptimizationLearner(learner_base.ExperimentalEpisodicLearner):
  """An optimization-based learner."""

  def __init__(self, adapt_embedding_predicate, num_update_steps,
               additional_evaluation_update_steps, first_order,
               adapt_batch_norm, clip_grad_norm, update_fn, **kwargs):
    """Initializes a `ExperimentalOptimizationLearner` instance.

    Args:
      adapt_embedding_predicate: A callable that returns True for `tf.Variable`
        attributes of the embedding function should be adapted for each task.
      num_update_steps: The number of inner loop optimization steps to take.
      additional_evaluation_update_steps: The number of additional inner loop
        optimization steps to take during evaluation (on the meta-test and
        meta-validation sets).
      first_order: If True, prevent the computation of higher order gradients.
      adapt_batch_norm: If True, adapt the scale and offset parameteres of batch
        normalization layers in the inner loop of optimization.
      clip_grad_norm: If not None, gradient dimensions are independently clipped
        to lie in the interval [-clip_grad_norm, clip_grad_norm] before being
        processed by the `update_fn`.
      update_fn: A Callable that takes in a learning rate and produces a
        function triple defining an iterative optimization process; see `sgd`
        and `adam` for examples.
      **kwargs: Keyword arguments common to all `ExperimentalEpisodicLearner`s.
    """
    self.adapt_embedding_predicate = adapt_embedding_predicate
    self.num_update_steps = num_update_steps
    self.additional_evaluation_update_steps = additional_evaluation_update_steps
    self.adapt_batch_norm = adapt_batch_norm
    self.first_order = first_order
    self.clip_grad_norm = clip_grad_norm
    self.update_fn = update_fn
    super(ExperimentalOptimizationLearner, self).__init__(**kwargs)
    assert isinstance(self.embedding_fn,
                      reparameterizable_base.ReparameterizableModule)

  def compute_loss(self, onehot_labels, predictions):
    """Computes the loss on the query set of a given episode."""
    return (self.outer_objective(
        onehot_labels=onehot_labels, predictions=predictions))

  @property
  def trainable_variables(self):
    """Returns a tuple of variables to update in the outer optimization loop."""
    raise NotImplementedError

  @property
  def task_parameters(self):
    """Returns a tuple of variables to update in the inner optimization loop."""
    raise NotImplementedError

  def episodic_init_ops(self, labels, embeddings, task_parameters):
    raise NotImplementedError

  def inner_loop_prediction(self, embeddings):
    raise NotImplementedError

  def inner_objective(self, onehot_labels, predictions, iteration_idx):
    raise NotImplementedError

  def outer_loop_prediction(self, embeddings):
    raise NotImplementedError

  def outer_objective(self, onehot_labels, predictions):
    raise NotImplementedError

  def forward_pass(self, data):
    """Wrapper around `detailed_forward_pass` to return query set predictions.

    Args:
      data: A `meta_dataset.providers.Episode` containing the data for the
        episode.

    Returns:
      A Tensor of the predictions on the query set.
    """
    forward_pass_result = self.detailed_forward_pass(data)

    post_adaptation_query_results = (
        forward_pass_result.post_adaptation_query_results)

    return post_adaptation_query_results.predictions

  def detailed_forward_pass(self, data):
    """Returns all information from a forward pass of the `OptimizationLearner`.

    Args:
      data: A `meta_dataset.providers.Episode` containing the data for the
        episode.

    Returns:
      A `collections.NamedTuple` that contains the results of the forward pass.
    """
    # Loop initialization.
    init_loop_variables = self.task_parameters
    init_loop_variable_refs = [
        v.experimental_ref() for v in init_loop_variables
    ]

    # Construct ops for data-dependent episodic initialization.
    episodic_init_ops = self.episodic_init_ops(
        labels=data.support_labels,
        embeddings=self.embedding_fn(data.support_images, training=True),
        task_parameters=init_loop_variables,
    )

    def _forward_pass(iteration_idx_, variables_mapping_, images_,
                      onehot_labels_):
      """Helper function to compute the outputs of a forward pass."""

      with self.embedding_fn.reparameterize(variables_mapping_):
        # TODO(eringrant): Implement non-transductive batch normalization (i.e.,
        # pass the support set statistics through the query set forward pass.
        embeddings_ = self.embedding_fn(images_, training=True)

      # TODO(eringrant): `head_fn` is an attribute of the subclass.
      with self.head_fn.reparameterize(variables_mapping_):
        predictions_ = self.head_fn(embeddings_)[:, :data.way]

      accuracy_ = tf.reduce_mean(
          input_tensor=self.compute_accuracy(
              onehot_labels=onehot_labels_, predictions=predictions_))

      inner_objective_ = self.inner_objective(
          onehot_labels=onehot_labels_,
          predictions=predictions_,
          iteration_idx=iteration_idx_)

      outer_objective_ = self.outer_objective(
          onehot_labels=onehot_labels_,
          predictions=predictions_,
      )

      return ForwardPass(
          embeddings=embeddings_,
          predictions=predictions_,
          inner_objective_value=inner_objective_,
          outer_objective_value=outer_objective_,
          accuracy=accuracy_,
      )

    def _objective_fn(loop_variables_, iteration_idx_):
      """Evaluate the support set objective given `loop_variables_`."""

      # Get attribute paths for the loop_variables.
      loop_variables_mapping_ = dict(
          zip(init_loop_variable_refs, loop_variables_))

      adaptation_support_results = _forward_pass(
          iteration_idx_=iteration_idx_,
          variables_mapping_=loop_variables_mapping_,
          images_=data.support_images,
          onehot_labels_=data.onehot_support_labels)

      return adaptation_support_results.inner_objective_value

    def _e_step(loop_variables_):
      """Evaluate expectations given `loop_variables_`."""

      # Get attribute paths for the loop_variables.
      loop_variables_dict_ = dict(zip(init_loop_variable_refs, loop_variables_))

      with self.embedding_fn.reparameterize(loop_variables_dict_):
        # TODO(eringrant): training to True for normalization with batch stats.
        # Figure out the appropriate way to pass this around.
        train_embeddings_ = self.embedding_fn(data.train_images, training=True)

      class_embeddings_ = learner_base.class_specific_data(
          data.onehot_train_labels, train_embeddings_, self.logit_dim)

      def _compute_responsibilities(examples_, class_idx):
        train_predictions_ = tf.squeeze(
            self.head_fn(
                embeddings=examples_, components=True, class_idx=[class_idx]),
            axis=1)
        return tf.nn.softmax(train_predictions_, axis=-1)

      with self.head_fn.reparameterize(loop_variables_dict_):
        class_responsibilities_ = [
            _compute_responsibilities(embeddings_, class_idx=i)
            for i, embeddings_ in enumerate(class_embeddings_)
        ]

      return class_embeddings_, class_responsibilities_

    def _m_step(preupdate_vars, all_embeddings_, all_responsibilities_):
      """Compute parameter estimates given `loop_variables_`."""

      means, log_scales, logits = zip(*map(
          reparameterizable_distributions.fit_gaussian_mixture, all_embeddings_,
          all_responsibilities_, itertools.repeat(self.head_fn.damping)))

      def flatten(x):
        return list(itertools.chain.from_iterable(x))

      means = flatten(means)
      log_scales = flatten(log_scales)
      logits = flatten(logits)

      if not self.head_fn.estimate_loc:
        means = [None for _ in means]

      if not self.head_fn.estimate_scale:
        log_scales = [None for _ in log_scales]

      if not self.head_fn.estimate_logits:
        logits = [None for _ in logits]

      updated_vars = means + log_scales + logits

      # Replace constant variables.
      # TODO(eringrant): This interface differs from just excluding these
      # variables from `task_variables`.
      no_none_updated_vars = []
      for preupdate_var, updated_var in zip(preupdate_vars, updated_vars):
        if updated_var is None:
          no_none_updated_vars.append(preupdate_var)
        else:
          no_none_updated_vars.append(updated_var)

      # TODO(eringrant): This assumes an ordering of mean, log_scales,
      # mixing_logits.
      return no_none_updated_vars

    # Loop body.
    with tf.control_dependencies(episodic_init_ops):

      # Inner loop of expectation maximization.
      num_em_steps = getattr(self, 'num_em_steps', 0)
      if num_em_steps > 0:
        loop_variables = em_loop(
            num_updates=self.num_em_steps,
            e_step=_e_step,
            m_step=_m_step,
            variables=loop_variables)

      # Inner loop of gradient-based optimization.
      num_optimizer_steps = (
          self.num_update_steps + (self.additional_evaluation_update_steps
                                   if not self.is_training else 0))
      if num_optimizer_steps > 0:
        # pylint: disable=no-value-for-parameter
        final_loop_variables = optimizer_loop(
            num_updates=num_optimizer_steps,
            objective_fn=_objective_fn,
            update_fn=self.update_fn,
            variables=init_loop_variables,
            first_order=self.first_order,
            clip_grad_norm=self.clip_grad_norm,
        )
        # pylint: enable=no-value-for-parameter

      # If no inner loop adaptation is performed, ensure the episodic
      # initialization is still part of the graph via a control dependency.
      if num_optimizer_steps + num_em_steps == 0:
        loop_variables = [tf.identity(v) for v in init_loop_variables]

    # Get variable references to use when remapping the loop_variables.
    init_loop_variables_mapping = dict(
        zip(init_loop_variable_refs, init_loop_variables))
    final_loop_variables_mapping = dict(
        zip(init_loop_variable_refs, final_loop_variables))

    # Collect statistics about the inner optimization.
    with tf.compat.v1.name_scope('pre-adaptation'):
      with tf.compat.v1.name_scope('support'):
        pre_adaptation_support_results = _forward_pass(
            iteration_idx_=0,
            variables_mapping_=init_loop_variables_mapping,
            images_=data.support_images,
            onehot_labels_=data.onehot_support_labels)

      with tf.compat.v1.name_scope('query'):
        pre_adaptation_query_results = _forward_pass(
            iteration_idx_=0,
            variables_mapping_=init_loop_variables_mapping,
            images_=data.query_images,
            onehot_labels_=data.onehot_query_labels)

    with tf.compat.v1.name_scope('post-adaptation'):
      with tf.compat.v1.name_scope('support'):
        post_adaptation_support_results = _forward_pass(
            iteration_idx_=num_optimizer_steps,
            variables_mapping_=final_loop_variables_mapping,
            images_=data.support_images,
            onehot_labels_=data.onehot_support_labels,
        )

      with tf.compat.v1.name_scope('query'):
        post_adaptation_query_results = _forward_pass(
            iteration_idx_=num_optimizer_steps,
            variables_mapping_=final_loop_variables_mapping,
            images_=data.query_images,
            onehot_labels_=data.onehot_query_labels,
        )

    def _support_module_objective_fn(module_variables_, module_variable_refs_):
      """Evaluate the query set objective given `module_variables_`."""
      # Use the values of the parameters at convergence as the default value.
      variables_mapping_ = final_loop_variables_mapping.copy()

      # Loop over and replace the module-specific variables.
      for module_variable_ref, module_variable in zip(module_variable_refs_,
                                                      module_variables_):
        variables_mapping_[module_variable_ref] = module_variable

      adaptation_query_results = _forward_pass(
          iteration_idx_=num_optimizer_steps,
          variables_mapping_=variables_mapping_,
          images_=data.support_images,
          onehot_labels_=data.onehot_support_labels,
      )

      return adaptation_query_results.inner_objective_value

    def _query_module_objective_fn(module_variables_, module_variable_refs_):
      """Evaluate the query set objective given `module_variables_`."""
      # Use the values of the parameters at convergence as the default value.
      variables_mapping_ = final_loop_variables_mapping.copy()

      # Loop over and replace the module-specific variables.
      for module_variable_ref, module_variable in zip(module_variable_refs_,
                                                      module_variables_):
        variables_mapping_[module_variable_ref] = module_variable

      adaptation_query_results = _forward_pass(
          iteration_idx_=num_optimizer_steps,
          variables_mapping_=variables_mapping_,
          images_=data.query_images,
          onehot_labels_=data.onehot_query_labels)

      return adaptation_query_results.inner_objective_value

    return Adaptation(
        pre_adaptation_support_results=pre_adaptation_support_results,
        post_adaptation_support_results=post_adaptation_support_results,
        pre_adaptation_query_results=pre_adaptation_query_results,
        post_adaptation_query_results=post_adaptation_query_results,
        objective_fn=_objective_fn,
        support_module_objective_fn=_support_module_objective_fn,
        query_module_objective_fn=_query_module_objective_fn,
        forward_pass_fn=_forward_pass,
        init_loop_variables_mapping=init_loop_variables_mapping,
        final_loop_variables_mapping=final_loop_variables_mapping,
    )


@gin.configurable
class HeadAndBackboneLearner(ExperimentalOptimizationLearner):
  """A head-and-backbone learner."""

  def __init__(self,
               head_cls,
               adapt_head_predicate,
               episodic_head_init_fn=None,
               **kwargs):
    """Initializes a `HeadAndBackboneLearner` instance.

    Args:
      head_cls: A subclass of `ReparameterizableModule` used to instantiate the
        head function.
      adapt_head_predicate: A callable that returns True for `tf.Variable`
        attributes of the head function should be adapted for each task.
      episodic_head_init_fn: A callable that takes in a tuple of one-hot labels,
        embeddings and head classifier weights, and produces intialization
        operations to be executed at the start of each episode. If None, no
        episodic initialization is performed.
      **kwargs: Keyword arguments common to all
        `ExperimentalOptimizationLearner`s.
    """
    super(HeadAndBackboneLearner, self).__init__(**kwargs)
    assert issubclass(head_cls, reparameterizable_base.ReparameterizableModule)
    self.adapt_head_predicate = adapt_head_predicate
    self.head_fn = head_cls(output_dim=self.logit_dim)

    def no_op_initialization(onehot_labels, embeddings, *vbls):
      del onehot_labels
      del embeddings
      del vbls
      return [tf.no_op()]

    self.episodic_head_init_fn = episodic_head_init_fn or no_op_initialization

  def compute_regularizer(self, onehot_labels, predictions):
    """Computes a regularizer, maybe using `predictions` and `onehot_labels`."""
    del onehot_labels
    del predictions
    return (tf.reduce_sum(input_tensor=self.embedding_fn.losses) +
            tf.reduce_sum(input_tensor=self.head_fn.losses))

  def build(self):
    """Instantiate the parameters belonging to this `HeadAndBackboneLearner`."""
    super(HeadAndBackboneLearner, self).build()
    if not self.head_fn.built:
      self.head_fn.build(self.embedding_shape)
    self.output_shape = self.head_fn.compute_output_shape(self.embedding_shape)

  def episodic_init_ops(self, labels, embeddings, task_parameters):
    """Return operations for episodic initalization of `task_parameters`."""
    # Isolate the head parameters.
    head_parameters = task_parameters[len(list(self.backbone_parameters)):]
    assert len(head_parameters) == len(list(self.head_parameters))
    return self.episodic_head_init_fn(labels, embeddings, *head_parameters)

  def inner_objective(self, onehot_labels, predictions, iteration_idx):
    """Alias for softmax cross entropy loss."""
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(onehot_labels, predictions)

  def outer_objective(self, onehot_labels, predictions):
    """Alias for softmax cross entropy loss."""
    cce = tf.keras.losses.CategoricalCrossentropy()
    regularization = self.compute_regularizer(
        onehot_labels=onehot_labels, predictions=predictions)
    return cce(onehot_labels, predictions) + regularization

  @property
  def variables(self):
    """Returns a tuple of this Learner's variables."""
    if not self._built:
      raise learner_base.NotBuiltError
    return self.embedding_fn.variables + self.head_fn.variables

  @property
  def trainable_variables(self):
    """Returns a tuple of this Learner's trainable variables."""
    if not self._built:
      raise learner_base.NotBuiltError
    return (self.embedding_fn.trainable_variables +
            self.head_fn.trainable_variables)

  @property
  def task_parameters(self):
    """Returns a tuple of the variables to be adapted for each task."""
    if not self._built:
      raise learner_base.NotBuiltError
    return list(itertools.chain(self.backbone_parameters, self.head_parameters))

  @property
  def backbone_parameters(self):
    return list(
        self.embedding_fn.reparameterizables(self.adapt_embedding_predicate))

  @property
  def head_parameters(self):
    return list(self.head_fn.reparameterizables(self.adapt_head_predicate))


@gin.configurable(allowlist=['prototype_multiplier'])
def proto_maml_fc_layer_init_fn(labels, embeddings, weights, biases,
                                prototype_multiplier):
  """Return a list of operations for reparameterized ProtoNet initialization."""

  # This is robust to classes missing from the training set, but assumes that
  # the last class is present.
  num_ways = tf.cast(
      tf.math.reduce_max(input_tensor=tf.unique(labels)[0]) + 1, tf.int32)

  # When there are no examples for a given class, we default its prototype to
  # zeros, per the implementation of `tf.math.unsorted_segment_mean`.
  prototypes = tf.math.unsorted_segment_mean(embeddings, labels, num_ways)

  # Scale the prototypes, which acts as a regularizer on the weights and biases.
  prototypes *= prototype_multiplier

  # logit = -<squared Euclidian distance to prototype>
  #       = -(x - p)^T.(x - p)
  #       = 2 x^T.p - p^T.p - x^T.x
  #       = x^T.w + b
  #         where w = 2p, b = -p^T.p
  output_weights = tf.transpose(a=2 * prototypes)
  output_biases = -tf.reduce_sum(input_tensor=prototypes * prototypes, axis=1)

  # We zero-pad to align with the original weights and biases.
  output_weights = tf.pad(
      tensor=output_weights,
      paddings=[[
          0, 0
      ], [0, tf.shape(input=weights)[1] - tf.shape(input=output_weights)[1]]],
      mode='CONSTANT',
      constant_values=0)
  output_biases = tf.pad(
      tensor=output_biases,
      paddings=[[
          0, tf.shape(input=biases)[0] - tf.shape(input=output_biases)[0]
      ]],
      mode='CONSTANT',
      constant_values=0)

  return [
      weights.assign(output_weights),
      biases.assign(output_biases),
  ]


def zero_init_fn(labels, embeddings, *vbls):
  """Return a list of operations for initialization at zero."""
  del labels
  del embeddings
  return [vbl.assign(tf.zeros_like(vbl)) for vbl in vbls]


@gin.configurable
class MAML(HeadAndBackboneLearner):
  """A 'model-agnostic' meta-learner."""

  def __init__(self, proto_maml_fc_layer_init, zero_fc_layer_init, **kwargs):
    """Initializes a MAML instance.

    Args:
      proto_maml_fc_layer_init: Whether to use `PrototypicalNetwork`-equivalent
        fc layer initialization.
      zero_fc_layer_init: Whether to initialize the parameters of the output
        layer to zero.
      **kwargs: Keyword arguments common to all `HeadAndBackboneLearner`s.

    Raises:
      ValueError: If both `proto_maml_fc_layer_init` and `zero_fc_layer_init`
      are `True`.
    """
    if proto_maml_fc_layer_init and zero_fc_layer_init:
      raise ValueError('Conflicting initialization options for `MAML`.')

    super(MAML, self).__init__(
        episodic_head_init_fn=(proto_maml_fc_layer_init_fn
                               if proto_maml_fc_layer_init else
                               zero_init_fn if zero_fc_layer_init else None),
        adapt_embedding_predicate=reparameterizable_base.is_trainable_variable,
        adapt_head_predicate=reparameterizable_base.is_trainable_variable,
        head_cls=reparameterizable_backbones.LinearModel,
        **kwargs)


@gin.configurable
class ANIL(HeadAndBackboneLearner):
  """An 'almost-no-inner-loop' learner."""

  def __init__(self, proto_maml_fc_layer_init, zero_fc_layer_init, **kwargs):
    """Initializes an ANIL instance.

    Args:
      proto_maml_fc_layer_init: Whether to use `PrototypicalNetwork`-equivalent
        fc layer initialization.
      zero_fc_layer_init: Whether to initialize the parameters of the output
        layer to zero.
      **kwargs: Keyword arguments common to all `HeadAndBackboneLearner`s.

    Raises:
      ValueError: If both `proto_maml_fc_layer_init` and `zero_fc_layer_init`
      are `True`.
    """
    if proto_maml_fc_layer_init and zero_fc_layer_init:
      raise ValueError('Conflicting initialization options for `ANIL`.')

    super(ANIL, self).__init__(
        episodic_head_init_fn=(proto_maml_fc_layer_init_fn
                               if proto_maml_fc_layer_init else
                               zero_init_fn if zero_fc_layer_init else None),
        adapt_embedding_predicate=lambda x: False,
        adapt_head_predicate=reparameterizable_base.is_trainable_variable,
        head_cls=reparameterizable_backbones.LinearModel,
        **kwargs)


@gin.configurable
def generative_then_discriminative_schedule(proportion_generative, num_updates):
  num_generative_updates = int(proportion_generative * num_updates)
  num_discriminative_updates = num_updates - num_generative_updates
  return [0.0] * num_generative_updates + [1.0] * num_discriminative_updates


@gin.configurable
class GenerativeClassifier(HeadAndBackboneLearner):
  """A generative classifier."""

  def __init__(self, generative_scaling, interpolation_schedule, **kwargs):
    """Initializes a GenerativeClassifier instance.

    Args:
      generative_scaling:
      interpolation_schedule: A callable that produces a sequence of
        coefficients used to interpolate between the generative and
        discriminative objectives. additional_evaluation_update_steps] array of
        coefficients used to interpolate between the generative and
        discriminative objectives.
      **kwargs: Keyword arguments common to all `HeadAndBackboneLearner`s.
    """

    super(GenerativeClassifier, self).__init__(
        adapt_embedding_predicate=lambda x: False,
        adapt_head_predicate=reparameterizable_base.is_trainable_variable,
        **kwargs)
    assert isinstance(
        self.head_fn,
        reparameterizable_distributions.ReparameterizableClassMixture)

    self.generative_scaling = generative_scaling

    self.gen_disc_interpolation = (
        interpolation_schedule(num_updates=self.num_update_steps) +
        [1.0] * self.additional_evaluation_update_steps
    )  # Assume discriminative.
    assert all(coef >= 0 for coef in self.gen_disc_interpolation), (
        'Interpolation coefficient should be nonnegative.')

    # Validate interpolation coefficient.
    # TODO(eringrant): generalize to other models admitting EM.
    if isinstance(self.head_fn,
                  reparameterizable_distributions.GaussianMixture):
      # Override the usual generative training to perform EM.
      try:
        num_em_steps = self.gen_disc_interpolation.index(1.0)
      except ValueError:
        # All steps are EM.
        num_em_steps = self.num_update_steps
      assert (
          all(coef == 0.0
              for coef in self.gen_disc_interpolation[:num_em_steps]) and
          all(coef == 1.0
              for coef in self.gen_disc_interpolation[num_em_steps:])
      ), ('Each step must be fully discriminative or generative when using EM.')
      self.num_em_steps = num_em_steps
      self.num_update_steps -= num_em_steps

  @property
  def task_parameters(self):
    return self.head_fn.task_parameters

  def joint_log_likelihood(self, onehot_labels, log_probs):
    """Compute p(z, y)."""
    labels = tf.cast(
        tf.reduce_sum(input_tensor=onehot_labels, axis=0), dtype=tf.float32)
    class_log_probs = tf.math.log(labels / tf.reduce_sum(input_tensor=labels))
    return log_probs + tf.expand_dims(class_log_probs, 0)

  def inner_objective(self, onehot_labels, predictions, iteration_idx):
    """Compute the inner-loop objective."""
    # p(z, y), joint log-likelihood.
    joint_log_probs = self.joint_log_likelihood(onehot_labels, predictions)
    labels = tf.expand_dims(tf.argmax(input=onehot_labels, axis=-1), axis=-1)
    numerator = tf.gather(joint_log_probs, labels, axis=-1, batch_dims=1)

    # p(z), normalization constant.
    evidence = tf.reduce_logsumexp(
        input_tensor=joint_log_probs, axis=-1, keepdims=True)

    # p(y | z) if interpolation coefficient > 0 else p(z, y).
    # TODO(eringrant): This assumes that `interp` is either 1 or 0.
    # Adapt to a hybridized approach.
    interp = tf.gather(self.gen_disc_interpolation, iteration_idx)
    scale = tf.cond(
        pred=interp > 0.0,
        true_fn=lambda: 1.0,
        false_fn=lambda: self.generative_scaling)

    return -scale * tf.reduce_mean(
        input_tensor=numerator - interp * evidence, axis=0)

  def outer_objective(self, onehot_labels, predictions):
    """Compute the outer-loop objective."""
    joint_log_probs = self.joint_log_likelihood(onehot_labels, predictions)
    cce = tf.keras.losses.CategoricalCrossentropy()
    regularization = self.compute_regularizer(
        onehot_labels=onehot_labels, predictions=predictions)
    return cce(onehot_labels, joint_log_probs) + regularization

  def validate_model_independence(self, labels, log_probs, task_parameters):
    """Partition gradients into those assumed active and inactive."""
    num_task_parameters = len(task_parameters)
    # pylint: disable=g-complex-comprehension
    on_gradients = [[
        tf.norm(tensor=on_gradient) for on_gradient in on_gradients
    ] for on_gradients in [
        tf.gradients(
            ys=tf.gather(log_probs, tf.compat.v1.where(tf.equal(labels, i))),
            xs=task_parameters[i * num_task_parameters:(i + 1) *
                               num_task_parameters]) for i in range(1)
    ]]
    off_gradients = [[
        tf.norm(tensor=off_gradient) for off_gradient in off_gradients
    ] for off_gradients in [
        tf.gradients(
            ys=tf.gather(log_probs, tf.compat.v1.where(tf.equal(labels, i))),
            xs=task_parameters[i * num_task_parameters:(i + 1) *
                               num_task_parameters]) for i in range(1)
    ]]
    # pylint: enable=g-complex-comprehension

    return (list(itertools.chain.from_iterable(on_gradients)),
            list(itertools.chain.from_iterable(off_gradients)))
