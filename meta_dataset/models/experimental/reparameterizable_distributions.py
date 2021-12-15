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

"""Parameteric density estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import gin.tf
from meta_dataset.learners.experimental import base as learner_base
from meta_dataset.models.experimental import reparameterizable_base
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

META_PARAMETER_SCOPE = 'meta_parameter'
TASK_PARAMETER_SCOPE = 'task_parameter'


def is_meta_variable(obj):
  return isinstance(obj, tf.Variable) and META_PARAMETER_SCOPE in obj.name


def is_task_variable(obj):
  return isinstance(obj, tf.Variable) and TASK_PARAMETER_SCOPE in obj.name


def _assign_pairs(vbls, values):
  return [vbl.assign(value) for vbl, value in zip(vbls, values)]


def _split_and_squeeze(tensor, num_splits, axis=0):
  return [
      tf.squeeze(t)
      for t in tf.split(tensor, axis=axis, num_or_size_splits=num_splits)
  ]


def fit_gaussian(embeddings, damping=1e-7, full_covariance=False):
  """Fits a unimodal Gaussian distribution to `embeddings`.

  Args:
    embeddings: A [batch_size, embedding_dim] tf.Tensor of embeddings.
    damping: The scale of the covariance damping coefficient.
    full_covariance: Whether to use a full or diagonal covariance.

  Returns:
    Parameter estimates (means and log variances) for a Gaussian model.
  """
  if full_covariance:
    num, dim = tf.split(tf.shape(input=embeddings), num_or_size_splits=2)
    num, dim = tf.squeeze(num), tf.squeeze(dim)
    sample_mean = tf.reduce_mean(input_tensor=embeddings, axis=0)
    centered_embeddings = embeddings - sample_mean
    sample_covariance = tf.einsum('ij,ik->kj', centered_embeddings,
                                  centered_embeddings)  # Outer product.
    sample_covariance += damping * tf.eye(dim)  # Positive definiteness.
    sample_covariance /= tf.cast(num, dtype=tf.float32)  # Scale by N.
    return sample_mean, sample_covariance
  else:
    sample_mean, sample_variances = tf.nn.moments(x=embeddings)
    log_variances = tf.math.log(sample_variances +
                                damping * tf.ones_like(sample_variances))
    return sample_mean, log_variances


def fit_gaussian_mixture(embeddings,
                         responsibilities,
                         damping=1e-7,
                         full_covariance=False):
  """Fits a unimodal Gaussian distribution `embeddings`.

  Args:
    embeddings: A [batch_size, embedding_dim] tf.Tensor of embeddings.
    responsibilities: The per-component responsibilities.
    damping: The scale of the covariance damping coefficient.
    full_covariance: Whether to use a full or diagonal covariance.

  Returns:
    Parameter estimates for a Gaussian mixture model.
  """

  num, dim = tf.split(tf.shape(input=embeddings), num_or_size_splits=2)
  num, dim = tf.squeeze(num), tf.squeeze(dim)
  num_classes = responsibilities.shape[1]

  mixing_proportion = tf.einsum('jk->k', responsibilities)
  mixing_proportion /= tf.cast(num, dtype=tf.float32)
  mixing_logits = tf.math.log(mixing_proportion)

  sample_mean = tf.einsum('ij,ik->jk', responsibilities, embeddings)
  sample_mean /= tf.reduce_sum(
      input_tensor=responsibilities, axis=0)[:, tf.newaxis]
  centered_embeddings = (
      embeddings[:, tf.newaxis, :] - sample_mean[tf.newaxis, :, :])

  if full_covariance:
    sample_covariance = tf.einsum('ijk,ijl->ijkl', centered_embeddings,
                                  centered_embeddings)  # Outer product.
    sample_covariance += damping * tf.eye(dim)  # Positive definiteness.
    weighted_covariance = tf.einsum('ij,ijkl->jkl', responsibilities,
                                    sample_covariance)
    weighted_covariance /= tf.reduce_sum(
        input_tensor=responsibilities, axis=0)[:, tf.newaxis, tf.newaxis]

    return (
        _split_and_squeeze(sample_mean, num_splits=num_classes),
        _split_and_squeeze(weighted_covariance, num_splits=num_classes),
        [mixing_logits],
    )
  else:
    avg_x_squared = (
        tf.matmul(responsibilities, embeddings**2, transpose_a=True) /
        tf.reduce_sum(input_tensor=responsibilities, axis=0)[:, tf.newaxis])
    avg_means_squared = sample_mean**2
    avg_x_means = (
        sample_mean *
        tf.matmul(responsibilities, embeddings, transpose_a=True) /
        tf.reduce_sum(input_tensor=responsibilities, axis=0)[:, tf.newaxis])
    sample_variances = (
        avg_x_squared - 2 * avg_x_means + avg_means_squared +
        damping * tf.ones(dim))
    log_variances = tf.math.log(sample_variances)
    return (
        _split_and_squeeze(sample_mean, num_splits=num_classes),
        _split_and_squeeze(log_variances, num_splits=num_classes),
        [mixing_logits],
    )


class ReparameterizableClassMixture(
    reparameterizable_base.ReparameterizableModule):
  """A reparameterizable mixture of class-conditional generative models.

  All meta-level variables must be created when the instance is instantiated
  (which is automatically wrapped in a name_scope).

  Subclasses must implement:
    - trainable_parameters: Returns the list of parameters to be optimized in
      the outer loop.
    - init_task_parameters(class_embeddings): Init episodic task parameters.
    - components: Construct components on-the-fly from the task parameters.
  """

  def __init__(self, num_dims, output_dim, name=None):
    """A class mixture.

    Args:
      num_dims: int, number of dimensions.
      output_dim: int, number of components.
      name: str, name scope to use for variable creation.
    """
    super(ReparameterizableClassMixture, self).__init__(name=name)
    self.num_dims = num_dims
    self.num_components = output_dim

    with tf.compat.v1.name_scope(META_PARAMETER_SCOPE):
      self.build_meta_parameters()
    with tf.compat.v1.name_scope(TASK_PARAMETER_SCOPE):
      self.build_task_parameters()

  def reparameterizables(self, predicate, with_path=False):
    """Override `reparameterizables` to return only task parameters."""

    def predicate_and_is_task_variable(obj):
      return predicate(obj) and is_task_variable(obj)

    return super(ReparameterizableClassMixture, self).reparameterizables(
        predicate=predicate_and_is_task_variable, with_path=with_path)

  @property
  def task_parameters(self):
    return self.reparameterizables(
        lambda obj: isinstance(obj, tf.Variable), with_path=False)

  def trainable_variables(self):
    """Override `trainable_variables` to return only meta-parameters."""

    def trainable_and_is_meta_variable(obj):
      return reparameterizable_base.is_trainable_variable(
          obj) and is_meta_variable(obj)

    return super(ReparameterizableClassMixture, self).reparameterizables(
        predicate=trainable_and_is_meta_variable, with_path=False)

  def build_meta_parameters(self):
    """Assign to attributes the task parameters."""
    raise NotImplementedError

  def build_task_parameters(self):
    """Assign to attributes the meta parameters."""
    raise NotImplementedError

  def episodic_init_ops(self, onehot_labels, embeddings):
    """Perform (data-dependent) initialization operations."""
    raise NotImplementedError

  @property
  def components(self):
    """A list of tfd.Distributions constructed on-the-fly from task params."""
    raise NotImplementedError

  def __call__(self, embeddings):
    """Compute log probabilities of embeddings under each class-conditional.

    Args:
      embeddings: A [num_examples, embedding_dim] Tensor of embeddings.

    Returns:
      A [num_examples, num_components] Tensor of log_probabalities.
    """
    return tf.stack([c.log_prob(embeddings) for c in self.components], axis=-1)


@gin.configurable
class MultivariateNormalDiag(ReparameterizableClassMixture):
  """A parametrized normal distribution."""

  def __init__(self, estimate_scale, damping, **kwargs):
    """A class mixture.

    Args:
      estimate_scale: Whether to estimate the scale in addition to the location.
      damping: The scale of the covariance damping coefficient.
      **kwargs: Keyword arguments common to `ReparameterizableClassMixture`s.
    """
    super(MultivariateNormalDiag, self).__init__(**kwargs)
    self.damping = damping
    self.estimate_scale = estimate_scale

  def build_meta_parameters(self):
    """Assign to attributes the task parameters."""
    return

  def build_task_parameters(self):
    """Assign to attributes the meta parameters."""
    self.locs = [
        tf.Variable(tf.zeros((self.num_dims)), name='loc_{}'.format(i))
        for i in range(self.num_components)
    ]
    self.log_scales = [
        tf.Variable(tf.zeros((self.num_dims)), name='log_scale_{}'.format(i))
        for i in range(self.num_components)
    ]

  def episodic_init_ops(self, onehot_labels, embeddings, task_parameters):
    """Perform (data-dependent) initialization operations."""
    del task_parameters
    class_embeddings = learner_base.class_specific_data(onehot_labels,
                                                        embeddings,
                                                        self.num_components)

    locs, log_scales = zip(
        *map(fit_gaussian, class_embeddings, itertools.repeat(self.damping)))
    if not self.estimate_scale:
      log_scales = [tf.zeros_like(log_scale) for log_scale in log_scales]

    return (_assign_pairs(self.locs, locs) +
            _assign_pairs(self.log_scales, log_scales))

  @property
  def components(self):
    """A list of tfd.Distributions constructed on-the-fly from task params."""
    # pylint: disable=g-complex-comprehension
    return [
        tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.math.softplus(log_scale),
            allow_nan_stats=False)
        for loc, log_scale in zip(self.locs, self.log_scales)
    ]
    # pylint: enable=g-complex-comprehension


@gin.configurable
class GaussianMixture(ReparameterizableClassMixture):
  """A parametrized Gaussian mixture distribution."""

  def __init__(self,
               num_modes,
               damping,
               loc_initializer=tf.initializers.random_uniform(),
               log_scale_initializer=tf.initializers.zeros(),
               logits_initializer=tf.initializers.zeros(),
               trainable_loc=True,
               trainable_scale=True,
               trainable_logits=True,
               estimate_loc=True,
               estimate_scale=True,
               estimate_logits=True,
               **kwargs):
    """A parametrized Gaussian mixture distribution.

    Args:
      num_modes: int, number of modes.
      damping: The scale of the covariance damping coefficient.
      loc_initializer: initializer for the location vector.
      log_scale_initializer: initializer for the log-scale vector.
      logits_initializer: initializer for the mixture's logit vector.
      trainable_loc: bool, if `True` and `trainable` is `True`, the mixture locs
        are also trainable.
      trainable_scale: bool, if `True` and `trainable` is `True`, the mixture
        scales are also trainable.
      trainable_logits: bool, if `True` and `trainable` is `True`, the mixture
        logits are also trainable.
      estimate_loc: Whether to estimate the location parameters.
      estimate_scale: Whether to estimate the scale parameters.
      estimate_logits: Whether to estimate the logit parameters.
      **kwargs: Keyword arguments commom to `ReparameterizableClassMixture`s.
    """
    self.num_modes = num_modes
    self.damping = damping

    self.loc_initializer = loc_initializer
    self.log_scale_initializer = log_scale_initializer
    self.logits_initializer = logits_initializer

    self.trainable_loc = trainable_loc
    self.trainable_scale = trainable_scale
    self.trainable_logits = trainable_logits

    self.estimate_loc = estimate_loc
    self.estimate_scale = estimate_scale
    self.estimate_logits = estimate_logits

    super(GaussianMixture, self).__init__(**kwargs)

  def build_meta_parameters(self):
    """Assign to attributes the task parameters."""
    self.meta_loc = tf.Variable(
        self.loc_initializer([self.num_modes, self.num_dims]),
        trainable=self.trainable_loc,
        name='meta_loc')
    self.meta_log_scale = tf.Variable(
        self.log_scale_initializer([self.num_modes, self.num_dims]),
        trainable=self.trainable_scale,
        name='meta_log_scale')
    self.meta_logits = tf.Variable(
        self.logits_initializer([self.num_modes]),
        trainable=self.trainable_logits,
        name='meta_logits')

  def build_task_parameters(self):
    """Assign to attributes the meta-parameters."""

    def _construct_variables():
      """Construct an initialization for task parameters."""

      def _split_mode_params(params):
        return [
            tf.squeeze(p) for p in tf.split(
                params, axis=0, num_or_size_splits=self.num_modes)
        ]

      locs = _split_mode_params(tf.zeros_like(self.meta_loc))
      log_scales = _split_mode_params(tf.zeros_like(self.meta_log_scale))
      logits = tf.zeros_like(self.meta_logits)

      return (
          [tf.Variable(loc, 'loc') for loc in locs],
          [tf.Variable(log_scale, 'log_scale') for log_scale in log_scales],
          tf.Variable(logits, 'logits'),
      )

    locs, log_scales, logits = [], [], []
    for i in range(self.num_components):
      with tf.compat.v1.name_scope('class_{}'.format(i)):
        class_locs, class_log_scales, class_logits = _construct_variables()
        locs += [class_locs]
        log_scales += [class_log_scales]
        logits += [class_logits]

    self.task_locs = locs
    self.task_log_scales = log_scales
    self.task_logits = logits

  def episodic_init_ops(self, onehot_labels, embeddings):
    """Returns data-independent `GaussianMixture` initialization operations."""
    del onehot_labels
    del embeddings

    def _split_mode_params(params):
      return [
          tf.squeeze(p)
          for p in tf.split(params, axis=0, num_or_size_splits=self.num_modes)
      ]

    init_ops = []
    for component_locs, component_log_scales, component_logits in zip(
        self.task_locs, self.task_log_scales, self.task_logits):
      init_ops += _assign_pairs(component_locs,
                                _split_mode_params(self.meta_loc))
      init_ops += _assign_pairs(component_log_scales,
                                _split_mode_params(self.meta_log_scale))
      init_ops += [component_logits.assign(self.meta_logits)]

    return init_ops

  @property
  def components(self):
    """A list of tfd.Distributions constructed on-the-fly from task params."""
    # pylint: disable=g-complex-comprehension
    return [
        tfd.Mixture(
            cat=tfd.Categorical(logits=logits),
            components=[
                tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.math.softplus(log_scale),
                    allow_nan_stats=False)
                for loc, log_scale in zip(locs, log_scales)
            ])
        for locs, log_scales, logits in zip(
            self.task_locs, self.task_log_scales, self.task_logits)
    ]
    # pylint: enable=g-complex-comprehension

  def __call__(self, embeddings, components=False, class_idx=None):
    """Compute log probabilities of embeddings under each class-conditional."""
    if class_idx:
      class_models = [self.components[i] for i in class_idx]
    else:
      class_models = self.components

    if components:
      return tf.stack([
          tf.stack([c.log_prob(embeddings)
                    for c in cs.components], axis=-1)
          for cs in class_models
      ],
                      axis=1)
    else:
      return tf.stack([c.log_prob(embeddings) for c in class_models], axis=-1)


class DeepDensity(ReparameterizableClassMixture):
  """A parametric density model."""

  @property
  def meta_attribute_name(self):
    raise NotImplementedError

  @property
  def task_attribute_name(self):
    raise NotImplementedError

  @property
  def keyed_variables(self):
    return

  def build_class_density_model(self):
    """Build a deep density model for a single class."""
    raise NotImplementedError

  def episodic_init_ops(self, onehot_labels, embeddings):
    """Perform (data-independent) initialization operations."""
    del onehot_labels
    del embeddings

    init_ops = []
    for i in range(self.num_components):
      component_variables = dict(
          (k, v)
          for k, v in self.keyed_variables.items()
          if '{}/{}'.format(self.task_attribute_name, i) in k)
      sorted_component_variable_keys = sorted(component_variables.keys())
      sorted_meta_variable_keys = sorted(self.meta_parameters.keys())
      sorted_component_variables = (
          component_variables[k] for k in sorted_component_variable_keys)
      sorted_meta_variables = (
          self.meta_parameters[k] for k in sorted_meta_variable_keys)
      init_ops += [
          assignee_variables.assign(assigner_variables)
          for assignee_variables, assigner_variables in zip(
              sorted_component_variables, sorted_meta_variables)
      ]
    return init_ops

  def trainable_variables(self):
    """Return a dict of meta_parameters to optimize in the inner loop."""
    # Override this because reflection seems not to work.
    return dict((k, v)
                for k, v in self.keyed_variables.items()
                if self.task_attribute_name in k)

  def build_meta_parameters(self):
    """Assign to attributes the meta parameters."""
    setattr(self, self.meta_attribute_name, self.build_class_density_model())

  def build_task_parameters(self):
    """Assign to attributes the meta parameters."""
    task_densities = []
    for i in range(self.num_components):
      with tf.compat.v1.name_scope('class_{}'.format(i)):
        task_densities += [self.build_class_density_model()]
    setattr(self, self.task_attribute_name, task_densities)


@gin.configurable
class RealNVPModel(DeepDensity):
  """A RealNVP density model."""

  @property
  def meta_attribute_name(self):
    return 'meta_shift_and_log_scale_model'

  @property
  def task_attribute_name(self):
    return 'task_shift_and_log_scale_models'

  def __init__(self, num_coupling_layers, num_hidden_units, num_hidden_layers,
               **kwargs):
    """A parameterized RealNVP Distribution.

    Args:
      num_coupling_layers: int, number of RealNVP affine coupling layers.
      num_hidden_units: int, number of hidden units within a hidden layer in the
        function computing shift and log-scale coefficients within a RealNVP
        affine coupling layer.
      num_hidden_layers: int, number of hidden layers in the function that
        computes shift and log-scale coefficients within a RealNVP affine
        coupling layer.
      **kwargs: Keyword arguments common to all `DeepDensity` instances.
    """
    self.num_hidden_units = num_hidden_units
    self.num_hidden_layers = num_hidden_layers
    self.num_coupling_layers = num_coupling_layers
    super(RealNVPModel, self).__init__(**kwargs)

  def _compute_num_masked(self, index):
    """Returns the number of masked units for the coupling layer at `index`."""
    rounding_fn = (math.ceil if index % 2 == 0 else math.floor)
    return int(rounding_fn(self.num_dims / 2.0))

  def build_class_density_model(self):
    """Build a deep density model for a single class."""

    # Build "shift and log-scale" models and their corresponding functions for
    # the RealNVP affine coupling layers.
    shift_and_log_scale_models = []

    for i in range(self.num_coupling_layers):
      # Compute the number of masked units for this coupling layer. Coupling
      # layers are chained by reversing the order of units between each layer,
      # which means that for an odd number of dimensions the number of masked
      # units will vary across coupling layers (as we can't partition the
      # input into two equally-sized sets of units).
      num_masked = self._compute_num_masked(i)

      # Build the model computing the shift and log-scale coefficients within
      # the coupling layer.
      layers = []
      for j in range(self.num_hidden_layers):
        layers.append(
            tf.keras.layers.Dense(
                self.num_hidden_units,
                activation=tf.nn.relu,
                trainable=True,
                name='dense_{}'.format(j)))
      # The output layer produces shift and log-scale coefficients for the
      # other `num_dims - num_masked` non-masked units.
      layers.append(
          tf.keras.layers.Dense(
              2 * (self.num_dims - num_masked),
              activation=None,
              trainable=True,
              name='dense_{}'.format(self.num_hidden_layers)))

      shift_and_log_scale_model = tf.keras.Sequential(
          layers, name='coupling_layer_{}'.format(i))

      # We force variable creation here so that the parametrized distribution
      # has immediate access to its underlying variables.
      shift_and_log_scale_model.build([1, num_masked])
      shift_and_log_scale_models.append(shift_and_log_scale_model)

    return shift_and_log_scale_models

  @property
  def components(self):
    """A list of tfd.Distributions constructed on-the-fly from task params."""

    def _make_shift_and_log_scale_fn(shift_and_log_scale_model, num_masked):
      """Returns a function that computes shift and log-scale coefficients.

      RealNVP expects the function to accept an `output_dims` argument, which
      allows lazy variable instantiation. We already know what its value is
      (`num_dims - num_masked`), so we simply make sure it's what we expect.

      Args:
        shift_and_log_scale_model: A Keras model that computes the fprop.
        num_masked: int, number of masked unit.

      Returns:
        A function that computes shift and log-scale coefficients.
      """

      def _shift_and_log_scale_fn(x, output_dims, *args, **kwargs):
        del args
        del kwargs
        if output_dims != self.num_dims - num_masked:
          raise ValueError('Expected {} output_dims, got {}.'.format(
              self.num_dims - num_masked, output_dims))
        return tf.split(shift_and_log_scale_model(x), 2, axis=-1)

      return _shift_and_log_scale_fn

    class_bijectors = []
    for j in range(self.num_components):
      bijectors = []
      for i, shift_and_log_scale_model in enumerate(
          getattr(self, self.task_attribute_name)[j]):
        num_masked = self._compute_num_masked(i)

        # Create functions to compute the shift and log-scale coefficients from
        # the parameterized shift_and_log_scale_models.
        shift_and_log_scale_fn = _make_shift_and_log_scale_fn(
            shift_and_log_scale_model, num_masked)

        bijector = tfb.RealNVP(
            num_masked=num_masked,
            shift_and_log_scale_fn=shift_and_log_scale_fn)

        # We reverse the order of units in-between RealNVP coupling layers,
        # which allows us to chain them.
        if i > 0:
          bijectors.append(tfb.Permute(list(range(self.num_dims))[::-1]))

        bijectors.append(bijector)
      class_bijectors.append(bijectors)

    return [
        tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(tf.zeros([self.num_dims])),
            bijector=tfb.Chain(bijectors)) for bijectors in class_bijectors
    ]


@gin.configurable
class MADEModel(DeepDensity):
  """TODO."""

  @property
  def meta_attribute_name(self):
    return 'meta_autoregressive_network'

  @property
  def task_attribute_name(self):
    return 'task_autoregressive_networks'

  def __init__(self, num_hidden_units, num_hidden_layers, shift_only, **kwargs):
    """A parameterized RealNVP Distribution.

    Args:
      num_hidden_units: int, the number of units in each hidden layer.
      num_hidden_layers: int, the number of hidden layers.
      shift_only: Whether the shift parameters are included.
      **kwargs: Keyword arguments common to all `DeepDensity` models.
    """
    self.num_hidden_units = num_hidden_units
    self.num_hidden_layers = num_hidden_layers
    self.num_params_per_input = 1 if shift_only else 2
    super(MADEModel, self).__init__(**kwargs)

  def build_class_density_model(self):
    """Build a deep density model for a single class."""
    autoregressive_network = tfb.AutoregressiveNetwork(
        params=self.num_params_per_input,
        event_shape=self.num_dims,
        hidden_units=[self.num_hidden_units] * self.num_hidden_layers,
        activation='relu',
    )
    autoregressive_network.build([1, self.num_dims])
    return autoregressive_network

  @property
  def components(self):
    """A list of tfd.Distributions constructed on-the-fly from task params."""

    def _make_masked_autoregressive_shift_and_log_scale_fn(
        masked_autoregressive_model):
      """Returns a function that computes shift and log-scale coefficients.

      Args:
        masked_autoregressive_model: A Keras model that computes the fprop.

      Returns:
        A function that computes shift and log-scale coefficients.
      """
      if self.num_params_per_input == 1:
        return lambda x: (masked_autoregressive_model(x)[Ellipsis, 0], None)
      else:
        return lambda x: tf.unstack(masked_autoregressive_model(x), axis=-1)

    masked_autoregressive_shift_and_log_scale_fns = [
        _make_masked_autoregressive_shift_and_log_scale_fn(made)
        for made in getattr(self, self.task_attribute_name)
    ]

    # pylint: disable=g-complex-comprehension
    return [
        tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0., scale=1.),
            bijector=tfb.MaskedAutoregressiveFlow(made_fn),
            event_shape=[self.num_dims])
        for made_fn in masked_autoregressive_shift_and_log_scale_fns
    ]
    # pylint: enable=g-complex-comprehension
