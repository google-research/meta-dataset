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
# pyformat: disable
"""Implementation of the criticality measures from [Chatterji et al. (2020)][1].

#### References

[1]: Chatterji, Niladri S., Behnam Neyshabur, and Hanie Sedghi. The intriguing
     role of module criticality in the generalization of deep networks.
     In _Proceedings of 8th International Conference on Learning
     Representations_, 2020.
     https://arxiv.org/abs/1912.00528
"""
# pyformat: enable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from typing import Callable, Iterable

import gin.tf
import numpy as np
import tensorflow as tf

ModuleCriticalityAnalysis = collections.namedtuple(
    'ModuleCriticalityAnalysis', (
        'criticality_score',
        'alpha',
        'sigma',
        'loss_value',
        'num_samples_per_iteration',
        'alpha_grid_size',
        'sigma_grid_size',
        'sigma_ratio',
        'initial_objective_value',
        'final_objective_value',
    ))


def relative_error_condition(error,
                             reference_error,
                             rtol = 1.01):
  return tf.reduce_mean(error) <= rtol * reference_error


def _squared_frobenius_norm(t):
  return tf.reduce_sum(tf.square(t))


def _interpolate_and_perturb(
    alpha,
    sigma,
    params_init,
    params_final,
    objective_fn,
    num_samples_per_iteration,
    loss_threshold_condition,
    normalize_error,
):
  """Interpolate `params_init` and `params_final`, perturb, then evaluate.

  Args:
    alpha: The linear interpolation coefficient.
    sigma: The standard deviation of the Gaussian perturbations.
    params_init: The initial parameter settings, activated when `alpha` is zero.
    params_final: The final parameter settings, activated when `alpha` is one.
    objective_fn: A function that returns the objective value when passed an
      (interpolated) parameter value.
    num_samples_per_iteration: Number of perturbations to sample each iteration.
    loss_threshold_condition: A function that takes in a reference objective
      value and a candidate objective value and produces a thresholding
      decision.
    normalize_error: Whether to normalize the error that is minimized over in
      the definition of criticality by the Frobenius norm of the distance
      between initial and final parameters.

  Returns:
    The average loss of the interpolation across perturbations.
  """

  # Linearly interpolate.
  params = ((1 - alpha) * param_init + alpha * param_final
            for param_init, param_final in zip(params_init, params_final))

  # Resample perturbations and evaluate the objective.
  perturbed_losses = []
  for _ in range(num_samples_per_iteration):
    perturbed_losses += [
        objective_fn(
            (param + tf.random.normal(tf.shape(param), mean=0.0, stddev=sigma)
             for param in params))
    ]

  # Compute the Frobenius norm between the final and initial parameter values.
  squared_distance_norm = tf.reduce_sum(
      tuple(
          _squared_frobenius_norm(param_final - param_init)
          for param_init, param_final in zip(params_init, params_final)))

  # Estimate the expected loss over perturbations.
  mean_perturbed_loss = tf.reduce_mean(perturbed_losses)

  # Note: This error normalization was not included in the final paper, but
  # was discussed as a variant that accounts for large differences in norms
  # between different parameter tensors.
  if normalize_error:
    # Normalize by the Frobenius norm of the parameter distance.
    mean_perturbed_loss /= tf.math.sqrt(squared_distance_norm)

  # Compute the weighted norm in the definition of criticality.
  weighted_squared_distance_norm = ((alpha / sigma)**2 * squared_distance_norm)

  return (
      loss_threshold_condition(mean_perturbed_loss),
      mean_perturbed_loss,
      weighted_squared_distance_norm,
  )


@gin.configurable(
    allowlist=(
        'num_samples_per_iteration',
        'alpha_grid_size',
        'sigma_grid_size',
        'sigma_ratio',
        'loss_threshold_condition',
        'normalize_error',
    ))
def compute_module_criticality(
    objective_fn,
    module_variables_init,
    module_variables_final,
    num_samples_per_iteration=10,
    alpha_grid_size=10,
    sigma_grid_size=10,
    sigma_ratio=1.0,
    loss_threshold_condition=relative_error_condition,
    normalize_error=False,
):
  """Compute the criticality of a module parameterized by `module_variables`.

  Args:
    objective_fn: A callable that takes in an iterable of the module-specific
      variables and produces the value of the objective function.
    module_variables_init: A list of tf.Tensors; the variables of the module at
      initialization.
    module_variables_final: A list of tf.Tensors; the variables of the module at
      convergence.
    num_samples_per_iteration: Number of perturbations to sample each iteration.
    alpha_grid_size: The number of values to test for alpha, the interpolation
      coefficient.
    sigma_grid_size: The number of values to test for sigma, the standard
      deviation of the perturbation.
    sigma_ratio: Positive scalar multiplier k for values of sigma, to enforce
      that the tested values of sigma lie in [k * 1e-16, k]; the default is 1.0,
      implying that the tested values of sigma lie in the interval [1e-16, 1].
    loss_threshold_condition: A callable that takes in a reference objective
      value and a candidate objective value and produces a thresholding
      decision.
    normalize_error: Whether to normalize the error that is minimized over in
      the definition of criticality by the Frobenius norm of the distance
      between initial and final parameters.

  Returns:
    A `collections.NamedTuple` that contains the results of the criticality
    analysis.
  """
  initial_objective_value = objective_fn(module_variables_init)
  final_objective_value = objective_fn(module_variables_final)

  # Test a 2D grid of alpha and sigma values.
  float_zero = tf.cast(0, tf.float32)
  alphas, sigmas = tf.meshgrid(
      tf.linspace(float_zero, 1, alpha_grid_size + 1),
      tf.linspace(float_zero + 1e-16, 1, sigma_grid_size + 1) * sigma_ratio,
  )
  alphas, sigmas = tf.reshape(alphas, [-1]), tf.reshape(sigmas, [-1])

  def _evaluate_alpha_sigma(alpha_sigma):
    alpha, sigma = alpha_sigma
    return _interpolate_and_perturb(
        alpha=alpha,
        sigma=sigma,
        params_init=module_variables_init,
        params_final=module_variables_final,
        objective_fn=objective_fn,
        loss_threshold_condition=functools.partial(
            loss_threshold_condition, reference_error=final_objective_value),
        normalize_error=normalize_error,
        num_samples_per_iteration=num_samples_per_iteration,
    )

  (threshold_conditions, interpolated_and_perturbed_losses,
   interpolated_and_perturbed_norms) = tf.map_fn(
       _evaluate_alpha_sigma,
       elems=(alphas, sigmas),
       dtype=(tf.bool, tf.float32, tf.float32),
   )

  masked_interpolated_and_perturbed_norms = tf.where(
      threshold_conditions, interpolated_and_perturbed_norms,
      tf.ones_like(interpolated_and_perturbed_norms) * np.inf)
  idx_min = tf.math.argmin(masked_interpolated_and_perturbed_norms)
  (loss_final, norm_final, alpha_final,
   sigma_final) = (interpolated_and_perturbed_losses[idx_min],
                   interpolated_and_perturbed_norms[idx_min], alphas[idx_min],
                   sigmas[idx_min])

  return ModuleCriticalityAnalysis(
      criticality_score=norm_final,
      alpha=alpha_final,
      sigma=sigma_final,
      loss_value=loss_final,
      num_samples_per_iteration=num_samples_per_iteration,
      alpha_grid_size=alpha_grid_size,
      sigma_grid_size=sigma_grid_size,
      sigma_ratio=sigma_ratio,
      initial_objective_value=initial_objective_value,
      final_objective_value=final_objective_value,
  )


NetworkCriticalityAnalysis = collections.namedtuple(
    'NetworkCriticalityAnalysis', (
        'network_criticality_score',
        'module_criticality_analyses',
    ))


def compute_network_criticality(
    parameters,
    module_objective_fn,
    init_loop_variables_mapping,
    final_loop_variables_mapping,
):
  """Compute the criticality of a network parameterized by `parameters`."""

  def _module_criticality_by_ref(layer_ref):
    return compute_module_criticality(
        functools.partial(
            module_objective_fn,
            module_variable_refs_=(layer_ref,),
        ),
        (init_loop_variables_mapping[layer_ref],),
        (final_loop_variables_mapping[layer_ref],),
    )

  layer_refs = tuple(layer.experimental_ref() for layer in parameters)
  module_criticality_analyses = map(_module_criticality_by_ref, layer_refs)
  network_criticality = sum(
      module_criticality_analysis.criticality_score
      for module_criticality_analysis in module_criticality_analyses)

  return NetworkCriticalityAnalysis(
      network_criticality_score=network_criticality,
      module_criticality_analyses=dict(
          zip(
              (layer.name for layer in parameters),
              module_criticality_analyses,
          )))
