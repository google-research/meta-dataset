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
"""Configuration classes for data processing.

Config classes that parametrize the behaviour of different stages of the data
processing pipeline, and are set up via `gin`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf


@gin.configurable
class DataConfig(object):
  """Common configuration options for creating data processing pipelines."""

  def __init__(
      self,
      image_height,
      shuffle_buffer_size,
      read_buffer_size_bytes,
      num_prefetch,
      eval_dumped_episodes_dir=None,
      eval_extra_dumped_episodes_dir=None,
      vtab_query_size_limit=500,
  ):
    """Initialize a DataConfig.

    Args:
      image_height: An integer, the desired height for the images output by the
        data pipeline. Images are squared and have 3 channels (RGB), so each
        image will have shape [image_height, image_height, 3],
      shuffle_buffer_size: An integer, the size of the example buffer in the
        tf.data.Dataset.shuffle operations (there is typically one shuffle per
        class in the episodic setting, one per dataset in the batch setting).
        Classes with fewer examples as this number are shuffled in-memory.
      read_buffer_size_bytes: An integer, the size (in bytes) of the read buffer
        for each tf.data.TFRecordDataset (there is typically one for each class
        of each dataset).
      num_prefetch: int, the number of examples to prefetch for each class of
        each dataset. Prefetching occurs just after the class-specific Dataset
        object is constructed. If < 1, no prefetching occurs.
      eval_dumped_episodes_dir: str or None, if given reads the the evaluation
        episodes from given folder in episodic mode.
      eval_extra_dumped_episodes_dir: str or None, if given reads the the
        evaluation episodes for the "extra datasets" (mnist, cifar10, cifar100
        and unshuffled_traffic_sign) from this folder.
      vtab_query_size_limit: int, limits the size of the query sets for VTAB
        episodes.
    """
    self.image_height = image_height
    self.shuffle_buffer_size = shuffle_buffer_size
    self.read_buffer_size_bytes = read_buffer_size_bytes
    self.num_prefetch = num_prefetch
    self.eval_dumped_episodes_dir = eval_dumped_episodes_dir
    self.eval_extra_dumped_episodes_dir = eval_extra_dumped_episodes_dir
    self.vtab_query_size_limit = vtab_query_size_limit


@gin.configurable
class DataAugmentation(object):
  """Configurations for performing data augmentation."""

  def __init__(self, enable_jitter, jitter_amount, enable_gaussian_noise,
               gaussian_noise_std):
    """Initialize a DataAugmentation.

    Args:
      enable_jitter: bool whether to use image jitter (pad each image using
        reflection along x and y axes and then random crop).
      jitter_amount: amount (in pixels) to pad on all sides of the image.
      enable_gaussian_noise: bool whether to use additive Gaussian noise.
      gaussian_noise_std: Standard deviation of the Gaussian distribution.
    """
    self.enable_jitter = enable_jitter
    self.jitter_amount = jitter_amount
    self.enable_gaussian_noise = enable_gaussian_noise
    self.gaussian_noise_std = gaussian_noise_std


@gin.configurable
class EpisodeDescriptionConfig(object):
  """Configuration options for episode characteristics."""

  def __init__(self,
               num_ways,
               num_support,
               num_query,
               min_ways,
               max_ways_upper_bound,
               max_num_query,
               max_support_set_size,
               max_support_size_contrib_per_class,
               min_log_weight,
               max_log_weight,
               ignore_dag_ontology,
               ignore_bilevel_ontology,
               ignore_hierarchy_probability,
               simclr_episode_fraction,
               min_examples_in_class=0,
               num_unique_descriptions=0,
               episode_description_switch_frequency=1):
    """Initialize a EpisodeDescriptionConfig.

    This is used in sampling.py in Trainer and in EpisodeDescriptionSampler to
    determine the parameters of episode creation relating to the ways and shots.

    Args:
      num_ways: Integer, fixes the number of classes ("ways") to be used in each
        episode. None leads to variable way.
      num_support: An integer, a tuple of two integers, or None. In the first
        case, the number of examples per class in the support set. In the second
        case, the range from which to sample the number of examples per class in
        the support set. Both of these cases would yield class-balanced
        episodes, i.e. all classes have the same number of support examples.
        Finally, if None, the number of support examples will vary both within
        each episode (introducing class imbalance) and across episodes.
      num_query: Integer, fixes the number of examples for each class in the
        query set.
      min_ways: Integer, the minimum value when sampling ways.
      max_ways_upper_bound: Integer, the maximum value when sampling ways. Note
        that the number of available classes acts as another upper bound.
      max_num_query: Integer, the maximum number of query examples per class.
      max_support_set_size: Integer, the maximum size for the support set.
      max_support_size_contrib_per_class: Integer, the maximum contribution for
        any given class to the support set size.
      min_log_weight: Float, the minimum log-weight to give to any particular
        class when determining the number of support examples per class.
      max_log_weight: Float, the maximum log-weight to give to any particular
        class.
      ignore_dag_ontology: Whether to ignore ImageNet's DAG ontology when
        sampling classes from it. This has no effect if ImageNet is not part of
        the benchmark.
      ignore_bilevel_ontology: Whether to ignore Omniglot's DAG ontology when
        sampling classes from it. This has no effect if Omniglot is not part of
        the benchmark.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.
      simclr_episode_fraction: Float, fraction of episodes that will be
        converted to SimCLR Episodes as described in the CrossTransformers
        paper.
      min_examples_in_class: An integer, the minimum number of examples that a
        class has to contain to be considered. All classes with fewer examples
        will be ignored. 0 means no classes are ignored, so having classes with
        no examples may trigger errors later. For variable shots, a value of 2
        makes it sure that there are at least one support and one query samples.
        For fixed shots, you could set it to `num_support + num_query`.
      num_unique_descriptions: An integer, the number of unique episode
        descriptions to use. If set to x > 0, x episode descriptions are
        pre-generated, and repeatedly iterated over. This is especially helpful
        when running on TPUs as it avoids the use of
        tf.data.Dataset.from_generator. If set to x = 0, no such upper bound on
        number of unique episode descriptions is set. Note that this is the
        number of unique episode descriptions _for each source dataset_, not
        total unique episode descriptions.
      episode_description_switch_frequency: An integer, the number of calls to
        Dataset before the class IDs are switched. This is useful in case of
        finite number of unrolls that are trained on the same class IDs, but
        using different samples. Default: 1 (i.e. episode switches every call).

    Raises:
      RuntimeError: if incompatible arguments are passed.
    """
    arg_groups = {
        'num_ways': (num_ways, ('min_ways', 'max_ways_upper_bound'),
                     (min_ways, max_ways_upper_bound)),
        'num_query': (num_query, ('max_num_query',), (max_num_query,)),
        'num_support':
            (num_support,
             ('max_support_set_size', 'max_support_size_contrib_per_class',
              'min_log_weight', 'max_log_weight'),
             (max_support_set_size, max_support_size_contrib_per_class,
              min_log_weight, max_log_weight)),
    }

    for first_arg_name, values in arg_groups.items():
      first_arg, required_arg_names, required_args = values
      if ((first_arg is None) and any(arg is None for arg in required_args)):
        # Get name of the nones
        none_arg_names = [
            name for var, name in zip(required_args, required_arg_names)
            if var is None
        ]
        raise RuntimeError(
            'The following arguments: %s can not be None, since %s is None. '
            'Arguments can be set up with gin, for instance by providing '
            '`--gin_file=learn/gin/setups/data_config.gin` or calling '
            '`gin.parse_config_file(...)` in the code. Please ensure the '
            'following gin arguments of EpisodeDescriptionConfig are set: '
            '%s' % (none_arg_names, first_arg_name, none_arg_names))

    self.num_ways = num_ways
    self.num_support = num_support
    self.num_query = num_query
    self.min_ways = min_ways
    self.max_ways_upper_bound = max_ways_upper_bound
    self.max_num_query = max_num_query
    self.max_support_set_size = max_support_set_size
    self.max_support_size_contrib_per_class = max_support_size_contrib_per_class
    self.min_log_weight = min_log_weight
    self.max_log_weight = max_log_weight
    self.ignore_dag_ontology = ignore_dag_ontology
    self.ignore_bilevel_ontology = ignore_bilevel_ontology
    self.ignore_hierarchy_probability = ignore_hierarchy_probability
    self.simclr_episode_fraction = simclr_episode_fraction
    self.min_examples_in_class = min_examples_in_class
    self.num_unique_descriptions = num_unique_descriptions
    self.episode_description_switch_frequency = episode_description_switch_frequency

  @property
  def max_ways(self):
    """Returns the way (maximum way if variable) of the episode."""
    return self.num_ways or self.max_ways_upper_bound
