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

"""The Meta-Dataset TFDS API."""

import concurrent.futures
import functools
from typing import Optional

from absl import logging
from meta_dataset.data import config as config_lib
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import reader
from meta_dataset.data import sampling
from meta_dataset.data import tfds as meta_dataset_lib
import tensorflow as tf
import tensorflow.random.experimental as tfre
import tensorflow_datasets as tfds


_NUM_WORKERS = 10

create_rng_state = functools.partial(tf.random.create_rng_state, alg='threefry')


def episode_dataset(
    builder,
    md_version,
    meta_split,
    source_id = None,
    shuffle_seed = None,
    **as_dataset_kwargs):
  """Creates an episode dataset.

  This function creates an episode dataset for a single source. For multi-source
  pipelines, use the `meta_dataset` function.

  Args:
    builder: Meta-Dataset builder.
    md_version: Meta-Dataset md_version in {'v1', 'v2'}.
    meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
    source_id: source ID to output alongside episodes, if not None.
    shuffle_seed: class dataset shuffle buffer seed.
    **as_dataset_kwargs: kwargs passed to the `as_dataset` method.

  Returns:
    The episode dataset.
  """

  dataset_spec = dataset_spec_lib.as_dataset_spec(
      builder.info.metadata['dataset_specs'][md_version])
  data_config = config_lib.DataConfig()
  episode_description_config = config_lib.EpisodeDescriptionConfig()

  def _as_class_dataset(args):
    relative_label, seed = args
    seed = None if seed is None else seed[0]
    class_dataset = builder.as_class_dataset(
        md_version=md_version,
        meta_split=meta_split,
        relative_label=relative_label,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        shuffle_seed=seed,
        num_prefetch=data_config.num_prefetch,
        decoders={'image': tfds.decode.SkipDecoding()},
        read_config=tfds.ReadConfig(try_autocache=False),
        as_supervised=True,
        **as_dataset_kwargs
    ).repeat()
    logging.info("Created class %d for %s's meta-%s split",
                 relative_label, builder.builder_config.name, meta_split)
    return class_dataset

  num_classes = len(dataset_spec.get_classes(getattr(learning_spec.Split,
                                                     meta_split.upper())))
  # If a shuffle seed is passed, we split it into `num_classes` independent
  # shufle seeds so that each class datasets' shuffle buffer is seeded
  # differently. Sharing random seeds across shuffle buffers is considered bad
  # practice because it can introduce correlations across random sequences of
  # examples for different classes.
  shuffle_seeds = ([None] * num_classes if shuffle_seed is None else
                   tfre.stateless_split(create_rng_state(shuffle_seed),
                                        num_classes))
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_NUM_WORKERS) as executor:
    class_datasets = list(executor.map(_as_class_dataset,
                                       enumerate(shuffle_seeds)))

  placeholder_id = reader.PLACEHOLDER_CLASS_ID
  class_datasets.append(tf.data.Dataset.zip((
      tf.data.Dataset.from_tensors(b'').repeat(),
      tf.data.Dataset.from_tensors(tf.cast(placeholder_id, tf.int64)).repeat()
  )))

  sampler = sampling.EpisodeDescriptionSampler(
      dataset_spec,
      getattr(learning_spec.Split, meta_split.upper()),
      episode_description_config,
      use_dag_hierarchy=builder.builder_config.name == 'ilsvrc_2012',
      use_bilevel_hierarchy=builder.builder_config.name == 'omniglot')
  chunk_sizes = sampler.compute_chunk_sizes()

  dataset = tf.data.Dataset.choose_from_datasets(
      class_datasets,
      tf.data.Dataset.from_generator(
          functools.partial(
              reader.episode_representation_generator,
              dataset_spec=dataset_spec,
              split=getattr(learning_spec.Split, meta_split.upper()),
              pool=None,
              sampler=sampler),
          tf.int64,
          tf.TensorShape([None, 2]),
      ).map(reader.decompress_episode_representation).unbatch()
  ).batch(
      sum(chunk_sizes)
  ).prefetch(
      1
  ).map(
      functools.partial(
          pipeline.process_episode,
          chunk_sizes=chunk_sizes,
          image_size=data_config.image_height,
          simclr_episode_fraction=(
              episode_description_config.simclr_episode_fraction))
  )

  if source_id is not None:
    dataset = tf.data.Dataset.zip(
        (dataset, tf.data.Dataset.from_tensors(source_id).repeat()))

  return dataset


def full_ways_dataset(
    md_source,
    md_version,
    meta_split,
    shuffle_files=True,
    read_config=None,
    data_dir=None,
    version=None,
    **as_dataset_kwargs):
  """Creates a full-ways dataset.

  Here, "full-ways" means that the label space is constructed from all of
  `md_source`'s `meta_split` classes. For instance, calling `full_ways_dataset`
  with `md_source='aircraft'` and `meta_split='valid'` returns a dataset with
  all of Aircraft's validation classes.

  Args:
    md_source: data source from which to construct the full-ways dataset.
    md_version: Meta-Dataset md_version in {'v1', 'v2'}.
    meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
    shuffle_files: WRITEME.
    read_config: WRITEME.
    data_dir: TFDS data directory.
    version: dataset version at which to load the data. Note that this refers
      to the dataset implementation version, and is **not** the same as
      the benchmark verion (either 'v1' or 'v2').
    **as_dataset_kwargs: kwargs passed to the `as_dataset` method.

  Returns:
    The full-ways dataset.
  """
  builder = meta_dataset_lib.MetaDataset(
      data_dir=data_dir,
      config=md_source,
      version=version
  )
  start, stop = builder.get_start_stop(md_version, meta_split)
  read_config = read_config or tfds.ReadConfig(
      interleave_cycle_length=builder.info.splits[
          f'all_classes[{start}:{stop}]'].num_shards,
      interleave_block_length=4,
      enable_ordering_guard=False
  )
  return builder.as_full_ways_dataset(
      md_version=md_version,
      meta_split=meta_split,
      shuffle_files=shuffle_files,
      read_config=read_config,
      **as_dataset_kwargs
  )


def meta_dataset(
    md_sources,
    md_version,
    meta_split,
    shuffle_seed=None,
    source_sampling_seed=None,
    data_dir=None,
    version=None,
    **as_dataset_kwargs):
  """Creates a Meta-Dataset dataset.

  This function creates an episode dataset for all sources in `md_sources`. For
  single-source pipelines, use the `episode_dataset` function.

  Args:
    md_sources: data sources from which to draw episodes.
    md_version: Meta-Dataset md_version in {'v1', 'v2'}.
    meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
    shuffle_seed: class dataset shuffle buffer seed.
    source_sampling_seed: random seed for source sampling.
    data_dir: TFDS data directory.
    version: dataset version at which to load the data. Note that this refers
      to the dataset implementation version, and is **not** the same as
      the benchmark verion (either 'v1' or 'v2').
    **as_dataset_kwargs: kwargs passed to the `as_dataset` method.

  Returns:
    The episode dataset.
  """
  episode_datasets = []
  # If a shuffle seed is passed, we split it into `len(md_sources)` independent
  # shufle seeds so that each episode dataset's randomness shuffle buffer is
  # seeded differently.
  shuffle_seeds = ([None] * len(md_sources) if shuffle_seed is None else
                   tfre.stateless_split(create_rng_state(shuffle_seed),
                                        len(md_sources)))
  for source_id, (source, seed) in enumerate(zip(md_sources, shuffle_seeds)):
    seed = None if seed is None else seed[0]
    episode_datasets.append(
        episode_dataset(
            builder=meta_dataset_lib.MetaDataset(
                data_dir=data_dir,
                config=source,
                version=version),
            md_version=md_version,
            meta_split=meta_split,
            source_id=source_id,
            shuffle_seed=seed,
            **as_dataset_kwargs)
    )
  return tf.data.Dataset.sample_from_datasets(
      episode_datasets, seed=source_sampling_seed)
