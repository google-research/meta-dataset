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

r"""Meta-Dataset data sources.

To generate the tfrecord files associated with all data sources and store them
in ~/tensorflow_datasets/meta_dataset, from this directory run

tfds build --datasets=meta_dataset --manual_dir=<MANUAL_DIR>

where <MANUAL_DIR> is the directory where the `ILSVRC2012_img_train.tar` file
was downloaded (see `MetaDataset.MANUAL_DOWNLOAD_INSTRUCTIONS`).

"""

import json
import os
from typing import Optional, Tuple

from meta_dataset import dataset_conversion
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data.tfds import constants
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_SPECS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(dataset_conversion.__file__)),
    'dataset_specs')


class MetaDatasetConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Meta-Dataset."""

  def __init__(self, name, dataset_spec_prefixes, filenames, manual_filenames,
               generate_examples_fn, **kwargs):
    self.dataset_spec_prefixes = dataset_spec_prefixes
    self.filenames = filenames
    self.manual_filenames = manual_filenames
    self.generate_examples_fn = generate_examples_fn
    super(MetaDatasetConfig, self).__init__(
        name=name, version=tfds.core.Version('1.0.0'), **kwargs)


class MetaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Meta-Dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial implementation',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  This dataset requires you to download the ILSVRC 2012 training data
  (`ILSVRC2012_img_train.tar`) manually into `download_config.manual_dir`
  (defaults to `~/tensorflow_datasets/downloads/manual/`). You need to register
  on http://www.image-net.org/download-images in order to get the link to
  download the dataset.
  """

  BUILDER_CONFIGS = [MetaDatasetConfig(**builder_config_kwargs)
                     for builder_config_kwargs in constants.BUILDER_CONFIGS]

  def as_class_dataset(
      self,
      md_version,
      meta_split,
      relative_label,
      shuffle_buffer_size = 0,
      shuffle_seed = None,
      num_prefetch = 0,
      **as_dataset_kwargs):
    """Creates a class dataset.

    Args:
      md_version: Meta-Dataset version in {'v1', 'v2'}.
      meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
      relative_label: label, relative to the meta-split.
      shuffle_buffer_size: shuffle buffer size for the class dataset.
      shuffle_seed: shuffle buffer random seed.
      num_prefetch: number of examples to prefetch.
      **as_dataset_kwargs: kwargs passed to the `as_dataset` method.

    Returns:
      The class dataset.

    Raises:
      ValueError, if the source does not provide `meta_split` in `md_version`.
    """
    start, stop = self.get_start_stop(md_version, meta_split, relative_label)
    class_dataset = self.as_dataset(
        split=f'all_classes[{start}:{stop}]', **as_dataset_kwargs)
    if num_prefetch > 0:
      class_dataset = class_dataset.prefetch(num_prefetch)
    shuffle_buffer_size = min(stop - start, shuffle_buffer_size)
    if shuffle_buffer_size > 1:
      class_dataset = class_dataset.shuffle(
          buffer_size=shuffle_buffer_size,
          seed=shuffle_seed,
          reshuffle_each_iteration=True)
    return class_dataset

  def as_full_ways_dataset(
      self,
      md_version,
      meta_split,
      **as_dataset_kwargs):
    """Creates a full-ways dataset.

    Args:
      md_version: Meta-Dataset version in {'v1', 'v2'}.
      meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
      **as_dataset_kwargs: kwargs passed to the `as_dataset` method.

    Returns:
      The full-ways dataset.

    Raises:
      ValueError, if the source does not provide `meta_split` in `md_version`.
    """
    start, stop = self.get_start_stop(md_version, meta_split)
    full_ways_dataset = self.as_dataset(
        split=f'all_classes[{start}:{stop}]', **as_dataset_kwargs)
    return full_ways_dataset

  def get_start_stop(
      self,
      md_version,
      meta_split,
      relative_label = None):
    """Returns the start and stop indices for a class or split.

    Args:
      md_version: Meta-Dataset version in {'v1', 'v2'}.
      meta_split: meta-split (case-insensitive) in {'train', 'valid', 'test'}.
      relative_label: label, relative to the meta-split. If None, returns the
        start and stop indices for the entire split.

    Returns:
      (start, stop) indices for the class/split.

    Raises:
      ValueError, if the source does not provide `meta_split` in `md_version`.
    """
    full_ways = relative_label is None
    slice_dict = self.info.metadata['meta_split_slices' if full_ways else
                                    'relative_class_slices']

    if md_version not in slice_dict:
      raise ValueError(f'source {self.builder_config.name} '
                       f'not present in version {md_version}')

    slice_dict = slice_dict[md_version]

    meta_split = meta_split.lower()
    if not slice_dict[meta_split]:
      raise ValueError(f'source {self.builder_config.name} does not offer a '
                       f'{meta_split} split in version {md_version}')

    if full_ways:
      return slice_dict[meta_split]
    else:
      # `self.info.metadata` is read from a `metadata.json` file stored on disk
      # alongside the tfrecord files. In JSON, keys are always of type str, so
      # we cast them to int before indexing with `relative_label`.
      #
      # We cast the keys to int instead of casting `relative_label` to str,
      # because in the event that the builder's data path is incorrect the
      # metadata will be created on-the-fly and the slice keys will be of type
      # int, and this block should be robust to that to avoid a cryptic
      # exception being raised.
      slices = {int(k): v for k, v in slice_dict[meta_split].items()}
      return slices[relative_label]

  def _info(self):
    """Returns the dataset metadata."""
    dataset_spec_prefixes = self.builder_config.dataset_spec_prefixes
    dataset_specs = {}
    meta_split_slices = {}
    relative_class_slices = {}
    class_names = None
    total_num_examples = None

    for version, dataset_spec_prefix in dataset_spec_prefixes.items():
      file_path = os.path.join(DATASET_SPECS_DIR,
                               f'{dataset_spec_prefix}_dataset_spec.json')

      dataset_spec_dict = json.loads(tfds.core.as_path(file_path).read_text())
      # For ILSVRC2012, slicing entire meta-splits (i.e. "batch" training)
      # is greatly simplified if we adopt V1's label-to-name mapping. For V2
      # this doesn't change anything, since all classes are in the meta-training
      # split, but for V1 this means that we can access any meta-split with only
      # one slicing instruction.
      if dataset_spec_prefix == 'ilsvrc_2012_v2':
        dataset_spec_dict['class_names'] = dataset_specs['v1']['class_names']
      dataset_specs[version] = dataset_spec_dict
      dataset_spec = dataset_spec_lib.as_dataset_spec(dataset_spec_dict)

      # Compute start and stop indices of (relative) class-level slices in the
      # data source.
      meta_split_class_ids = {
          meta_split: dataset_spec.get_classes(getattr(learning_spec.Split,
                                                       meta_split.upper()))
          for meta_split in ('train', 'valid', 'test')
      }
      images_per_class = {
          meta_split: [dataset_spec.get_total_images_per_class(class_id)
                       for class_id in class_ids]
          for meta_split, class_ids in meta_split_class_ids.items()
      }
      cumulative_per_class = np.cumsum(
          [0] + sum(images_per_class.values(), [])).tolist()
      start_stop_per_class = zip(cumulative_per_class[:-1],
                                 cumulative_per_class[1:])
      class_slices = dict(enumerate(start_stop_per_class))
      relative_class_slices[version] = {
          meta_split: {i_rel: class_slices[i_abs] for i_rel, i_abs in
                       enumerate(class_ids)}
          for meta_split, class_ids in meta_split_class_ids.items()
      }

      # Compute start and stop indices of split-level slices in the data source.
      images_per_split = {k: sum(v) for k, v in images_per_class.items()}
      cumulative_per_meta_split = np.cumsum(
          [0] + list(images_per_split.values())).tolist()
      start_stop_per_meta_split = zip(cumulative_per_meta_split[:-1],
                                      cumulative_per_meta_split[1:])
      meta_split_slices[version] = {
          meta_split: (None if stop == start else (start, stop))
          for meta_split, (start, stop) in zip(meta_split_class_ids.keys(),
                                               start_stop_per_meta_split)
      }

      # Check for consistency between dataset specifications of different
      # versions.
      if total_num_examples and cumulative_per_class[-1] != total_num_examples:
        raise ValueError('dataset specifications have different numbers of '
                         'examples')
      total_num_examples = total_num_examples or cumulative_per_class[-1]

      if class_names and list(dataset_spec.class_names.values()) != class_names:
        raise ValueError('dataset specifications have different classes')
      class_names = class_names or list(dataset_spec.class_names.values())

    return tfds.core.DatasetInfo(
        builder=self,
        description='Meta-Dataset data sources.',
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'format': tfds.features.Text(),
            'filename': tfds.features.Text(),
            'label': tfds.features.ClassLabel(num_classes=len(class_names)),
            'class_name': tfds.features.Text(),
        }),
        metadata=tfds.core.MetadataDict(
            meta_split_slices=meta_split_slices,
            relative_class_slices=relative_class_slices,
            dataset_specs=dataset_specs,
            class_names=class_names,
            total_num_examples=total_num_examples,
            class_slices=class_slices,
        ),
        # This is important, since we rely on examples of the same class being
        # contiguous to build class-specific datasets with the TFDS slicing API.
        disable_shuffling=True,
        supervised_keys=('image', 'label'),
        homepage='https://github.com/google-research/meta-dataset',
        citation=constants.CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    paths = dl_manager.download_and_extract(self.builder_config.filenames)
    paths.update(  # pytype: disable=attribute-error  # gen-stub-imports
        {name: dl_manager.manual_dir / manual_name
         for name, manual_name in self.builder_config.manual_filenames.items()}
    )
    return {
        'all_classes': self._generate_examples(
            metadata=self.info.metadata,
            paths=paths,
            generate_examples_fn=self.builder_config.generate_examples_fn)
    }

  def _generate_examples(self, metadata, paths, generate_examples_fn):
    """Yields examples."""
    return generate_examples_fn(metadata=metadata, paths=paths)
