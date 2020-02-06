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
# pyformat: disable
r"""Main file for converting the datasets used in the benchmark into records.

Example command to convert dataset omniglot:
# pylint: disable=line-too-long
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=<path/to/omniglot> \
  --records_root=<path/to/records> \
  --splits_root=<path/to/splits>
# pylint: enable=line-too-long
"""
# pyformat: enable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from meta_dataset.dataset_conversion import dataset_to_records
import tensorflow.compat.v1 as tf

tf.flags.DEFINE_string(
    'mini_imagenet_records_dir',
    # This dataset is for diagnostic purposes only, which is why we want to
    # store it in a different location than the other datasets.
    '',
    'The path to store the tf.Records of MiniImageNet.')

tf.flags.DEFINE_string('dataset', 'omniglot',
                       'The name of the dataset to convert to records.')

FLAGS = tf.flags.FLAGS


class ConverterArgs(
    collections.namedtuple('ConverterArgs', 'data_root, long_name')):
  """Arguments to be passed to a DatasetConverter's constructor.

  Attributes:
    data_root: string, path to the root of the dataset.
    long_name: string, dataset name in longer or capitalized form.
  """


def _dataset_name_to_converter_and_args(flags=FLAGS):
  """Returns a dict mapping dataset name to (converter class, arguments).

  This (converter class, arguments) pair will be used to build the corresponding
  DatasetConverter object.

  Args:
    flags: A tf.flags.FlagValues object, by default tf.flags.FLAGS, containing
      the data_root of the datasets.
  """
  # The dictionary is built inside a function, rather than at the module
  # top-level, because the FLAGS are not available at import time.
  return {
      # Datasets in the same order as reported in the article.
      'ilsvrc_2012': (dataset_to_records.ImageNetConverter,
                      ConverterArgs(
                          data_root=flags.ilsvrc_2012_data_root,
                          long_name='ImageNet ILSVRC-2012')),
      'omniglot': (dataset_to_records.OmniglotConverter,
                   ConverterArgs(
                       data_root=flags.omniglot_data_root,
                       long_name='Omniglot')),
      'aircraft': (dataset_to_records.AircraftConverter,
                   ConverterArgs(
                       data_root=flags.aircraft_data_root,
                       long_name='FGVC-Aircraft Benchmark')),
      'cu_birds': (dataset_to_records.CUBirdsConverter,
                   ConverterArgs(
                       data_root=flags.cu_birds_data_root,
                       long_name='CU Birds')),
      'dtd': (dataset_to_records.DTDConverter,
              ConverterArgs(
                  data_root=flags.dtd_data_root,
                  long_name='Describable Textures Dataset')),
      'quickdraw': (dataset_to_records.QuickdrawConverter,
                    ConverterArgs(
                        data_root=flags.quickdraw_data_root,
                        long_name='Quick, Draw!')),
      'fungi': (dataset_to_records.FungiConverter,
                ConverterArgs(
                    data_root=flags.fungi_data_root,
                    long_name='fungi 2018 FGVCx')),
      'vgg_flower': (dataset_to_records.VGGFlowerConverter,
                     ConverterArgs(
                         data_root=flags.vgg_flower_data_root,
                         long_name='VGG Flower')),
      'traffic_sign': (dataset_to_records.TrafficSignConverter,
                       ConverterArgs(
                           data_root=flags.traffic_sign_data_root,
                           long_name='Traffic Sign')),
      'mscoco':
          (dataset_to_records.MSCOCOConverter,
           ConverterArgs(data_root=flags.mscoco_data_root, long_name='MSCOCO')),
      # Diagnostics-only dataset
      'mini_imagenet': (dataset_to_records.MiniImageNetConverter,
                        ConverterArgs(
                            data_root=flags.mini_imagenet_data_root,
                            long_name='MiniImageNet')),
  }


def main(argv):
  del argv

  dataset_name_to_converter_and_args = _dataset_name_to_converter_and_args(
      flags=FLAGS)
  if FLAGS.dataset not in dataset_name_to_converter_and_args:
    raise NotImplementedError(
        'Dataset {} not supported. Supported datasets are {}'.format(
            FLAGS.dataset, sorted(dataset_name_to_converter_and_args.keys())))

  converter_class, converter_args = dataset_name_to_converter_and_args[
      FLAGS.dataset]
  if FLAGS.dataset == 'mini_imagenet':
    # MiniImagenet is for diagnostics purposes only, do not use the default
    # records_path to avoid confusion.
    records_path = FLAGS.mini_imagenet_records_dir
  else:
    records_path = None
  converter = converter_class(
      name=FLAGS.dataset,
      data_root=converter_args.data_root,
      records_path=records_path)
  logging.info('Creating %s specification and records in directory %s...',
               converter_args.long_name, converter.records_path)
  converter.convert_dataset()


if __name__ == '__main__':
  tf.app.run(main)
