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
"""Interfaces for dataset specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import logging
from meta_dataset import data
from meta_dataset.data import imagenet_specification
from meta_dataset.data import learning_spec
import numpy as np
import six
from six.moves import cPickle as pkl
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

# Global records root directory, for all datasets (except diagnostics).
tf.flags.DEFINE_string('records_root_dir', '',
                       'Root directory containing a subdirectory per dataset.')
FLAGS = tf.flags.FLAGS


def get_classes(split, classes_per_split):
  """Gets the sequence of class labels for a split.

  Class id's are returned ordered and without gaps.

  Args:
    split: A Split, the split for which to get classes.
    classes_per_split: Matches each Split to the number of its classes.

  Returns:
    The sequence of classes for the split.

  Raises:
    ValueError: An invalid split was specified.
  """
  num_classes = classes_per_split[split]

  # Find the starting index of classes for the given split.
  if split == learning_spec.Split.TRAIN:
    offset = 0
  elif split == learning_spec.Split.VALID:
    offset = classes_per_split[learning_spec.Split.TRAIN]
  elif split == learning_spec.Split.TEST:
    offset = (
        classes_per_split[learning_spec.Split.TRAIN] +
        classes_per_split[learning_spec.Split.VALID])
  else:
    raise ValueError('Invalid dataset split.')

  # Get a contiguous range of classes from split.
  return range(offset, offset + num_classes)


def _check_validity_of_restricted_classes_per_split(
    restricted_classes_per_split, classes_per_split):
  """Check the validity of the given restricted_classes_per_split.

  Args:
    restricted_classes_per_split: A dict mapping Split enums to the number of
      classes to restrict to for that split.
    classes_per_split: A dict mapping Split enums to the total available number
      of classes for that split.

  Raises:
    ValueError: if restricted_classes_per_split is invalid.
  """
  for split_enum, num_classes in restricted_classes_per_split.items():
    if split_enum not in [
        learning_spec.Split.TRAIN, learning_spec.Split.VALID,
        learning_spec.Split.TEST
    ]:
      raise ValueError('Invalid key {} in restricted_classes_per_split.'
                       'Valid keys are: learning_spec.Split.TRAIN, '
                       'learning_spec.Split.VALID, and '
                       'learning_spec.Split.TEST'.format(split_enum))
    if num_classes > classes_per_split[split_enum]:
      raise ValueError('restricted_classes_per_split can not specify a '
                       'number of classes greater than the total available '
                       'for that split. Specified {} for split {} but have '
                       'only {} available for that split.'.format(
                           num_classes, split_enum,
                           classes_per_split[split_enum]))


def get_total_images_per_class(data_spec, class_id=None, pool=None):
  """Returns the total number of images of a class in a data_spec and pool.

  Args:
    data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
    class_id: The class whose number of images will be returned. If this is
      None, it is assumed that the dataset has the same number of images for
      each class.
    pool: A string ('train' or 'test', optional) indicating which example-level
      split to select, if the current dataset has them.

  Raises:
    ValueError: when
      - no class_id specified and yet there is class imbalance, or
      - no pool specified when there are example-level splits, or
      - pool is specified but there are no example-level splits, or
      - incorrect value for pool.
    RuntimeError: the DatasetSpecification is out of date (missing info).
  """
  if class_id is None:
    if len(set(data_spec.images_per_class.values())) != 1:
      raise ValueError('Not specifying class_id is okay only when all classes'
                       ' have the same number of images')
    class_id = 0

  if class_id not in data_spec.images_per_class:
    raise RuntimeError('The DatasetSpecification should be regenerated, as '
                       'it does not have a non-default value for class_id {} '
                       'in images_per_class.'.format(class_id))
  num_images = data_spec.images_per_class[class_id]

  if pool is None:
    if isinstance(num_images, collections.Mapping):
      raise ValueError('DatasetSpecification {} has example-level splits, so '
                       'the "pool" argument has to be set (to "train" or '
                       '"test".'.format(data_spec.name))
  elif not data.POOL_SUPPORTED:
    raise NotImplementedError('Example-level splits or pools not supported.')

  return num_images


class BenchmarkSpecification(
    collections.namedtuple(
        'BenchmarkSpecification', 'name, image_shape, dataset_spec_list,'
        'has_dag_ontology, has_bilevel_ontology, splits_to_contribute')):
  """The specification of a benchmark, consisting of multiple datasets.

    Args:
      name: string, the name of the benchmark.
      image_shape: a sequence of dimensions representing the shape that each
        image (of each dataset) will be resized to.
      dataset_spec_list: a list of DatasetSpecification or
        HierarchicalDatasetSpecification instances for the benchmarks' datasets.
      has_dag_ontology: A list of bools, whose length is the same as the number
        of datasets in the benchmark. Its elements indicate whether each dataset
        (in the order specified in the benchmark_spec.dataset_spec_list list)
        has a DAG-structured ontology. In that case, the corresponding dataset
        specification must be an instance of HierarchicalDatasetSpecification.
      has_bilevel_ontology: A list of bools of the same length and structure as
        has_dag_ontology, this time indicating whether each dataset has a
        bi-level ontology (comprised of superclasses and subclasses). In that
        case, the corresponding dataset specification must be an instance of
        BiLevelDatasetSpecification.
      splits_to_contribute: A list of sets of the same length as the number of
        datasets in the benchmark. Each element is a set which can be one of
        {'train'}, {'valid'}, {'train', 'valid'} or {'test'} indicating which
        meta-splits the corresponding dataset should contribute to. Note that a
        dataset can not contribute to a split if it has zero classes assigned to
        that split. But we do have the option to ignore a dataset for a
        particular split even if it has a non-zero number of classes for it.
  """

  def __new__(cls, name, image_shape, dataset_spec_list, has_dag_ontology,
              has_bilevel_ontology, splits_to_contribute):
    if len(has_dag_ontology) != len(dataset_spec_list):
      raise ValueError('The length of has_dag_ontology must be the number of '
                       'datasets.')
    if len(has_bilevel_ontology) != len(dataset_spec_list):
      raise ValueError('The length of has_bilevel_ontology must be the number '
                       'of datasets.')
    if len(splits_to_contribute) != len(dataset_spec_list):
      raise ValueError('The length of splits_to_contribute must be the number '
                       'of datasets.')
    # Ensure that HierarchicalDatasetSpecification is used iff has_dag_ontology.
    for i, has_dag in enumerate(has_dag_ontology):
      if has_dag and not isinstance(dataset_spec_list[i],
                                    HierarchicalDatasetSpecification):
        raise ValueError('Dataset {} has dag ontology, but does not have a '
                         'hierarchical dataset specification.'.format(i))
      if not has_dag and isinstance(dataset_spec_list[i],
                                    HierarchicalDatasetSpecification):
        raise ValueError('Dataset {} has no dag ontology, but is represented '
                         'using a HierarchicalDatasetSpecification.'.format(i))
    # Ensure that BiLevelDatasetSpecification is used iff has_bilevel_ontology.
    for i, is_bilevel in enumerate(has_bilevel_ontology):
      if is_bilevel and not isinstance(dataset_spec_list[i],
                                       BiLevelDatasetSpecification):
        raise ValueError('Dataset {} has bilevel ontology, but does not have a '
                         'bilevel dataset specification.'.format(i))
      if not is_bilevel and isinstance(dataset_spec_list[i],
                                       BiLevelDatasetSpecification):
        raise ValueError(
            'Dataset {} has no bilevel ontology, but is '
            'represented using a BiLevelDatasetSpecification.'.format(i))
    # Check the validity of the given value for splits_to_contribute.
    valid_values = [{'train'}, {'valid'}, {'train', 'valid'}, {'test'}]
    for splits in splits_to_contribute:
      if splits not in valid_values:
        raise ValueError(
            'Found an invalid element: {} in splits_to_contribute. '
            'Valid elements are: {}'.format(splits, valid_values))
    # Ensure that no dataset is asked to contribute to a split for which it does
    # not have any classes.
    for dataset_spec, dataset_splits in zip(dataset_spec_list,
                                            splits_to_contribute):
      dataset_spec.initialize()
      if isinstance(dataset_spec, BiLevelDatasetSpecification):
        classes_per_split = dataset_spec.superclasses_per_split
      else:
        classes_per_split = dataset_spec.classes_per_split
      invalid_train_split = ('train' in dataset_splits and
                             not classes_per_split[learning_spec.Split.TRAIN])
      invalid_valid_split = ('valid' in dataset_splits and
                             not classes_per_split[learning_spec.Split.VALID])
      invalid_test_split = ('test' in dataset_splits and
                            not classes_per_split[learning_spec.Split.TEST])
      if invalid_train_split or invalid_valid_split or invalid_test_split:
        raise ValueError('A dataset can not contribute to a split if it has '
                         'no classes assigned to that split.')
    self = super(BenchmarkSpecification,
                 cls).__new__(cls, name, image_shape, dataset_spec_list,
                              has_dag_ontology, has_bilevel_ontology,
                              splits_to_contribute)
    return self


class DatasetSpecification(
    collections.namedtuple('DatasetSpecification',
                           ('name, classes_per_split, images_per_class, '
                            'class_names, path, file_pattern'))):
  """The specification of a dataset.

    Args:
      name: string, the name of the dataset.
      classes_per_split: a dict specifying the number of classes allocated to
        each split.
      images_per_class: a dict mapping each class id to its number of images.
        Usually, the number of images is an integer, but if the dataset has
        'train' and 'test' example-level splits (or "pools"), then it is a dict
        mapping a string (the pool) to an integer indicating how many examples
        are in that pool. E.g., the number of images could be {'train': 5923,
        'test': 980}.
      class_names: a dict mapping each class id to the corresponding class name.
      path: the path to the dataset's files.
      file_pattern: a string representing the naming pattern for each class's
        file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
        The first gap will be replaced by the class id in both cases, while in
        the latter case the second gap will be replaced with by a shard index,
        or one of 'train', 'valid' or 'test'. This offers support for multiple
        shards of a class' images if a class is too large, that will be merged
        later into a big pool for sampling, as well as different splits that
        will be treated as disjoint pools for sampling the support versus query
        examples of an episode.
  """

  def initialize(self, restricted_classes_per_split=None):
    """Initializes a DatasetSpecification.

    Args:
      restricted_classes_per_split: A dict that specifies for each split, a
        number to restrict its classes to. This number must be no greater than
        the total number of classes of that split. By default this is None and
        no restrictions are applied (all classes are used).

    Raises:
      ValueError: Invalid file_pattern provided.
    """
    # Check that the file_pattern adheres to one of the allowable forms
    if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
      raise ValueError('file_pattern must be either "{}.tfrecords" or '
                       '"{}_{}.tfrecords" to support shards or splits.')
    if restricted_classes_per_split is not None:
      _check_validity_of_restricted_classes_per_split(
          restricted_classes_per_split, self.classes_per_split)
      # Apply the restriction.
      for split, restricted_num_classes in restricted_classes_per_split.items():
        self.classes_per_split[split] = restricted_num_classes

  def get_total_images_per_class(self, class_id=None, pool=None):
    """Returns the total number of images for the specified class.

    Args:
      class_id: The class whose number of images will be returned. If this is
        None, it is assumed that the dataset has the same number of images for
        each class.
      pool: A string ('train' or 'test', optional) indicating which
        example-level split to select, if the current dataset has them.

    Raises:
      ValueError: when
        - no class_id specified and yet there is class imbalance, or
        - no pool specified when there are example-level splits, or
        - pool is specified but there are no example-level splits, or
        - incorrect value for pool.
      RuntimeError: the DatasetSpecification is out of date (missing info).
    """
    return get_total_images_per_class(self, class_id, pool=pool)

  def get_classes(self, split):
    """Gets the sequence of class labels for a split.

    Labels are returned ordered and without gaps.

    Args:
      split: A Split, the split for which to get classes.

    Returns:
      The sequence of classes for the split.

    Raises:
      ValueError: An invalid split was specified.
    """
    return get_classes(split, self.classes_per_split)

  def to_dict(self):
    """Returns a dictionary for serialization to JSON.

    Each member is converted to an elementary type that can be serialized to
    JSON readily.
    """
    # Start with the dict representation of the namedtuple
    ret_dict = self._asdict()
    # Add the class name for reconstruction when deserialized
    ret_dict['__class__'] = self.__class__.__name__
    # Convert Split enum instances to their name (string)
    ret_dict['classes_per_split'] = {
        split.name: count
        for split, count in six.iteritems(ret_dict['classes_per_split'])
    }
    # Convert binary class names to unicode strings if necessary
    class_names = {}
    for class_id, name in six.iteritems(ret_dict['class_names']):
      if isinstance(name, six.binary_type):
        name = name.decode()
      elif isinstance(name, np.integer):
        name = six.text_type(name)
      class_names[class_id] = name
    ret_dict['class_names'] = class_names
    return ret_dict


class BiLevelDatasetSpecification(
    collections.namedtuple('BiLevelDatasetSpecification',
                           ('name, superclasses_per_split, '
                            'classes_per_superclass, images_per_class, '
                            'superclass_names, class_names, path, '
                            'file_pattern'))):
  """The specification of a dataset that has a two-level hierarchy.

    Args:
      name: string, the name of the dataset.
      superclasses_per_split: a dict specifying the number of superclasses
        allocated to each split.
      classes_per_superclass: a dict specifying the number of classes in each
        superclass.
      images_per_class: a dict mapping each class id to its number of images.
      superclass_names: a dict mapping each superclass id to its name.
      class_names: a dict mapping each class id to the corresponding class name.
      path: the path to the dataset's files.
      file_pattern: a string representing the naming pattern for each class's
        file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
        The first gap will be replaced by the class id in both cases, while in
        the latter case the second gap will be replaced with by a shard index,
        or one of 'train', 'valid' or 'test'. This offers support for multiple
        shards of a class' images if a class is too large, that will be merged
        later into a big pool for sampling, as well as different splits that
        will be treated as disjoint pools for sampling the support versus query
        examples of an episode.
  """

  def initialize(self, restricted_classes_per_split=None):
    """Initializes a DatasetSpecification.

    Args:
      restricted_classes_per_split: A dict that specifies for each split, a
        number to restrict its classes to. This number must be no greater than
        the total number of classes of that split. By default this is None and
        no restrictions are applied (all classes are used).

    Raises:
      ValueError: Invalid file_pattern provided
    """
    # Check that the file_pattern adheres to one of the allowable forms
    if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
      raise ValueError('file_pattern must be either "{}.tfrecords" or '
                       '"{}_{}.tfrecords" to support shards or splits.')
    if restricted_classes_per_split is not None:
      # Create a dict like classes_per_split of DatasetSpecification.
      classes_per_split = {}
      for split in self.superclasses_per_split.keys():
        num_split_classes = self._count_classes_in_superclasses(
            self.get_superclasses(split))
        classes_per_split[split] = num_split_classes

      _check_validity_of_restricted_classes_per_split(
          restricted_classes_per_split, classes_per_split)
    # The restriction in this case is applied in get_classes() below.
    self.restricted_classes_per_split = restricted_classes_per_split

  def get_total_images_per_class(self, class_id=None, pool=None):
    """Returns the total number of images for the specified class.

    Args:
      class_id: The class whose number of images will be returned. If this is
        None, it is assumed that the dataset has the same number of images for
        each class.
      pool: A string ('train' or 'test', optional) indicating which
        example-level split to select, if the current dataset has them.

    Raises:
      ValueError: when
        - no class_id specified and yet there is class imbalance, or
        - no pool specified when there are example-level splits, or
        - pool is specified but there are no example-level splits, or
        - incorrect value for pool.
      RuntimeError: the DatasetSpecification is out of date (missing info).
    """
    return get_total_images_per_class(self, class_id, pool=pool)

  def get_superclasses(self, split):
    """Gets the sequence of superclass labels for a split.

    Labels are returned ordered and without gaps.

    Args:
      split: A Split, the split for which to get the superclasses.

    Returns:
      The sequence of superclasses for the split.

    Raises:
      ValueError: An invalid split was specified.
    """
    return get_classes(split, self.superclasses_per_split)

  def _count_classes_in_superclasses(self, superclass_ids):
    return sum([
        self.classes_per_superclass[superclass_id]
        for superclass_id in superclass_ids
    ])

  def _get_split_offset(self, split):
    """Returns the starting class id of the contiguous chunk of ids of split.

    Args:
      split: A Split, the split for which to get classes.

    Raises:
      ValueError: Invalid dataset split.
    """
    if split == learning_spec.Split.TRAIN:
      offset = 0
    elif split == learning_spec.Split.VALID:
      previous_superclasses = range(
          0, self.superclasses_per_split[learning_spec.Split.TRAIN])
      offset = self._count_classes_in_superclasses(previous_superclasses)
    elif split == learning_spec.Split.TEST:
      previous_superclasses = range(
          0, self.superclasses_per_split[learning_spec.Split.TRAIN] +
          self.superclasses_per_split[learning_spec.Split.VALID])
      offset = self._count_classes_in_superclasses(previous_superclasses)
    else:
      raise ValueError('Invalid dataset split.')
    return offset

  def get_classes(self, split):
    """Gets the sequence of class labels for a split.

    Labels are returned ordered and without gaps.

    Args:
      split: A Split, the split for which to get classes.

    Returns:
      The sequence of classes for the split.
    """
    if not hasattr(self, 'restricted_classes_per_split'):
      self.initialize()
    offset = self._get_split_offset(split)
    if (self.restricted_classes_per_split is not None and
        split in self.restricted_classes_per_split):
      num_split_classes = self.restricted_classes_per_split[split]
    else:
      # No restriction, so include all classes of the given split.
      num_split_classes = self._count_classes_in_superclasses(
          self.get_superclasses(split))

    return range(offset, offset + num_split_classes)

  def get_class_ids_from_superclass_subclass_inds(self, split, superclass_id,
                                                  class_inds):
    """Gets the class ids of a number of classes of a given superclass.

    Args:
      split: A Split, the split for which to get classes.
      superclass_id: An int. The id of a superclass.
      class_inds: A list or sequence of ints. The indices into the classes of
        the superclass superclass_id that we wish to return class id's for.

    Returns:
      rel_class_ids: A list of ints of length equal to that of class_inds. The
        class id's relative to the split (between 0 and num classes in split).
      class_ids: A list of ints of length equal to that of class_inds. The class
        id's relative to the dataset (between 0 and the total num classes).
    """
    # The number of classes before the start of superclass_id, i.e. the class id
    # of the first class of the given superclass.
    superclass_offset = self._count_classes_in_superclasses(
        range(superclass_id))

    # Absolute class ids (between 0 and the total number of dataset classes).
    class_ids = [superclass_offset + class_ind for class_ind in class_inds]

    # Relative (between 0 and the total number of classes in the split).
    # This makes the assumption that the class id's are in a contiguous range.
    rel_class_ids = [
        class_id - self._get_split_offset(split) for class_id in class_ids
    ]

    return rel_class_ids, class_ids

  def to_dict(self):
    """Returns a dictionary for serialization to JSON.

    Each member is converted to an elementary type that can be serialized to
    JSON readily.
    """
    # Start with the dict representation of the namedtuple
    ret_dict = self._asdict()
    # Add the class name for reconstruction when deserialized
    ret_dict['__class__'] = self.__class__.__name__
    # Convert Split enum instances to their name (string)
    ret_dict['superclasses_per_split'] = {
        split.name: count
        for split, count in six.iteritems(ret_dict['superclasses_per_split'])
    }
    return ret_dict


class HierarchicalDatasetSpecification(
    collections.namedtuple('HierarchicalDatasetSpecification',
                           ('name, split_subgraphs, images_per_class, '
                            'class_names, path, file_pattern'))):
  """The specification of a hierarchical dataset.

    Args:
      name: string, the name of the dataset.
      split_subgraphs: a dict that maps each Split to a set of nodes of its
        corresponding graph.
      images_per_class: dict mapping each Split to a dict that maps each node in
        that split's subgraph to the number of images in the subgraph of that
        node. Note that we can't merge these three dicts into a single one, as
        there are nodes that will appear in more than one of these three
        subgraphs but will have different connections (parent/child pointers) in
        each one, therefore 'spanning' a different number of images.
      class_names: a dict mapping each class id to the corresponding class name.
        For ilsvrc_2012, the WordNet id's are used in the place of the names.
      path: the path to the dataset's files.
      file_pattern: a string representing the naming pattern for each class's
        file. The string must contain a placeholder for the class's ID (e.g. for
        ImageNet this is the WordNet id).
  """

  # TODO(etriantafillou): Make this class inherit from object instead
  # TODO(etriantafillou): Move this method to the __init__ of that revised class
  def initialize(self, restricted_classes_per_split=None):
    """Initializes a HierarchicalDatasetSpecification.

    Args:
      restricted_classes_per_split: A dict that specifies for each split, a
        number to restrict its classes to. This number must be no greater than
        the total number of classes of that split. By default this is None and
        no restrictions are applied (all classes are used).
    """
    # Set self.class_names_to_ids to the inverse dict of self.class_names.
    self.class_names_to_ids = dict(
        zip(self.class_names.values(), self.class_names.keys()))

    # Maps each Split enum to the number of its classes.
    self.classes_per_split = self.get_classes_per_split()

    if restricted_classes_per_split is not None:
      _check_validity_of_restricted_classes_per_split(
          restricted_classes_per_split, self.classes_per_split)
      # Apply the restriction.
      for split, restricted_num_classes in restricted_classes_per_split.items():
        self.classes_per_split[split] = restricted_num_classes

  def get_classes_per_split(self):
    """Returns a dict mapping each split enum to the number of its classes."""

    def count_split_classes(split):
      graph = self.split_subgraphs[split]
      leaves = imagenet_specification.get_leaves(graph)
      return len(leaves)

    classes_per_split = {}
    for split in [
        learning_spec.Split.TRAIN, learning_spec.Split.VALID,
        learning_spec.Split.TEST
    ]:
      classes_per_split[split] = count_split_classes(split)
    return classes_per_split

  def get_split_subgraph(self, split):
    """Returns the sampling subgraph DAG for the given split.

    Args:
      split: A Split, the split for which to get classes.
    """
    return self.split_subgraphs[split]

  def get_classes(self, split):
    """Returns a list of the class id's of classes assigned to split.

    Args:
      split: A Split, the split for which to get classes.
    """
    # The call to initialize computes self.classes_per_split. Do it only if it
    # hasn't already been done.
    if not hasattr(self, 'classes_per_split'):
      self.initialize()
    return get_classes(split, self.classes_per_split)

  def get_all_classes_same_example_count(self):
    """If all classes have the same number of images, return that number.

    Returns:
      An int, representing the common among all dataset classes number of
      examples, if the classes are balanced, or -1 to indicate class imbalance.
    """

    def list_leaf_num_images(split):
      return [
          self.images_per_class[split][n] for n in
          imagenet_specification.get_leaves(self.split_subgraphs[split])
      ]

    train_example_counts = set(list_leaf_num_images(learning_spec.Split.TRAIN))
    valid_example_counts = set(list_leaf_num_images(learning_spec.Split.VALID))
    test_example_counts = set(list_leaf_num_images(learning_spec.Split.TEST))

    is_class_balanced = (
        len(train_example_counts) == 1 and len(valid_example_counts) == 1 and
        len(test_example_counts) == 1 and
        len(train_example_counts | valid_example_counts
            | test_example_counts) == 1)

    if is_class_balanced:
      return list(train_example_counts)[0]
    else:
      return -1

  def get_total_images_per_class(self, class_id=None, pool=None):
    """Gets the number of images of class whose id is class_id.

    class_id can only be None in the case where all classes of the dataset have
    the same number of images.

    Args:
      class_id: The integer class id of a class.
      pool: None or string, unused. Should be None because no dataset with a DAG
        hierarchy supports example-level splits currently.

    Returns:
      An integer representing the number of images of class with id class_id.

    Raises:
      ValueError: no class_id specified yet there is class imbalance, or
        class_id is specified but doesn't correspond to any class, or "pool"
        is provided.
    """
    if pool is not None:
      raise ValueError('No dataset with a HierarchicalDataSpecification '
                       'supports example-level splits (pools).')

    common_num_class_images = self.get_all_classes_same_example_count()
    if class_id is None:
      if common_num_class_images < 0:
        raise ValueError('class_id can only be None in the case where all '
                         'dataset classes have the same number of images.')
      return common_num_class_images

    # Find the class with class_id in one of the split graphs.
    for s in learning_spec.Split:
      for n in self.split_subgraphs[s]:
        # Only consider leaves, as class_names_to_ids only has keys for them.
        if n.children:
          continue
        if self.class_names_to_ids[n.wn_id] == class_id:
          return self.images_per_class[s][n]
    raise ValueError('Class id {} not found.'.format(class_id))

  def to_dict(self):
    """Returns a dictionary for serialization to JSON.

    Each member is converted to an elementary type that can be serialized to
    JSON readily.
    """
    # Start with the dict representation of the namedtuple
    ret_dict = self._asdict()
    # Add the class name for reconstruction when deserialized
    ret_dict['__class__'] = self.__class__.__name__
    # Convert the graph for each split into a serializable form
    split_subgraphs = {}
    for split, subgraph in six.iteritems(ret_dict['split_subgraphs']):
      exported_subgraph = imagenet_specification.export_graph(subgraph)
      split_subgraphs[split.name] = exported_subgraph
    ret_dict['split_subgraphs'] = split_subgraphs
    # WordNet synsets to their WordNet ID as a string in images_per_class.
    images_per_class = {}
    for split, synset_counts in six.iteritems(ret_dict['images_per_class']):
      wn_id_counts = {
          synset.wn_id: count for synset, count in six.iteritems(synset_counts)
      }
      images_per_class[split.name] = wn_id_counts
    ret_dict['images_per_class'] = images_per_class

    return ret_dict


def as_dataset_spec(dct):
  """Hook to `json.loads` that builds a DatasetSpecification from a dict.

  Args:
     dct: A dictionary with string keys, corresponding to a JSON file.

  Returns:
    Depending on the '__class__' key of the dictionary, a DatasetSpecification,
    HierarchicalDatasetSpecification, or BiLevelDatasetSpecification. Defaults
    to returning `dct`.
  """
  if '__class__' not in dct:
    return dct

  if dct['__class__'] not in ('DatasetSpecification',
                              'HierarchicalDatasetSpecification',
                              'BiLevelDatasetSpecification'):
    return dct

  def _key_to_int(dct):
    """Returns a new dictionary whith keys converted to ints."""
    return {int(key): value for key, value in six.iteritems(dct)}

  def _key_to_split(dct):
    """Returns a new dictionary whith keys converted to Split enums."""
    return {
        learning_spec.Split[key]: value for key, value in six.iteritems(dct)
    }

  if dct['__class__'] == 'DatasetSpecification':
    images_per_class = {}
    for class_id, n_images in six.iteritems(dct['images_per_class']):
      # If n_images is a dict, it maps each class ID to a string->int
      # dictionary containing the size of each pool.
      if isinstance(n_images, dict):
        # Convert the number of classes in each pool to int.
        n_images = {
            pool: int(pool_size) for pool, pool_size in six.iteritems(n_images)
        }
      else:
        n_images = int(n_images)
      images_per_class[int(class_id)] = n_images

    return DatasetSpecification(
        name=dct['name'],
        classes_per_split=_key_to_split(dct['classes_per_split']),
        images_per_class=images_per_class,
        class_names=_key_to_int(dct['class_names']),
        path=dct['path'],
        file_pattern=dct['file_pattern'])

  elif dct['__class__'] == 'BiLevelDatasetSpecification':
    return BiLevelDatasetSpecification(
        name=dct['name'],
        superclasses_per_split=_key_to_split(dct['superclasses_per_split']),
        classes_per_superclass=_key_to_int(dct['classes_per_superclass']),
        images_per_class=_key_to_int(dct['images_per_class']),
        superclass_names=_key_to_int(dct['superclass_names']),
        class_names=_key_to_int(dct['class_names']),
        path=dct['path'],
        file_pattern=dct['file_pattern'])

  elif dct['__class__'] == 'HierarchicalDatasetSpecification':
    # Load subgraphs associated to each split, and build global mapping from
    # WordNet ID to Synset objects.
    split_subgraphs = {}
    wn_id_to_node = {}
    for split in learning_spec.Split:
      split_subgraphs[split] = imagenet_specification.import_graph(
          dct['split_subgraphs'][split.name])
      for synset in split_subgraphs[split]:
        wn_id = synset.wn_id
        if wn_id in wn_id_to_node:
          raise ValueError(
              'Multiple `Synset` objects associated to the same WordNet ID')
        wn_id_to_node[wn_id] = synset

    images_per_class = {}
    for split_name, wn_id_counts in six.iteritems(dct['images_per_class']):
      synset_counts = {
          wn_id_to_node[wn_id]: int(count)
          for wn_id, count in six.iteritems(wn_id_counts)
      }
      images_per_class[learning_spec.Split[split_name]] = synset_counts

    return HierarchicalDatasetSpecification(
        name=dct['name'],
        split_subgraphs=split_subgraphs,
        images_per_class=images_per_class,
        class_names=_key_to_int(dct['class_names']),
        path=dct['path'],
        file_pattern=dct['file_pattern'])

  else:
    return dct


def load_dataset_spec(dataset_records_path, convert_from_pkl=False):
  """Loads dataset specification from directory containing the dataset records.

  Newly-generated datasets have the dataset specification serialized as JSON,
  older ones have it as a .pkl file. If no JSON file is present and
  `convert_from_pkl` is passed, this method will load the .pkl and serialize it
  to JSON.

  Args:
    dataset_records_path: A string, the path to the directory containing
      .tfrecords files and dataset_spec.
    convert_from_pkl: A boolean (False by default), whether to convert a
      dataset_spec.pkl file to JSON.

  Returns:
    A DatasetSpecification, BiLevelDatasetSpecification, or
      HierarchicalDatasetSpecification, depending on the dataset.

  Raises:
    RuntimeError: If no suitable dataset_spec file is found in directory
      (.json or .pkl depending on `convert_from_pkl`).
  """
  json_path = os.path.join(dataset_records_path, 'dataset_spec.json')
  pkl_path = os.path.join(dataset_records_path, 'dataset_spec.pkl')
  if tf.io.gfile.exists(json_path):
    with tf.io.gfile.GFile(json_path, 'r') as f:
      data_spec = json.load(f, object_hook=as_dataset_spec)
  elif tf.io.gfile.exists(pkl_path):
    if convert_from_pkl:
      logging.info('Loading older dataset_spec.pkl to convert it.')
      with tf.io.gfile.GFile(pkl_path, 'rb') as f:
        data_spec = pkl.load(f)
      with tf.io.gfile.GFile(json_path, 'w') as f:
        json.dump(data_spec.to_dict(), f, indent=2)
    else:
      raise RuntimeError(
          'No dataset_spec.json file found in directory %s, but an older '
          'dataset_spec.pkl was found. You can try to pass '
          '`convert_from_pkl=True` to convert it, or you may need to run the '
          'conversion again in order to make sure you have the latest version.'
          % dataset_records_path)
  else:
    raise RuntimeError('No dataset_spec file found in directory %s' %
                       dataset_records_path)

  # Replace outdated path of where to find the dataset's records.
  data_spec = data_spec._replace(path=dataset_records_path)
  return data_spec
