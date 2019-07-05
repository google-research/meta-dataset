# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
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
from meta_dataset import data
from meta_dataset.data import imagenet_specification
from meta_dataset.data import learning_spec
from six.moves import range
from six.moves import zip
import tensorflow as tf

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

  return range(offset, offset + num_classes)


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

  def initialize(self):
    """Initializes a DatasetSpecification.

    Raises:
      ValueError: Invalid file_pattern provided
    """
    # Check that the file_pattern adheres to one of the allowable forms
    if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
      raise ValueError('file_pattern must be either "{}.tfrecords" or '
                       '"{}_{}.tfrecords" to support shards or splits.')

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

  def initialize(self):
    """Initializes a DatasetSpecification.

    Raises:
      ValueError: Invalid file_pattern provided
    """
    # Check that the file_pattern adheres to one of the allowable forms
    if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
      raise ValueError('file_pattern must be either "{}.tfrecords" or '
                       '"{}_{}.tfrecords" to support shards or splits.')

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

    Raises:
      ValueError: An invalid split was specified.
    """
    offset = self._get_split_offset(split)
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
  def initialize(self):
    """Initializes a HierarchicalDatasetSpecification."""
    # Set self.class_names_to_ids to the inverse dict of self.class_names.
    self.class_names_to_ids = dict(
        zip(self.class_names.values(), self.class_names.keys()))

    # Maps each Split enum to the number of its classes.
    self.classes_per_split = self.get_classes_per_split()

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
    # Computes self.classes_per_split.
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
