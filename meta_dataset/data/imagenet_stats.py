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
"""Computes stats of the graphs created in imagenet_specification.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import numpy as np
from six.moves import range


def log_graph_stats(nodes,
                    num_images,
                    get_leaves_fn,
                    get_spanning_leaves_fn,
                    graph_name=None,
                    min_way=5,
                    max_way=50):
  """Compute and display statistics about the graph defined by nodes.

  In particular, the statistics that are computed are:
  the number of nodes, the numbers of roots and leaves, the min/max/mean number
  of images living in the leaves, the min/max/mean number of children of
  internal nodes, the min/max/mean depth of leaves.

  Args:
    nodes: A set of Synsets representing a graph.
    num_images: A dict mapping each node's WordNet id to the number of images
      living in the leaves spanned by that node.
    get_leaves_fn: A function that returns the set of leaves of a graph defined
      by a given set of nodes, e.g. get_leaves in imagenet_specification.py
    get_spanning_leaves_fn: A function that returns a dict mapping each node of
      a given set of nodes to the set of leaf Synsets spanned by that node, e.g.
      get_spanning_leaves in imagenet_specification.py.
    graph_name: A name for the graph (for the printed logs).
    min_way: The smallest allowable way of an episode.
    max_way: The largest allowable way of an episode.
  """
  logging.info(
      'Graph statistics%s:',
      ' of graph {}'.format(graph_name) if graph_name is not None else '')
  logging.info('Number of nodes: %d', len(nodes))

  # Compute the dict mapping internal nodes to their spanning leaves. Note that
  # this is different for the different splits since even for nodes that may be
  # shared across splits, their connectivity will be different.
  spanning_leaves = get_spanning_leaves_fn(nodes)

  # Compute the number of roots and leaves
  num_roots = 0
  for n in nodes:
    if not n.parents:
      num_roots += 1
      logging.info('Root: %s', n.words)
  logging.info('Number of roots: %d', num_roots)
  leaves = get_leaves_fn(nodes)
  logging.info('Number of leaves: %d', len(leaves))

  # Compute the number of images in the leaves
  num_leaf_images = []
  for n in nodes:
    if n.children:
      continue
    num_leaf_images.append(num_images[n])
  logging.info('Number of leaf images: min %d, max %d, median %f',
               min(num_leaf_images), max(num_leaf_images),
               np.median(num_leaf_images))

  # Compute the average number of children of internal nodes
  num_children = []
  for n in nodes:
    if not n.children:
      continue
    num_children.append(len(n.children))
  logging.info(
      'Number of children of internal nodes: min %d, max %d, mean %f median %f',
      min(num_children), max(num_children), np.mean(num_children),
      np.median(num_children))

  # Compute the average number of leaves spanned by internal nodes.
  num_span_leaves = []
  for n in nodes:
    if not n.children:
      continue
    num_span_leaves.append(len(spanning_leaves[n]))
  logging.info(
      'Number of spanning leaves of internal nodes: min %d, max %d, mean %f '
      'median %f', min(num_span_leaves), max(num_span_leaves),
      np.mean(num_span_leaves), np.median(num_span_leaves))

  # Log the effects of restricting the allowable 'way' of episodes.
  all_reachable_leaves = set()  # leaves reachable under the restriction.
  possible_ways_in_range = []
  for v in spanning_leaves.values():
    way = len(v)
    if way >= min_way and way <= max_way:
      possible_ways_in_range.append(way)
      all_reachable_leaves |= set(v)
  logging.info(
      'When restricting the allowable way to be between %d and %d, '
      'the achievable ways are: %s', min_way, max_way, possible_ways_in_range)
  logging.info(
      'So there is a total of %d available internal nodes and a '
      'total of %d different ways.', len(possible_ways_in_range),
      len(set(possible_ways_in_range)))
  # Are all leaves reachable when using the restricted way?
  logging.info(' %d / %d are reachable.', len(all_reachable_leaves),
               len(leaves))


def log_stats_finegrainedness(nodes,
                              get_leaves_fn,
                              get_lowest_common_ancestor_fn,
                              graph_name=None,
                              num_per_height_to_print=2,
                              num_leaf_pairs=10000,
                              path='longest'):
  """Gather some stats relating to the heights of LCA's of random leaf pairs.

  Args:
    nodes: A set of Synsets.
    get_leaves_fn: A function that returns the set of leaves of a graph defined
      by a given set of nodes, e.g. get_leaves in imagenet_specification.py
    get_lowest_common_ancestor_fn: A function that returns the lowest common
      ancestor node of a given pair of Synsets and its height, e.g. the
      get_lowest_common_ancestor function in imagenet_specification.py.
    graph_name: A name for the graph defined by nodes (for logging).
    num_per_height_to_print: An int. The number of example leaf pairs and
      corresponding lowest common ancestors to print for each height.
    num_leaf_pairs: An int. The number of random leaf pairs to sample.
    path: A str. The 'path' argument of get_lowest_common_ancestor. Can be
      either 'longest' or 'all.
  """
  logging.info(
      'Finegrainedness analysis of %s graph using %s paths in '
      'finding the lowest common ancestor.', graph_name, path)
  leaves = get_leaves_fn(nodes)
  # Maps the height of the lowest common ancestor of two leaves to the 'example'
  # in which that height occurred. The example is a tuple of the string words
  # associated with (first leaf, second leaf, lowest common ancestor).
  heights_to_examples = collections.defaultdict(list)
  # Maps the height of the lowest common ancestor of two leaves to the number of
  # leaf pairs whose LCA has that height and is the root.
  heights_to_num_lca_root = collections.defaultdict(int)
  # A list of all observed LCA heights.
  heights = []
  # Sample a number of random pairs of leaves, and compute the height of their
  # lowest common ancestor.
  for _ in range(num_leaf_pairs):
    first_ind = np.random.randint(len(leaves))
    second_ind = np.random.randint(len(leaves))
    while first_ind == second_ind:
      second_ind = np.random.randint(len(leaves))
    leaf_a = leaves[first_ind]
    leaf_b = leaves[second_ind]
    lca, height = get_lowest_common_ancestor_fn(leaf_a, leaf_b, path=path)
    heights.append(height)

    heights_to_examples[height].append((leaf_a.words, leaf_b.words, lca.words))
    if not lca.parents:
      heights_to_num_lca_root[height] += 1

  name_message = ' of the {} graph'.format(
      graph_name) if graph_name is not None else ''
  stats_message = 'mean: {}, median: {}, max: {}, min: {}'.format(
      np.mean(heights), np.median(heights), max(heights), min(heights))
  logging.info(
      'Stats on the height of the Lowest Common Ancestor of random leaf pairs%s'
      ': %s', name_message, stats_message)

  # For each given height, how many pairs of leaves are there?
  heights_to_num_examples = {}
  heights_to_proportion_root = {}
  for h, examples in heights_to_examples.items():
    heights_to_num_examples[h] = len(examples) / num_leaf_pairs
    heights_to_proportion_root[h] = heights_to_num_lca_root[h] / float(
        len(examples))
  logging.info(
      'Proportion of example leaf pairs (out of num_leaf_pairs '
      'random pairs) for each height of the LCA of the leaves: %s',
      heights_to_num_examples)

  # What proportion of those have the root as LCA, for each possible height?
  logging.info(
      'Proportion of example leaf pairs per height whose LCA is the root: %s',
      heights_to_proportion_root)

  logging.info('Examples with different fine-grainedness:\n')
  for height in heights_to_examples.keys():
    # Get representative examples of this height.
    for i, example in enumerate(heights_to_examples[height]):
      if i == num_per_height_to_print:
        break
      logging.info('Examples with height %s:\nleafs: %s and %s. LCA: %s',
                   height, example[0], example[1], example[2])
