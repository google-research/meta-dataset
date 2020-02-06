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
"""Prepares the ILSVRC2012 subset of ImageNet for integration in the benchmark.

This requires creating a data structure to navigate the subset of the ontology
of ImageNet that is relevant. This data structure is required both for creating
class splits in a hierarchy-aware manner and also for the episode generation
algorithm. This algorithm as well as more context about handling hierarchy and
ImageNet's ontology in particular is described in the article.
"""
# TODO(manzagop): relocate the code pertaining to imagenet ingestion to
# dataset_conversion. The code dealing with sampling from a tree should stay.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import logging
from meta_dataset.data import imagenet_stats
import numpy as np
import six
import tensorflow.compat.v1 as tf

tf.flags.DEFINE_string(
    'ilsvrc_2012_data_root',
    '',
    'Path to the root of the ImageNet data.')

tf.flags.DEFINE_string(
    'path_to_is_a',
    '',
    'Path to the file containing is-a relationships (parent, child) pairs. '
    'If empty, it defaults to "wordnet.is_a.txt" in ilsvrc_2012_data_root.')

tf.flags.DEFINE_string(
    'path_to_words',
    '',
    'Path to the file containing (synset, word description) pairs. '
    'If empty, it defaults to "words.txt" in ilsvrc_2012_data_root.')

FLAGS = tf.flags.FLAGS


class Synset(object):
  """A Synset object."""

  def __init__(self, wn_id, words, children, parents):
    """Initialize a Synset.

    Args:
      wn_id: WordNet id
      words: word description of the synset
      children: a set of children Synsets
      parents: a set of parent Synsets
    """
    self.wn_id = wn_id
    self.words = words
    self.children = children
    self.parents = parents


def get_node_ancestors(synset):
  """Create a set consisting of all and only the ancestors of synset.

  Args:
    synset: A Synset.

  Returns:
    ancestors: A set of Synsets
  """
  ancestors = set()
  # In the following line, synset.parents already is a set but we create a copy
  # of it instead of using synset.parents directly as later we are 'popping'
  # elements from this set, which would otherwise result to permanently removing
  # parents of synset which is undesirable.
  to_visit = set(synset.parents)
  visited = set()
  while to_visit:
    ancestor = to_visit.pop()
    ancestors.add(ancestor)
    visited.add(ancestor)
    # Same as in the comment above, we create a copy of ancestor.parents
    to_visit = to_visit | set(ancestor.parents) - visited
  return ancestors


def get_ancestors(synsets):
  """Create a set consisting of all and only the ancestors of leaves.

  Args:
    synsets: A list of Synsets.

  Returns:
    A set of Synsets.
  """
  all_ancestors = set()
  for s in synsets:
    all_ancestors = all_ancestors | get_node_ancestors(s)
  return all_ancestors


def isolate_graph(nodes):
  """Remove links between Synsets in nodes and Synsets that are not in nodes.

  This effectively isolates the graph defined by nodes from the rest of the
  Synsets. The resulting set of nodes is such that following any of their child/
  parent pointers can only lead to other Synsets of the same set of nodes.
  This requires breaking the necessary parent / child links.

  Args:
    nodes: A set of Synsets
  """
  for n in nodes:
    n.children = list(nodes & set(n.children))
    n.parents = list(nodes & set(n.parents))


def isolate_node(node):
  """Isolate node from its children and parents by breaking those links."""
  for p in node.parents:
    p.children.remove(node)
  for c in node.children:
    c.parents.remove(node)
  node.children = []
  node.parents = []


def collapse(nodes):
  """Collapse any nodes that only have a single child.

  Collapsing of a node is done by removing that node and attaching its child to
  its parent(s).

  Args:
    nodes: A set of Synsets.

  Returns:
    A set containing the Synsets in nodes that were not collapsed, with
    potentially modified children and parents lists due to collapsing other
    synsets.
  """

  def collapse_once(nodes):
    """Perform a pass of the collapsing as described above."""
    num_collapsed = 0
    non_collapsed_nodes = set()
    for n in nodes:
      if len(n.children) == 1:
        # attach the only child to all of n's parents
        n.children[0].parents += n.parents
        for p in n.parents:
          p.children.append(n.children[0])

        # Remove all connections to and from n
        isolate_node(n)

        num_collapsed += 1
      else:
        non_collapsed_nodes.add(n)

    assert len(nodes) - len(non_collapsed_nodes) == num_collapsed
    return non_collapsed_nodes, num_collapsed

  nodes, num_collapsed = collapse_once(nodes)
  while num_collapsed:
    nodes, num_collapsed = collapse_once(nodes)
  return nodes


def get_leaves(nodes):
  """Return a list containing the leaves of the graph defined by nodes."""
  leaves = []
  for n in nodes:
    if not n.children:
      leaves.append(n)
  return leaves


def get_synsets_from_ids(wn_ids, synsets):
  """Finds the Synsets in synsets whose WordNet id's are in wn_ids.

  Args:
    wn_ids: A list of WordNet id's.
    synsets: A set of Synsets.

  Returns:
    A dict mapping each WordNet id in wn_ids to the corresponding Synset.
  """
  wn_ids = set(wn_ids)
  requested_synsets = {}
  for s in synsets:
    if s.wn_id in wn_ids:
      requested_synsets[s.wn_id] = s

  found = set(requested_synsets.keys())
  assert found == wn_ids, ('Did not find synsets for ids: {}.'.format(wn_ids -
                                                                      found))
  return requested_synsets


def get_spanning_leaves(nodes):
  """Get the leaves that each node in nodes can reach.

  The number of leaves that a node can reach, i.e. that a node 'spans', provides
  an estimate of how 'high' in the DAG that node is, with nodes representing
  more general concepts spanning more leaves and being 'higher' than nodes
  representing more specific concepts.
  A leaf node spans exactly one leaf: itself.

  Args:
    nodes: A set of Synsets

  Returns:
    spanning_leaves: a dict mapping Synset instances to the set of leaf Synsets
      that are their descendants.
  """
  # First find the leaves
  leaves = get_leaves(nodes)

  # dict mapping WordNet id's to the list of leaf Synsets they span
  spanning_leaves = {}
  for n in nodes:
    spanning_leaves[n] = set()
    for l in leaves:
      if is_descendent(l, n) or l == n:
        spanning_leaves[n].add(l)
  return spanning_leaves


def get_num_spanning_images(spanning_leaves, num_leaf_images):
  """Create a dict mapping each node to the number of images in its sub-graph.

  This assumes that all images live in the leaves of the graph (as is the case
  in our graph by construction: we are only interested in using images from the
  ILSVRC 2012 synsets, and these synsets are all and only the leaves of the
  sampling graph).

  Args:
    spanning_leaves: a dict mapping each node to the set of leaves it spans.
    num_leaf_images: a dict mapping each leaf synset to its number of images.

  Returns:
    num_images: a dict that maps each node in the sampling graph to the number
      of images in the leaves that it spans.
  """
  num_images = {}
  for node, leaves in spanning_leaves.items():
    num_images[node] = sum([num_leaf_images[l.wn_id] for l in leaves])
  return num_images


def create_sampling_graph(synsets, root=None):
  """Create a DAG that only contains synsets and all of their ancestors.

  By construction, the leaves of this graph are all and only the Synsets in the
  synsets list. The internal nodes of the graph are all and only the ancestors
  of synsets. All children/parent pointers of the graph nodes are restricted to
  only lead to other nodes that also belong to the ancestor graph. Finally,
  appropriate collapsing is performed so that no node has only one child (since
  it's not possible to create an episode from that node).

  Args:
    synsets: A list of Synsets
    root: Optionally, a Synset. If provided, it imposes a restriction on which
      ancestors of the given synsets will be included in the sampling graph.
      Specifically, an ancestor of a Synset in synsets in will be included only
      if it is the root or a descendent of the root. This is useful when
      creating the validation and test sub- graphs, where we want the designated
      root to indeed have to upward connections in the corresponding subgraph.

  Returns:
    A set of the Synsets of the DAG.
  """
  # Get the set of Synsets containing all and only the ancestors of synsets.
  nodes = get_ancestors(synsets)

  if root is not None:
    # Remove from the ancestors nodes that aren't the root or descendents of it.
    nodes_to_remove = [
        n for n in nodes if not (root == n or is_descendent(n, root))
    ]
    nodes = nodes - set(nodes_to_remove)

  # The total nodes of the graph are the nodes in synsets, and their ancestors.
  nodes = nodes | set(synsets)

  # Remove all connections from nodes of the graph to other nodes.
  isolate_graph(nodes)

  # Remove all nodes with only one child and attach that child to their parents.
  nodes = collapse(nodes)
  return nodes


def propose_valid_test_roots(spanning_leaves,
                             margin=50,
                             desired_num_valid_classes=150,
                             desired_num_test_classes=150):
  """Propose roots for the validation and test sub-graphs.

  This is done as follows: each subgraph root will be the Synset that spans the
  largest number of leaves that still meets the criterion of spanning a number
  of leaves within its allowable range, which is: desired number of classes for
  that split +/- margin. We aim to include approx. 70% / 15% / 15% of the
  classes in the training / validation / testing splits, resp.

  Args:
    spanning_leaves: A dict mapping each Synset to the leaf Synsets that are
      reachable from it.
    margin: The number of additional or fewer leaves that the root of a split's
      subgraph can span compared to the expected number of classes for the
      corresponding split. This is needed for this splitting method, as there
      may not be a node in the tree that spans exactly the expected number of
      classes for some split.
    desired_num_valid_classes: num classes desirably assigned to the validation
      split. ILSVRC 2012 has a total of 1000 classes, so 15% corresponds to 150
      classes, hence the default value of 150.
    desired_num_test_classes: similarly as above, but for the test split.

  Returns:
    a dict that maps 'valid' and 'test' to the synset that spans the leaves that
      will desirably be assigned to that split.

  Raises:
    RuntimeError: When no candidate subgraph roots are available with the given
      margin value.
  """
  # Sort in decreasing order of the length of the lists of spanning leaves, so
  # e.g. the node that spans the most leaves will be the first element.  Ties
  # are broken by the WordNet ID of the Synset.
  def _sort_key(synset_and_leaves):
    synset, leaves = synset_and_leaves
    return (len(leaves), synset.wn_id)

  spanning_leaves_sorted = sorted(six.iteritems(spanning_leaves), key=_sort_key)
  spanning_leaves_sorted.reverse()

  # Get the candidate roots for the validation and test sub-graphs, by finding
  # the nodes whose number of spanning leaves are within the
  # allowed margin.
  valid_candidates, test_candidates = [], []
  for s, leaves in spanning_leaves_sorted:
    num_leaves = len(leaves)
    low_limit_valid = desired_num_valid_classes - margin
    high_limit_valid = desired_num_valid_classes + margin
    if low_limit_valid < num_leaves and num_leaves < high_limit_valid:
      valid_candidates.append(s)
    low_limit_test = desired_num_test_classes - margin
    high_limit_test = desired_num_test_classes + margin
    if low_limit_test < num_leaves and num_leaves < high_limit_test:
      test_candidates.append(s)

  if not valid_candidates or not test_candidates:
    raise RuntimeError('Found no root candidates. Try a different margin.')

  # For displaying the list of candidates
  for cand in valid_candidates:
    logging.info('Candidate %s, %s with %d spanning leaves', cand.words,
                 cand.wn_id, len(spanning_leaves[cand]))

  # Propose the first possible candidate for each of validation and test
  valid_root = valid_candidates[0]
  # Make sure not to choose the same root for testing as for validation
  test_candidate_ind = 0
  test_root = test_candidates[test_candidate_ind]
  while test_root == valid_root:
    test_candidate_ind += 1
    if test_candidate_ind == len(test_candidates):
      raise RuntimeError('No candidates for test root. Try a different margin.')
    test_root = test_candidates[test_candidate_ind]

  return {'valid': valid_root, 'test': test_root}


def get_class_splits(spanning_leaves, valid_test_roots=None, **kwargs):
  """Gets the assignment of classes (graph leaves) to splits.

  First, if valid_test_roots is not provided, roots for the validation and test
  sub-graphs are proposed by calling propose_valid_test_roots.

  Then, all classes spanned by the valid root Synset will be assigned to the
  validation set, all classes spanned by test root Synset will be assigned to
  the test split, and all remaining classes to the training split. When there
  are leaves spanned by both validation and test, they are assigned to one of
  the two randomly (roughly equally).

  Args:
    spanning_leaves: A dict mapping each Synset to the leaf Synsets that are
      reachable from it.
    valid_test_roots: A dict whose keys should be 'valid' and 'test' and whose
      value for a given key is a Synset that spans all and only the leaves that
      will desirably be assigned to the corresponding split.
    **kwargs: Keyword arguments for the root proposer that is used if
      valid_test_roots is None.

  Returns:
    split_classes: A dict that maps each of 'train', 'valid' and 'test' to the
      set of WordNet id's of the classes for the corresponding split.
    valid_test_roots: A dict of the same form as the corresponding optional
      argument.

  Raises:
    ValueError: when the provided valid_test_roots are invalid.
  """
  if valid_test_roots is not None:
    if valid_test_roots['valid'] is None or valid_test_roots['test'] is None:
      raise ValueError('A root cannot be None.')

  if valid_test_roots is None:
    valid_test_roots = propose_valid_test_roots(spanning_leaves, **kwargs)

  valid_root, test_root = valid_test_roots['valid'], valid_test_roots['test']

  # The WordNet id's of the validation and test classes
  valid_wn_ids = set([s.wn_id for s in spanning_leaves[valid_root]])
  test_wn_ids = set([s.wn_id for s in spanning_leaves[test_root]])

  # There may be overlap between the spanning leaves of the chosen roots for
  # the validation and test subtrees, which would cause overlap between the
  # classes assigned to these splits. This is addressed below by randomly
  # assigning each overlapping leaf to either validation or test classes
  # (roughly equally).
  overlap = [s for s in valid_wn_ids if s in test_wn_ids]
  logging.info('Size of overlap: %d leaves', len(overlap))
  assign_to_valid = True
  for s in overlap:
    if assign_to_valid:
      test_wn_ids.remove(s)
    else:
      valid_wn_ids.remove(s)
    assign_to_valid = not assign_to_valid

  # Training classes are all the remaining ones that are not already assigned
  leaves = get_leaves(spanning_leaves.keys())
  train_wn_ids = set([
      s.wn_id
      for s in leaves
      if s.wn_id not in valid_wn_ids and s.wn_id not in test_wn_ids
  ])

  split_classes = {
      'train': train_wn_ids,
      'valid': valid_wn_ids,
      'test': test_wn_ids
  }
  return split_classes, valid_test_roots


def init_split_subgraphs(class_splits, spanning_leaves, valid_test_roots):
  """Gets leaf and root Synsets from different copies of the graph.

  In particular, a new copy is created for each split. For all three splits, the
  leaf Synsets of the corresponding copy that correspond to split classes are
  returned. For the validation and test graphs, the corresponding root Synsets
  are returned as well from the new copies. These will have the same WordNet id
  and name as those in valid_test_roots but are nodes from the copy of the graph
  instead of the original one.

  Args:
    class_splits: a dict whose keys are 'train', 'valid' and 'test' and whose
      value for a given key is the set of WordNet id's of the classes that are
      assigned to the corresponding split.
    spanning_leaves: A dict mapping each Synset to the leaf Synsets that are
      reachable from it.
    valid_test_roots: A dict whose keys should be 'valid' and 'test' and whose
      value for a given key is a Synset that spans all and only the leaves that
      will desirably be assigned to the corresponding split.

  Returns:
    a dict mapping each of 'train', 'valid' and 'test' to the set of Synsets (of
    the respective copy of the graph) corresponding to the classes that are
    assigned to that split.

  Raises:
    ValueError: invalid keys for valid_test_roots, or same synset provided as
      the root of both valid and test.
  """
  # Get the wn_id's of the train, valid and test classes.
  train_wn_ids = class_splits['train']
  valid_wn_ids = class_splits['valid']
  test_wn_ids = class_splits['test']

  valid_root_wn_id = valid_test_roots['valid'].wn_id
  test_root_wn_id = valid_test_roots['test'].wn_id

  # Get 3 full copies of the graph that will be modified downstream.
  graph_copy_train, _ = copy_graph(spanning_leaves.keys())
  graph_copy_valid, valid_root = copy_graph(spanning_leaves.keys(),
                                            valid_root_wn_id)
  graph_copy_test, test_root = copy_graph(spanning_leaves.keys(),
                                          test_root_wn_id)

  # Get the nodes of each copy that correspond to the splits' assigned classes.
  train_classes = set([s for s in graph_copy_train if s.wn_id in train_wn_ids])
  valid_classes = set([s for s in graph_copy_valid if s.wn_id in valid_wn_ids])
  test_classes = set([s for s in graph_copy_test if s.wn_id in test_wn_ids])
  split_leaves = {
      'train': train_classes,
      'valid': valid_classes,
      'test': test_classes
  }
  split_roots = {'valid': valid_root, 'test': test_root}
  return split_leaves, split_roots


def copy_graph(nodes, root_wn_id=None):
  """Create a set of Synsets that are copies of the Synsets in nodes.

  A new Synset is created for each Synset of nodes and then the
  children/parent relationships of the new Synsets are set to mirror the
  corresponding ones in the Synsets of nodes.

  This assumes that nodes is an 'isolated' graph: all parents and
  children of nodes of nodes also belong to the graph.

  Optionally, if the WordNet id of a node is provided, the copy of that node
  will be returned.

  Args:
    nodes: A set of Synsets.
    root_wn_id: The wn_id field of the Synset that is intended to eventually be
      the root of the new graph.

  Returns:
    copy: A set of Synsets of the same size as nodes.
  """
  root_copy = None
  copy = {}  # maps wn_id's to Synsets
  parent_child_tuples = set()

  for s in nodes:
    copy[s.wn_id] = Synset(s.wn_id, s.words, set(), set())
    if root_wn_id is not None and s.wn_id == root_wn_id:
      root_copy = copy[s.wn_id]
    for c in s.children:
      assert c in nodes
      parent_child_tuples.add((s.wn_id, c.wn_id))

  # Add the analogous parent/child relations between nodes in
  # copy as those that existed in nodes
  for parent, child in parent_child_tuples:
    copy[parent].children.add(copy[child])
    copy[child].parents.add(copy[parent])

  return set(copy.values()), root_copy


def create_splits(spanning_leaves, split_enum, valid_test_roots=None, **kwargs):
  """Split the classes of ILSVRC 2012 into train / valid / test.

  Each split will be represented as a sub-graph of the overall sampling graph.
  The leaves of a split's sub-graph are the ILSVRC 2012 synsets that are
  assigned to that split, and its internal nodes are all and only the ancestors
  of those leaves. Each split's subgraph is 'isolated' from the rest of the
  synsets in that following pointers of nodes in that sub-graph is guaranteed
  to lead to other nodes within in.

  If valid_test_roots is not None, it should contain two Synsets, that are the
  proposed roots of the validation and test subtrees. Otherwise, a proposal for
  these two roots is made in get_class_splits.

  Args:
    spanning_leaves: A dict mapping each Synset to the leaf Synsets that are
      reachable from it.
    split_enum: A class that inherits from enum.Enum whose attributes are TRAIN,
      VALID, and TEST, which are mapped to enumerated constants.
    valid_test_roots: dict that provides for each of 'valid' and 'test' a synset
      that is the ancestor of all and only the leaves that will be assigned to
      the corresponding split.
    **kwargs: keyword args for the function used to propose valid_test_roots,
      which will be called if split_classes is empty and no valid_test_roots are
      provided.

  Returns:
    splits: a dict mapping each Split in split_enum to the set of Synsets in the
      subgraph of that split. This is different from the split_classes dict,
      which contained lists of only the leaves of the corresponding graphs.
    roots: a dict of the same type as valid_test_roots. If it was provided, it
      is returned unchanged. Otherwise the newly created one is returned.
  """
  # The classes (leaf Synsets of the overall graph) of each split.
  split_classes, valid_test_roots = get_class_splits(
      spanning_leaves, valid_test_roots=valid_test_roots, **kwargs)

  # The copies of the leaf and desired root Synsets for each split. Copies are
  # needed since in each sub-graph those nodes will have different children /
  # parent lists.
  leaves, roots = init_split_subgraphs(split_classes, spanning_leaves,
                                       valid_test_roots)

  # Create the split sub-graphs as described above.
  train_graph = create_sampling_graph(leaves['train'])
  valid_graph = create_sampling_graph(leaves['valid'], root=roots['valid'])
  test_graph = create_sampling_graph(leaves['test'], root=roots['test'])
  split_graphs = {
      split_enum.TRAIN: train_graph,
      split_enum.VALID: valid_graph,
      split_enum.TEST: test_graph
  }
  return split_graphs, roots


def get_synset_by_wnid(wnid, graph):
  """Return the synset of sampling_graph whose WordNet id is wnid."""
  for n in graph:
    if n.wn_id == wnid:
      return n
  return None


def is_descendent(d, a):
  """Returns whether d is a descendent of a.

  A node is not considered a descendent of itself.

  Args:
    d: A Synset.
    a: A Synset.
  """
  paths = get_upward_paths_from(d, end=a)
  # The second clause ensures that a node is not a descendent of itself (our
  # graphs are DAGs so this suffices to enforce this).
  return len(paths) and not (len(paths) == 1 and len(paths[0]) == 1)


def get_upward_paths_from(start, end=None):
  """Creates a list of paths that go from start either to end or to a root.

  There may be more than one such paths, since the structure we are traversing
  is a DAG (not strictly a tree). Every path is represented as a list of Synsets
  whose first elements is a Synset without parents and whose last element is s.

  Args:
    start: A Synset.
    end: A Synset. If not provided, the end point will be the first node that is
      encountered starting from start that does not have parents.

  Returns:
    A list of lists, containing all paths as described above.
  """

  def is_end_node(n):
    return (end is not None and n == end) or (end is None and not n.parents)

  if end is not None and not start.parents:
    # There are no upwards paths from start in which the specified end can be.
    return []

  if is_end_node(start):
    return [[start]]

  # If we got here, we haven't yet reached the target node and there are upward
  # paths to explore.
  parents = start.parents

  # A list of all paths from start to end (or to a root node).
  paths = []

  # Case where end is a direct parent of start:
  for p in parents:
    if is_end_node(p):
      # Found one path from start to end.
      paths.append([start, p])

    else:
      # Get a list of lists corresponding to paths between p and end.
      p_to_end_paths = get_upward_paths_from(p, end=end)
      if not p_to_end_paths:  # end not an ancestor of p.
        continue
      start_to_end_paths = [[start] + p_path for p_path in p_to_end_paths]
      paths.extend(start_to_end_paths)
  return paths


def find_lowest_common_in_paths(path_a, path_b):
  """Find the element with the smallest height that appears in both given lists.

  The height of an element here is defined as the maximum over the indices where
  it occurs in the two lists. For example if path_a = [2, 3, 5] and
  path_b = [5, 6, 2] then the height of element 2 is max(0 + 2) = 2 since the
  element 2 occurs in position 0 in the first list and position 2 in the second.

  Args:
    path_a: A list.
    path_b: A list.

  Returns:
    lowest_common: The element with the smallest 'height' that is common between
      path_a and path_b.
    height: The height of lowest_common, computed as described above.
  """
  # Maps elements that appear in both lists to their heights.
  common_elements, heights = [], []
  for element in path_a:
    if element in path_b:
      height = max(path_a.index(element), path_b.index(element))
      common_elements.append(element)
      heights.append(height)

  if not heights:
    raise ValueError('No common nodes in given paths {} and {}.'.format(
        [n.words for n in path_a], [n.words for n in path_b]))

  # Find the lowest common element.
  # There may be multiple common ancestors that share the same minimal height.
  # In that case the first one appearing in common_elements will be returned.
  min_height = min(heights)
  argmin_height = heights.index(min_height)
  lowest_common = common_elements[argmin_height]
  assert min_height > 0, ('The lowest common ancestor between two distinct '
                          'leaves cannot be a leaf.')
  return lowest_common, min_height


def get_lowest_common_ancestor(leaf_a, leaf_b, path='longest'):
  """Finds the lowest common ancestor of two leaves and its height.

  The height of a node here is defined as the maximum distance between that node
  and any of the leaves it spans.

  When there are multiple paths starting from a given leaf (due to it possibly
  having multiple parents), we rely on the value of path to choose which one to
  use. By default, we use the path whose length to the root is the longest. We
  find the lowest common ancestor of the two given leaves along the longest such
  path for each. Alternatively, all paths can be used in which case the minimum
  LCA over all is returned.

  Args:
    leaf_a: A Synset.
    leaf_b: A Synset.
    path: A str. One of 'longest', or 'all'.

  Returns:
    lca: A Synset. The lowest common ancestor.
    height_of_lca: An int. The height of the lowest common ancestor.

  Raises:
    ValueError: Invalid path. Must be 'longest', or 'all'.
  """
  if path not in ['longest', 'all']:
    raise ValueError('Invalid path. Must be "longest", or "all".')

  # A list of paths from a each leaf to the root.
  paths_a = get_upward_paths_from(leaf_a)
  paths_b = get_upward_paths_from(leaf_b)

  # Each element in paths_a is a path starting from leaf_a and ending at the
  # root (and analogously for paths_b). We pick the longest path of each list of
  # paths and find the lowest common ancestor between those two paths.
  if path == 'longest':
    path_a = paths_a[np.argmax([len(path_a) for path_a in paths_a])]
    path_b = paths_b[np.argmax([len(path_b) for path_b in paths_b])]
    lca, height_of_lca = find_lowest_common_in_paths(path_a, path_b)

  else:
    # Search for the LCA across all possible paths from the given leaves.
    lca, height_of_lca = None, None
    for path_a in paths_a:
      for path_b in paths_b:
        lca_candidate, height = find_lowest_common_in_paths(path_a, path_b)
        if height_of_lca is None or height < height_of_lca:
          lca = lca_candidate
          height_of_lca = height

  return lca, height_of_lca


def get_num_synset_2012_images(path, synsets_2012, files_to_skip=None):
  """Count the number of images of each class in ILSVRC 2012.

  Returns a dict mapping the WordNet of each class of ILSVRC 2012 to the
  number of its images.
  This assumes that within FLAGS.ilsvrc_2012_data_root there is a directory for
  every 2012 synset, named by that synset's WordNet ID (e.g. n15075141) and
  containing all images of that synset.

  If path contains this dict, it is read and returned, otherwise it is computed
  and stored at path.

  Args:
    path: An optional path to a cache where the computed dict is / may be
      stored.
    synsets_2012: A list of Synsets.
    files_to_skip: A set with the files that repeat in other datasets.

  Returns:
    a dict mapping the WordNet id of each ILSVRC 2012 class to its number of
    images.
  """
  if path:
    logging.info('Attempting to read number of leaf images from %s...', path)
    if tf.io.gfile.exists(path):
      with tf.io.gfile.GFile(path, 'r') as f:
        num_synset_2012_images = json.load(f)
        logging.info('Successful.')
        return num_synset_2012_images

  logging.info('Unsuccessful. Deriving number of leaf images...')
  if files_to_skip is None:
    files_to_skip = set()
  num_synset_2012_images = {}
  for s_2012 in synsets_2012:
    synset_dir = os.path.join(FLAGS.ilsvrc_2012_data_root, s_2012.wn_id)
    # Size of the set difference (-) between listed files and `files_to_skip`.
    num_synset_2012_images[s_2012.wn_id] = len(
        set(tf.io.gfile.listdir(synset_dir)) - files_to_skip)

  if path:
    with tf.io.gfile.GFile(path, 'w') as f:
      json.dump(num_synset_2012_images, f, indent=2)

  return num_synset_2012_images


def export_graph(nodes):
  """Returns a JSON-serializable representation of a graph.

  Synset objects are represented by a dictionary containing:
  - their WordNet ID ("wn_id")
  - their text description in words ("words")
  - the WordNet IDs of their children ("children_ids")
  - the WordNet IDs of their parents ("parents_ids")

  The conversion expects that:
  - The WordNet ID is a unique identifier for a Synset object.
  - The parents and children of each Synset in `nodes` is also in `nodes`.

  Args:
    nodes: A set of Synset objects, representing a complete graph.

  Returns:
    A list of dictionaries, following the representation described above.
  """
  node_representations = []
  wn_ids_to_synsets = {synset.wn_id: synset for synset in nodes}
  wn_ids = set(wn_ids_to_synsets.keys())
  if len(wn_ids) != len(nodes):
    raise ValueError('Duplicate WordNet IDs in the same graph')
  # Iterate in lexicographic order over the WordNet IDs
  for wn_id in sorted(wn_ids):
    synset = wn_ids_to_synsets[wn_id]
    children_ids = {child.wn_id for child in synset.children}
    if not children_ids.issubset(wn_ids):
      raise ValueError('Synset has children outside of the graph')
    parents_ids = {parent.wn_id for parent in synset.parents}
    if not parents_ids.issubset(wn_ids):
      raise ValueError('Synset has parents outside of the graph')
    node_repr = dict(
        wn_id=wn_id,
        words=synset.words,
        children_ids=sorted(children_ids),
        parents_ids=sorted(parents_ids))
    node_representations.append(node_repr)
  return node_representations


def import_graph(node_representations):
  """Returns a set of Synset nodes from JSON-serializable representation.

  See the documentation of `export_graph` for a description of the format
  of that representation.

  Args:
    node_representations: A list of dictionaries, each representing a Synset.

  Returns:
    A set of Synset objects (nodes), representing a graph.
  """
  graph = set()
  # Build one Synset node for each WordNet ID, and keep a mapping.
  # `children` and `parents` are initialized with empty sets.
  wn_id_to_node = dict()
  for node_repr in node_representations:
    wn_id = node_repr['wn_id']
    words = node_repr['words']
    if wn_id in wn_id_to_node:
      raise ValueError('Duplicate Word ID (%s, %s) in the imported graph.' %
                       (wn_id, words))
    node = Synset(wn_id=wn_id, words=words, children=set(), parents=set())
    wn_id_to_node[wn_id] = node

  # Fill in the `children` and `parents` with the Synset objects.
  for node_repr in node_representations:
    wn_id = node_repr['wn_id']
    node = wn_id_to_node[wn_id]
    children_ids = node_repr['children_ids']
    node.children.update(wn_id_to_node[child_id] for child_id in children_ids)
    parents_ids = node_repr['parents_ids']
    node.parents.update(wn_id_to_node[parent_id] for parent_id in parents_ids)
    graph.add(node)

  return graph


def create_imagenet_specification(split_enum,
                                  files_to_skip,
                                  path_to_num_leaf_images=None,
                                  log_stats=True):
  """Creates the dataset specification of ImageNet.

  This amounts to creating a data structure, a DAG specifically, whose nodes are
  synsets, and whose leaves are all and only the synsets corresponding to
  classes of the ILSVRC 2012 subset of ImageNet. Each synset is connected to
  others via child / parent links that indicate is-a relationships of the
  corresponding concepts (classes). This DAG will consist of all and only the
  nodes that are on paths starting from a leaf to a root (i.e. it won't contain
  any nodes that don't have leaf descendents) and it will be 'detached' from the
  remaining ImageNet synsets (that don't belong to the ILSVRC 2012 subset) in
  that by following a child or parent pointer of any DAG node we are guaranteed
  to reach another node that also belongs to the DAG.

  Args:
    split_enum: A class that inherits from enum.Enum whose attributes are TRAIN,
      VALID, and TEST, which are mapped to enumerated constants.
    files_to_skip: A set with the files that intersect with other datasets.
    path_to_num_leaf_images: A string, representing a path to a file containing
      a dict that maps the WordNet id of each ILSVRC 2012 class to the
      corresponding number of images. If no file is present, it will be created
      in order to save on future computation. If None, no attempt at reloading
      or storing the dict is made.
    log_stats: whether to print statistics about the sampling graph and the
      three split subgraphs

  Returns:
    A tuple of the following:
    splits: A dict mapping each Split in split_enum to the list of Synsets
      belonging to the subgraph for that split.
    split_num_images: A dict mapping each Split in split_enum to a dict for the
      corresponding split that maps each node in its subgraph to the number of
      images in the subgraph of that node.
    sampling_graph: A set of the Synsets that belong to the DAG described above
    synsets_2012: The list of Synsets of classes of ILSVRC 2012
    num_synset_2012_images: A dict mapping each WordNet id of ILSVRC 2012 to its
      number of images
  """

  # Create Synsets for all ImageNet synsets (82115 in total).
  data_root = FLAGS.ilsvrc_2012_data_root
  synsets = {}
  path_to_words = FLAGS.path_to_words
  if not path_to_words:
    path_to_words = os.path.join(data_root, 'words.txt')
  with tf.io.gfile.GFile(path_to_words) as f:
    for line in f:
      wn_id, words = line.rstrip().split('\t')
      synsets[wn_id] = Synset(wn_id, words, set(), set())

  # Populate the parents / children arrays of these Synsets.
  path_to_is_a = FLAGS.path_to_is_a
  if not path_to_is_a:
    path_to_is_a = os.path.join(data_root, 'wordnet.is_a.txt')
  with tf.io.gfile.GFile(path_to_is_a, 'r') as f:
    for line in f:
      parent, child = line.rstrip().split(' ')
      synsets[parent].children.add(synsets[child])
      synsets[child].parents.add(synsets[parent])

  # Get the WordNet id's of the synsets of ILSVRC 2012.
  wn_ids_2012 = tf.io.gfile.listdir(data_root)
  wn_ids_2012 = set(
      entry for entry in wn_ids_2012
      if tf.io.gfile.isdir(os.path.join(data_root, entry)))
  synsets_2012 = [s for s in synsets.values() if s.wn_id in wn_ids_2012]
  assert len(wn_ids_2012) == len(synsets_2012)

  # Get a dict mapping each WordNet id of ILSVRC 2012 to its number of images.
  num_synset_2012_images = get_num_synset_2012_images(path_to_num_leaf_images,
                                                      synsets_2012,
                                                      files_to_skip)

  # Get the graph of all and only the ancestors of the ILSVRC 2012 classes.
  sampling_graph = create_sampling_graph(synsets_2012)

  # Create a dict mapping each node to its reachable leaves.
  spanning_leaves = get_spanning_leaves(sampling_graph)

  # Create a dict mapping each node in sampling graph to the number of images of
  # ILSVRC 2012 synsets that live in the sub-graph rooted at that node.
  num_images = get_num_spanning_images(spanning_leaves, num_synset_2012_images)

  # Create class splits, each with its own sampling graph.
  # Choose roots for the validation and test subtrees (see the docstring of
  # create_splits for more information on how these are used).
  valid_test_roots = {
      'valid': get_synset_by_wnid('n02075296', sampling_graph),  # 'carnivore'
      'test':
          get_synset_by_wnid('n03183080', sampling_graph)  # 'device'
  }
  # The valid_test_roots returned here correspond to the same Synsets as in the
  # above dict, but are the copied versions of them for each subgraph.
  splits, valid_test_roots = create_splits(
      spanning_leaves, split_enum, valid_test_roots=valid_test_roots)

  # Compute num_images for each split.
  split_num_images = {}
  split_num_images[split_enum.TRAIN] = get_num_spanning_images(
      get_spanning_leaves(splits[split_enum.TRAIN]), num_synset_2012_images)
  split_num_images[split_enum.VALID] = get_num_spanning_images(
      get_spanning_leaves(splits[split_enum.VALID]), num_synset_2012_images)
  split_num_images[split_enum.TEST] = get_num_spanning_images(
      get_spanning_leaves(splits[split_enum.TEST]), num_synset_2012_images)

  # Compute statistics.
  if log_stats:
    imagenet_stats.log_graph_stats(
        sampling_graph,
        num_images,
        get_leaves,
        get_spanning_leaves,
        graph_name='all')
    imagenet_stats.log_graph_stats(
        splits[split_enum.TRAIN],
        split_num_images[split_enum.TRAIN],
        get_leaves,
        get_spanning_leaves,
        graph_name='train')
    imagenet_stats.log_graph_stats(
        splits[split_enum.VALID],
        split_num_images[split_enum.VALID],
        get_leaves,
        get_spanning_leaves,
        graph_name='valid')
    imagenet_stats.log_graph_stats(
        splits[split_enum.TEST],
        split_num_images[split_enum.TEST],
        get_leaves,
        get_spanning_leaves,
        graph_name='test')
    # Stats relevant to analysis of fine-graindness.
    imagenet_stats.log_stats_finegrainedness(
        splits[split_enum.TRAIN],
        get_leaves,
        get_lowest_common_ancestor,
        graph_name='train',
        path='longest')
    imagenet_stats.log_stats_finegrainedness(
        splits[split_enum.TEST],
        get_leaves,
        get_lowest_common_ancestor,
        graph_name='test',
        path='longest')

  # Note that spanning_leaves and num_images can easily be created from
  # sampling_graph if required.
  return (splits, split_num_images, sampling_graph, synsets_2012,
          num_synset_2012_images, valid_test_roots)
