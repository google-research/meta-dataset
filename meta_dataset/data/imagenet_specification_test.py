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
"""Tests for imagenet_specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meta_dataset.data import imagenet_specification as imagenet_spec
from meta_dataset.data import learning_spec
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

DESIRED_TOY_NUM_VALID_CLASSES = 2
DESIRED_TOY_NUM_TEST_CLASSES = 1
TOY_MARGIN = 1


def create_toy_graph():
  synsets = {}
  for wn_id, name in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
    synsets[name] = imagenet_spec.Synset(wn_id, name, set(), set())

  # (parent-child) tuples
  is_a_relations = [('a', 'b'), ('a', 'c'), ('b', 'g'), ('c', 'd'), ('c', 'e'),
                    ('e', 'f'), ('e', 'h')]
  # The graph is a tree that looks like:
  #        a
  #    b       c
  #  g       d   e
  #             h f
  for t in is_a_relations:
    parent, child = t
    synsets[parent].children.add(synsets[child])
    synsets[child].parents.add(synsets[parent])

  subset = ['f', 'g']
  synsets_subset = [s for s in synsets.values() if s.words in subset]

  # Get the graph of all and only the ancestors of synsets_subset
  graph_nodes = imagenet_spec.create_sampling_graph(synsets_subset)

  # The created graph should contain all and only the ancestors of subset and
  # collapses all nodes that have exactly 1 child. It should be:
  #    a
  #  g   f

  # Create a data structure mapping graph_nodes to their reachable leaves
  spanning_leaves = imagenet_spec.get_spanning_leaves(graph_nodes)
  return graph_nodes, spanning_leaves, synsets_subset


def validate_graph(graph_nodes, subset_synsets, test_instance):
  """Checks that the DAG structure is as expected."""
  # 1) Test that the leaves are all and only the ILSVRC 2012 synsets
  leaves = imagenet_spec.get_leaves(graph_nodes)
  test_instance.assertEqual(len(leaves), len(subset_synsets))
  test_instance.assertEqual(set(leaves), set(subset_synsets))

  # 2) Validate the connectivity
  # If a node is listed as a child of another, the latter must also be listed as
  # a parent of the former, and similarly if a node is listed as a parent of
  # another, the latter must also be listed as a child of the former.
  for n in graph_nodes:
    for c in n.children:
      test_instance.assertIn(n, c.parents)
    for p in n.parents:
      test_instance.assertIn(n, p.children)

  # 3) Check that no node has only 1 child, as it's not possible to create an
  # episode from such a node.
  for n in graph_nodes:
    test_instance.assertNotEqual(len(n.children), 1)

  # 4) Check that the graph is detached from the remaining non-graph synsets.
  # We want to guarantee that by following parent or child pointers of graph
  # nodes we will stay within the graph.
  for n in graph_nodes:
    for c in n.children:
      test_instance.assertIn(c, graph_nodes)
    for p in n.parents:
      test_instance.assertIn(p, graph_nodes)

  # 5) Check that every node in graph nodes is either an ILSVRC 2012 synset or
  # the ancestor of an ILSVRC 2012 synset
  for n in graph_nodes:
    if n in subset_synsets:
      continue
    has_2012_descendent = False
    for s in subset_synsets:
      has_2012_descendent = imagenet_spec.is_descendent(s, n)
      if has_2012_descendent:
        break
    test_instance.assertTrue(has_2012_descendent)


def validate_spanning_leaves(spanning_leaves, subset_synsets, test_instance):
  """Check the correctness of the spanning_leaves dict."""
  # 1) ILSVRC 2012 synsets should span exactly 1 leaf each (themselves)
  for s in subset_synsets:
    test_instance.assertEqual(spanning_leaves[s], set([s]))

  # 2) Checks regarding the number of leaves a node can span in relation to
  # the number of leaves its children span
  # - The number of leaves spanned by a node must be greater than or equal to
  #   the number of leaves spanned by any one of its children. It can't be
  #   less than it by definition. It can only be equal under one condition,
  #   described below.
  # - The total number of leaves spanned by the children of a node should be
  #   greater than or equal to the number of leaves spanned by that node
  #   (it could be greater if some leaf is spanned by more than one child of
  #   that node, which can happen since a node can have multiple parents.)
  for s, leaves in spanning_leaves.items():
    # These checks are not applicable to leafs (since they have no children)
    if not s.children:
      continue
    num_spanning_leaves = len(leaves)
    children_spanning_leaves = []
    num_leaf_children = 0
    for c in s.children:
      if not c.children:
        num_leaf_children += 1
      test_instance.assertGreaterEqual(num_spanning_leaves,
                                       len(spanning_leaves[c]))
      if num_spanning_leaves == len(spanning_leaves[c]):
        # A child c of a synset s may span the same number of leaves as s only
        # if c is a parent of all other children of s. Otherwise it would be
        # that s has a child that spans no synsets, which isn't allowed.
        for c_other in s.children:
          if c == c_other:
            continue
          test_instance.assertIn(c, c_other.parents)
      children_spanning_leaves += spanning_leaves[c]

    sum_of_children_leaves = len(children_spanning_leaves)
    assert num_spanning_leaves <= sum_of_children_leaves
    # The strict equality in the above can only hold under 1 condition:
    # If num_spanning_leaves of node n is less than the sum of spanning leaves
    # of n's children it must be that one of those leaves was double-counted,
    # in that some node along a path from that leaf to n had two parents,
    # causing that leaf to be a descendent of two different children of n.
    if num_spanning_leaves < sum_of_children_leaves:
      test_instance.assertNotEqual(
          len(children_spanning_leaves), len(set(children_spanning_leaves)))
      diff = abs(
          len(children_spanning_leaves) - len(set(children_spanning_leaves)))
      test_instance.assertEqual(
          diff, abs(num_spanning_leaves - sum_of_children_leaves))


def test_lowest_common_ancestor_(lca,
                                 height,
                                 leaf_a,
                                 leaf_b,
                                 test_instance,
                                 root=None):
  """Check the correctness of the lowest common ancestor and its height."""
  # First, check that it is a common ancestor of the longest paths.
  paths_a = imagenet_spec.get_upward_paths_from(leaf_a)
  longest_path_a = paths_a[np.argmax([len(p) for p in paths_a])]
  test_instance.assertIn(lca, longest_path_a)
  paths_b = imagenet_spec.get_upward_paths_from(leaf_b)
  longest_path_b = paths_b[np.argmax([len(p) for p in paths_b])]
  test_instance.assertIn(lca, longest_path_b)

  # Check that the LCA is not higher than the root.
  if root is not None:
    test_instance.assertFalse(imagenet_spec.is_descendent(root, lca))

  # Assert that there is no lower common ancestor than the given lca.
  for height_a, node in enumerate(longest_path_a):
    if node in longest_path_b:
      height_b = longest_path_b.index(node)
      node_height = max(height_a, height_b)
      if node == lca:
        test_instance.assertEqual(node_height, height)
      else:
        # It then must have greater height than the lca's height.
        test_instance.assertGreaterEqual(node_height, height)


def test_lowest_common_ancestor(graph_nodes, test_instance, root=None):
  # Test the computation of the lowest common ancestor of two nodes.
  # Randomly sample two leaves a number of times, find their lowest common
  # ancestor and its height and verify that they are computed correctly.
  leaves = imagenet_spec.get_leaves(graph_nodes)
  for _ in range(10000):
    first_ind = np.random.randint(len(leaves))
    second_ind = np.random.randint(len(leaves))
    while first_ind == second_ind:
      second_ind = np.random.randint(len(leaves))
    leaf_a = leaves[first_ind]
    leaf_b = leaves[second_ind]
    lca, height = imagenet_spec.get_lowest_common_ancestor(leaf_a, leaf_b)
    test_lowest_common_ancestor_(
        lca, height, leaf_a, leaf_b, test_instance, root=root)


def test_get_upward_paths(graph_nodes, test_instance, subgraph_root=None):
  """Test the correctness of imagenet_spec.get_upward_paths_from."""
  # Randomly sample a number of start nodes for get_upward_paths. For each, test
  # the behavior of get_upward_paths when either specifying an end node or not.
  graph_nodes_list = list(graph_nodes)
  num_tested = 0
  while num_tested < 10:
    start_node = np.random.choice(graph_nodes_list)

    if not start_node.parents:
      continue

    # Test the behavior of get_upward_paths_from without an end_node specified.
    paths = imagenet_spec.get_upward_paths_from(start_node)
    for p in paths:
      last_node = p[-1]
      if subgraph_root is not None:
        test_instance.assertEqual(last_node, subgraph_root)
      else:
        # Make sure the last node does not have parents (is a root).
        test_instance.assertLen(last_node.parents, 0)

    # Now test the case where an end node is given which is a direct parent of
    # the start node.
    end_node = np.random.choice(list(start_node.parents))
    paths = imagenet_spec.get_upward_paths_from(start_node, end_node)
    # There should be at least one path in paths that contains only
    # (start_node and end_node).
    found_direct_path = False
    for p in paths:
      if len(p) == 2 and p[0] == start_node and p[1] == end_node:
        found_direct_path = True
    test_instance.assertTrue(found_direct_path)

    num_tested += 1


class GraphCopyTest(tf.test.TestCase):
  """Test the correctness of imagenet_spec.copy_graph."""

  def validate_copy(self, graph, graph_copy):
    """Make sure graph_copy is a correct copy of graph."""
    # Make sure that for each node in graph, there is exactly one node in
    # graph_copy with the same WordNet id
    for n in graph:
      wn_id = n.wn_id
      found_wn_in_copy = False
      for n_copy in graph_copy:
        if n_copy.wn_id == wn_id:
          found_wn_in_copy = True
          break
      self.assertTrue(found_wn_in_copy)

    # Make sure that for every link in graph there is a corresponding link in
    # graph copy (correspondence is assessed via the WordNet id's of the nodes
    # that are being connected).
    graph_parent_child_links = set()
    for s in graph:
      for c in s.children:
        graph_parent_child_links.add((s.wn_id, c.wn_id))
    for s in graph_copy:
      for p, c in graph_parent_child_links:
        # Find the nodes in graph_copy whose wn_id's are p and c
        for n in graph_copy:
          if n.wn_id == c:
            c_node = n
          if n.wn_id == p:
            p_node = n
        self.assertIn(c_node, p_node.children)
        self.assertIn(p_node, c_node.parents)

  def test_toy_graph_copy(self):
    specification = create_toy_graph()
    toy_graph, _, _ = specification
    toy_graph_copy, _ = imagenet_spec.copy_graph(toy_graph)
    self.validate_copy(toy_graph, toy_graph_copy)


class TestGetSynsetsFromIds(tf.test.TestCase):
  """Test the correctness of imagenet_spec.get_synsets_from_ids()."""

  def test_on_toy_graph(self):
    specification = create_toy_graph()
    toy_graph, _, _ = specification
    wn_ids = [5, 0, 6]
    id_to_synset = imagenet_spec.get_synsets_from_ids(wn_ids, toy_graph)
    self.assertEqual(set(id_to_synset.keys()), set(wn_ids))
    for wn_id, synset in id_to_synset.items():
      self.assertEqual(wn_id, synset.wn_id)


class SplitCreationTest(tf.test.TestCase):
  """Test the correctness of imagenet_spec.propose_valid_test_roots."""

  def validate_roots(self, valid_test_roots, spanning_leaves):
    # Make sure that the number of leaves spanned by the proposed valid and test
    # roots are within the allowable margin.
    valid_root, test_root = valid_test_roots['valid'], valid_test_roots['test']
    num_valid_leaves = len(spanning_leaves[valid_root])
    num_test_leaves = len(spanning_leaves[test_root])
    self.assertGreaterEqual(num_valid_leaves,
                            DESIRED_TOY_NUM_VALID_CLASSES - TOY_MARGIN)
    self.assertLessEqual(num_valid_leaves,
                         DESIRED_TOY_NUM_VALID_CLASSES + TOY_MARGIN)
    self.assertGreaterEqual(num_test_leaves,
                            DESIRED_TOY_NUM_TEST_CLASSES - TOY_MARGIN)
    self.assertLessEqual(num_test_leaves,
                         DESIRED_TOY_NUM_TEST_CLASSES + TOY_MARGIN)

  def validate_splits(self, splits, spanning_leaves):
    # Make sure that the classes assigned to each split cover all the leaves
    # and no class is assigned to more than one splits
    train_wn_ids = splits['train']
    valid_wn_ids = splits['valid']
    test_wn_ids = splits['test']
    self.assertFalse(train_wn_ids & valid_wn_ids)
    self.assertFalse(train_wn_ids & test_wn_ids)
    self.assertFalse(test_wn_ids & valid_wn_ids)
    all_wn_ids = train_wn_ids | valid_wn_ids | test_wn_ids
    leaves = imagenet_spec.get_leaves(spanning_leaves.keys())
    self.assertLen(all_wn_ids, len(leaves))  # all covered

  def test_toy_root_proposer(self):
    specification = create_toy_graph()
    _, toy_span_leaves, _ = specification
    valid_test_roots = imagenet_spec.propose_valid_test_roots(
        toy_span_leaves,
        margin=TOY_MARGIN,
        desired_num_valid_classes=DESIRED_TOY_NUM_VALID_CLASSES,
        desired_num_test_classes=DESIRED_TOY_NUM_TEST_CLASSES)
    self.validate_roots(valid_test_roots, toy_span_leaves)

    # returns the lists of id's of classes belonging to each split
    # unlike create_splits which returns the nodes of the three subgraphs that
    # are constructed for the different splits
    splits, _ = imagenet_spec.get_class_splits(
        toy_span_leaves, valid_test_roots=valid_test_roots)
    self.validate_splits(splits, toy_span_leaves)


class ImagenetSpecificationTest(tf.test.TestCase):

  def validate_num_span_images(self, span_leaves, num_span_images):
    # Ensure that the number of images spanned by each node is exactly the
    # number of images living in the leaves spanned by that node
    for node, leaves in span_leaves.items():
      self.assertEqual(num_span_images[node],
                       sum([num_span_images[l] for l in leaves]))

  def validate_splits(self, splits):
    """Check the correctness of the class splits."""
    train_graph = splits[learning_spec.Split.TRAIN]
    valid_graph = splits[learning_spec.Split.VALID]
    test_graph = splits[learning_spec.Split.TEST]

    # Make sure that by following child/parent pointers of nodes of a given
    # split's subgraph, we will reach nodes that also belong to that subgraph.
    def ensure_isolated(nodes):
      for n in nodes:
        for c in n.children:
          self.assertIn(c, nodes)
        for p in n.parents:
          self.assertIn(p, nodes)

    ensure_isolated(train_graph)
    ensure_isolated(valid_graph)
    ensure_isolated(test_graph)

    train_classes = imagenet_spec.get_leaves(train_graph)
    valid_classes = imagenet_spec.get_leaves(valid_graph)
    test_classes = imagenet_spec.get_leaves(test_graph)

    # Ensure that there is no overlap between classes of different splits
    # and that combined they cover all ILSVRC 2012 classes
    all_classes = train_classes + valid_classes + test_classes
    self.assertLen(set(all_classes), 1000)  # all covered
    self.assertLen(set(all_classes), len(all_classes))  # no duplicates

  def test_imagenet_specification(self):
    spec = imagenet_spec.create_imagenet_specification(learning_spec.Split,
                                                       set())
    splits, _, graph_nodes, synsets_2012, num_synset_2012_images, roots = spec
    span_leaves = imagenet_spec.get_spanning_leaves(graph_nodes)
    num_span_images = imagenet_spec.get_num_spanning_images(
        span_leaves, num_synset_2012_images)

    validate_graph(graph_nodes, synsets_2012, self)
    validate_spanning_leaves(span_leaves, synsets_2012, self)
    self.validate_splits(splits)
    self.validate_num_span_images(span_leaves, num_span_images)

    test_lowest_common_ancestor(graph_nodes, self)
    test_get_upward_paths(graph_nodes, self)
    # Make sure that in no sub-tree can the LCA of two chosen leaves of that
    # sub-tree be a node that is an ancestor of the sub-tree's root.
    valid_subgraph, test_subgraph = splits[learning_spec.Split.VALID], splits[
        learning_spec.Split.TEST]
    valid_root, test_root = roots['valid'], roots['test']
    test_lowest_common_ancestor(valid_subgraph, self, valid_root)
    test_get_upward_paths(valid_subgraph, self, valid_root)
    test_lowest_common_ancestor(test_subgraph, self, test_root)
    test_get_upward_paths(test_subgraph, self, test_root)

  def test_toy_graph_specification(self):
    specification = create_toy_graph()
    toy_graph_nodes, toy_span_leaves, toy_synsets_2012 = specification
    validate_graph(toy_graph_nodes, toy_synsets_2012, self)
    validate_spanning_leaves(toy_span_leaves, toy_synsets_2012, self)


if __name__ == '__main__':
  tf.test.main()
