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

"""Modules whose tf.Variable attributes can be temporarily replaced."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import tensorflow.compat.v1 as tf


def is_variable(obj):
  return isinstance(obj, tf.Variable)


def is_trainable_variable(obj):
  return is_variable(obj) and getattr(obj, 'trainable', False)


def is_batch_norm_module(obj):
  return (isinstance(obj, tf.keras.layers.BatchNormalization) or
          'batch_norm' in obj.name)


def corner_case_setattr(target, attr, value):
  """Set `attr` of `target` (a list/dictionary/generic object) to `value`."""
  if isinstance(target, collections.abc.Sequence):
    target[int(attr)] = value
  elif isinstance(target, collections.abc.Mapping):
    target[attr] = value
  else:
    setattr(target, attr, value)


def corner_case_getattr(target, attr):
  """Get `attr` of `target` (a list/dictionary/generic object)."""
  if isinstance(target, collections.abc.Sequence):
    return target[int(attr)]
  elif isinstance(target, collections.abc.Mapping):
    return target[attr]
  else:
    return getattr(target, attr)


def chained_getattr(obj, path):
  """Follow the chain of attributes of `obj` defined by elements of `path`."""
  target = obj
  for attr in path:
    target = corner_case_getattr(target, attr)
  return target


@contextlib.contextmanager
def _patch_getattribute(cls, new_getattribute):
  """Patch the `__getattribute__` method of `cls` with `new_getattribute`."""
  # Adapted from
  # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/custom_getter.py#L31.
  old_getattribute = cls.__getattribute__  # pytype: disable=attribute-error
  cls.__getattribute__ = new_getattribute
  try:
    yield
  finally:
    cls.__getattribute__ = old_getattribute


def _custom_getter(getter, classes=(tf.Module,)):
  """Applies the given `getter` when getting members of given `classes`."""
  # Adapted from
  # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/custom_getter.py#L40.
  stack = contextlib.ExitStack()  # for entering patched contexts
  for cls in classes:
    orig_getattribute = cls.__getattribute__  # pytype: disable=attribute-error

    def new_getattribute(obj, name, orig_getattribute=orig_getattribute):
      attr = orig_getattribute(obj, name)
      return getter(attr)

    stack.enter_context(_patch_getattribute(cls, new_getattribute))
  return stack


class ReparameterizableModule(tf.Module):
  """A module with added functionality to temporarily replace parameters."""

  def reparameterize(self, replacements):
    """Replace attributes of this `ParameterizableModule` with `replacements`.

    Args:
      replacements: A dictionary mapping `Reference`s to replacement Tensors.

    Returns:
        A context manager; attributes are replaced according to
        `replacements` within its context.
    """

    def getter(v_source):
      """Return `v_source` or its replacement if specified in `replacements`."""
      try:
        source_ref = v_source.ref()
        v_sink = replacements[source_ref]
        return v_sink
      except KeyError:
        # `v_source` was not a member of the `replacements` dictionary; this
        # could occur if `ReparameterizableModule.reparameterizables` was
        # called with a predicate that returns `False` of `v_source` and was
        # used to produce the `replacements` dictionary.
        return v_source
      except (AttributeError, TypeError):
        # `v_source` is not a `tf.Tensor` and so does not have an
        # `ref` API; it could not have been replaced.
        return v_source

    return _custom_getter(getter)

  def reparameterizables(self,
                         variable_predicate=is_variable,
                         module_predicate=None,
                         with_path=False):
    """Return reparameterizable elements of this module.

    Args:
      variable_predicate: (Optional) A predicate that filters variables
        belonging to this `ReparameterizableModule` by properties of the
        variable; a variable is returned if it satisfies this predicate.
      module_predicate: (Optional) A predicate that filters the variables
        belonging to this `ReparameterizableModule` by which submodule owns
        them; a variable is returned if any submodules that refer to the
        variable, directly (i.e., as a parent) or indirectly (i.e., as an
        ancestor), satisfy this predicate.
      with_path: (Optional) Whether to include the path to each variable as well
        as the variable itself.

    Returns:
      A flattened generator for elements of the current module and all
      submodules satisfying the given predicates.
    """
    if module_predicate is None:
      # Just return all the variables satisfying `variable_predicate`.
      return self._flatten(
          recursive=True, predicate=variable_predicate, with_path=with_path)

    else:
      # Need to additionally verify `module_predicate` of submodules that
      # directly or indirectly own the variable.
      variables_with_paths = self._flatten(
          recursive=True, predicate=variable_predicate, with_path=True)

      def path_prefixes(path):
        return (path[:i] for i in range(1, len(path)))

      # TODO(eringrant): This function wraps the return value from a call to
      # `tf.Module._flatten`, which only checks predicates of leaf objects and
      # therefore precludes checking containment relationships.
      # However, the performance could be improved by avoiding redoing the BFS
      # within `tf.Module._flatten`.
      def satisfies_module_predicate(variable_with_path):
        """Return True if a module in the path satisfies `module_predicate`."""
        # Check if this module satisfies `module_predicate`.
        if module_predicate(self):
          return True

        # Check if a submodule satisfies `module_predicate`.
        path, _ = variable_with_path
        for path_prefix in path_prefixes(path):
          prefix_module = chained_getattr(self, path_prefix)
          if isinstance(prefix_module,
                        tf.Module) and module_predicate(prefix_module):
            return True

        # No modules or submodules satisfied `module_predicate`.
        return False

      filtered_variables = filter(satisfies_module_predicate,
                                  variables_with_paths)
      if with_path:
        return filtered_variables
      else:
        _, variables = zip(*filtered_variables)
        # De-duplicate.
        return [v_ref.deref() for v_ref in set(v.ref() for v in variables)]
