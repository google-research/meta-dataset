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

# Lint as: python3
"""Testing utilities for meta-dataset.

This module includes functions that inspect class APIs in order to recursively
enumerate arguments to a method of the class as well as arguments passed
implicitly (via *args and/or **kwargs) to the overriden method of any ancestor
classes. These functions are useful for concisely defining tests for classes
that take potentially-overlapping but non-identical arguments.

For examples, see
`meta_dataset.models.experimental.reparameterizable_base_test`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import itertools
import six


def get_argspec(fn):
  """Return the `ArgSpec` namespace for this (potentially wrapped) `fn`."""
  while hasattr(fn, '__wrapped__'):
    fn = fn.__wrapped__
  if six.PY3:
    return inspect.getfullargspec(fn)
  else:
    try:
      return inspect.getargspec(fn)  # pylint: disable=deprecated-method
    except TypeError:
      # Cannot inspect C variables (https://stackoverflow.com/a/7628130), so
      # return an empty `ArgSpec`.
      return inspect.ArgSpec([], None, None, ())


def get_inherited_args(cls, cls_method):
  """Return arguments to `cls_method` of `cls` and all parents of `cls`."""
  # TODO(eringrant): This slicing operation will not work for static methods
  # (for which the first argument is not the object instance).
  args = get_argspec(getattr(cls, cls_method)).args[1:]

  # Inspect all parent classes.
  for parent_cls in cls.__mro__:
    if cls_method in parent_cls.__dict__:
      args += get_argspec(getattr(parent_cls, cls_method)).args[1:]

  return set(args)


get_inherited_init_args = functools.partial(
    get_inherited_args, cls_method='__init__')
get_inherited_call_args = functools.partial(
    get_inherited_args, cls_method='__call__')


def get_valid_kwargs(module_cls, valid_module_init_args,
                     valid_module_call_args):
  """Return all valid kwarg configurations for `module_cls`."""
  init_args = get_inherited_init_args(module_cls)
  call_args = get_inherited_call_args(module_cls)

  valid_init_kwargs = (
      zip(itertools.repeat(init_arg), valid_module_init_args[init_arg])
      for init_arg in init_args
      if init_arg in valid_module_init_args)
  valid_call_kwargs = (
      zip(itertools.repeat(call_arg), valid_module_call_args[call_arg])
      for call_arg in call_args
      if call_arg in valid_module_call_args)

  init_kwargs_combos = (
      dict(combo) for combo in itertools.product(*valid_init_kwargs))
  call_kwargs_combos = (
      dict(combo) for combo in itertools.product(*valid_call_kwargs))

  return itertools.product(init_kwargs_combos, call_kwargs_combos)
