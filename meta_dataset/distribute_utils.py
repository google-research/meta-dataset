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

# Lint as: python2, python3
"""Utilities for working in a tf.distribute context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _pad(tensor, total_size, position):
  """Pad tensor with zeros along its first axis.

  The output will have first dimension total_size, and
  tensor will appear at position position.

  Args:
    tensor: tensor to pad.
    total_size: the output size.
    position: where in the output tensor should appear.

  Returns:
    output: The padded tensor.
  """
  shape_rest = tf.shape(tensor)[1:]
  after_dim = total_size - position - tf.shape(tensor)[0]
  pad_before = tf.zeros(
      tf.concat([[position], shape_rest], axis=0), dtype=tensor.dtype)
  pad_after = tf.zeros(
      tf.concat([[after_dim], shape_rest], axis=0), dtype=tensor.dtype)
  return tf.concat([pad_before, tensor, pad_after], axis=0)


def aggregate(tensor):
  """Aggregate a tensor across distributed replicas.

  If not running in a distributed context, this just returns the input tensor.

  Args:
    tensor: tensor aggregate.

  Returns:
    output: A single tensor with all values across different replicas
      concatenated along the first axis.  The output is in order of gpu index.
  """

  replica_ctx = tf.distribute.get_replica_context()
  if not replica_ctx:
    return tensor
  num = tf.shape(tensor)[0:1]
  padded_num = _pad(num, replica_ctx.num_replicas_in_sync,
                    replica_ctx.replica_id_in_sync_group)
  all_num = replica_ctx.all_reduce('sum', padded_num)
  index_in_output = tf.gather(
      tf.cumsum(tf.concat([[0], all_num], axis=0)),
      replica_ctx.replica_id_in_sync_group)
  total_num = tf.reduce_sum(all_num)
  padded_tensor = _pad(tensor, total_num, index_in_output)
  return replica_ctx.all_reduce('sum', padded_tensor)
