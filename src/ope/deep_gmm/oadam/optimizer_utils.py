"""Utils for OAdam optimizers.

Copied from: sonnet/v2/src/optimizers/optimizer_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence

import tensorflow.compat.v2 as tf

from src.ope.deep_gmm.oadam import replicator
from src.ope.deep_gmm.oadam import types

# Sonnet only supports a subset of distribution strategies since it makes use of
# a simplified update model and replica local variables.
# TODO(cjfj,petebu,tomhennigan) Add async parameter server strategy when needed.
# TODO(cjfj,petebu,tomhennigan) Add sync multi-worker GPU strategy when needed.
_SUPPORTED_STRATEGIES = (
    tf.distribute.OneDeviceStrategy,
    replicator.Replicator,
    replicator.TpuReplicator,
)


def check_distribution_strategy():
  if tf.distribute.has_strategy():
    strategy = tf.distribute.get_strategy()
    if not isinstance(strategy, _SUPPORTED_STRATEGIES):
      raise ValueError("Sonnet optimizers are not compatible with `{}`. "
                       "Please use one of `{}` instead.".format(
                           strategy.__class__.__name__, "`, `".join(
                               s.__name__ for s in _SUPPORTED_STRATEGIES)))


def check_updates_parameters(updates: Sequence[types.ParameterUpdate],
                             parameters: Sequence[tf.Variable]):
  if len(updates) != len(parameters):
    raise ValueError("`updates` and `parameters` must be the same length.")
  if not parameters:
    raise ValueError("`parameters` cannot be empty.")
  if all(x is None for x in updates):
    raise ValueError("No updates provided for any parameter.")


def check_same_dtype(update: types.ParameterUpdate, parameter: tf.Variable):
  if update.dtype != parameter.dtype:
    raise ValueError(
        "DType of update {!r} is not equal to that of parameter {!r}".format(
            update, parameter))


def deduplicate_indexed_slices(indexed_slice: tf.IndexedSlices):
  """Sums `values` associated with any non-unique `indices`.

  Args:
    indexed_slice: An indexed slice with potentially duplicated indices.

  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  values, indices = indexed_slice.values, indexed_slice.indices
  unique_indices, new_index_positions = tf.unique(indices)
  summed_values = tf.math.unsorted_segment_sum(values, new_index_positions,
                                               tf.shape(unique_indices)[0])
  return summed_values, unique_indices
