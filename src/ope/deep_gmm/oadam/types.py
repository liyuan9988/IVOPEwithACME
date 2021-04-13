"""Type aliases for Sonnet.

Copied from sonnet/v2/src/types.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from typing import Callable, Iterable, Mapping, Optional, Sequence, Text, Tuple, Union

# Parameter update type, used by optimizers.
ParameterUpdate = Optional[Union[tf.Tensor, tf.IndexedSlices]]

# Objects that can be treated like tensors (in TF2).
TensorLike = Union[np.ndarray, tf.Tensor, tf.Variable]

# Note that we have no way of statically verifying the tensor's shape.
BoolLike = Union[bool, np.bool, TensorLike]
IntegerLike = Union[int, np.integer, TensorLike]
FloatLike = Union[float, np.floating, TensorLike]

ShapeLike = Union[int, Sequence[int], tf.TensorShape]

# Note that this is effectively treated as `Any`; see b/109648354.
TensorNest = Union[TensorLike, Iterable['TensorNest'],
                   Mapping[Text, 'TensorNest'],]  # pytype: disable=not-supported-yet

ActivationFn = Callable[[TensorLike], TensorLike]
Axis = Union[int, slice, Sequence[int]]
GradFn = Callable[[tf.Tensor], Tuple[tf.Tensor, Optional[tf.Tensor]]]
