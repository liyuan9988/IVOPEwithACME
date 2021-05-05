"""Optimistic Adam module.

Forked from snt.optimizers.Adam.

Reference:
[1] Daskalakis C, Ilyas A, Syrgkanis V, Zeng H. Training GANs with Optimism. In
International Conference on Learning Representations 2018 Feb 15.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Sequence, Text, Union

import sonnet as snt
import tensorflow.compat.v2 as tf

from src.ope.deep_gmm.oadam import optimizer_utils
from src.ope.deep_gmm.oadam import types
from src.ope.deep_gmm.oadam import utils


def oadam_update(g, alpha, beta_1, beta_2, epsilon, t, m, v):
  """Implements 'Algorithm 1' from [1]."""
  old_m = m
  old_v = v
  m = beta_1 * m + (1. - beta_1) * g      # Biased first moment estimate.
  v = beta_2 * v + (1. - beta_2) * g * g  # Biased second raw moment estimate.
  m_hat = m / (1. - tf.pow(beta_1, t))    # Bias corrected 1st moment estimate.
  v_hat = v / (1. - tf.pow(beta_2, t))    # Bias corrected 2nd moment estimate.
  if t == 1:
    update = alpha * m_hat / (tf.sqrt(v_hat) + epsilon)
  else:
    # Old bias corrected moment estimates.
    old_m_hat = old_m / (1. - tf.pow(beta_1, (t - 1)))
    old_v_hat = old_v / (1. - tf.pow(beta_2, (t - 1)))
    update = alpha * (2 * m_hat / (tf.sqrt(v_hat) + epsilon)
                      - old_m_hat / (tf.sqrt(old_v_hat) + epsilon))
  return update, m, v


class OAdam(snt.Optimizer):
  """Optimistic Adam optimizer.

  Attributes:
    learning_rate: Step size (``alpha`` in the paper).
    beta1: Exponential decay rate for first moment estimate.
    beta2: Exponential decay rate for second moment estimate.
    epsilon: Small value to avoid zero denominator.
    step: Step count.
    m: Biased first moment estimate (a list with one value per parameter).
    v: Biased second raw moment estimate (a list with one value per parameter).
  """

  def __init__(
      self,
      learning_rate: Union[types.FloatLike, tf.Variable] = 0.001,
      beta1: Union[types.FloatLike, tf.Variable] = 0.9,
      beta2: Union[types.FloatLike, tf.Variable] = 0.999,
      epsilon: Union[types.FloatLike, tf.Variable] = 1e-8,
      name: Optional[Text] = None):
    """Constructs an `OAdam` module.

    Args:
      learning_rate: Step size (``alpha`` in the paper).
      beta1: Exponential decay rate for first moment estimate.
      beta2: Exponential decay rate for second moment estimate.
      epsilon: Small value to avoid zero denominator.
      name: Name of the module.
    """
    super(OAdam, self).__init__(name=name)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    # TODO(petebu): Consider allowing the user to pass in a step.
    self.step = tf.Variable(0, trainable=False, name="t", dtype=tf.int64)
    self.m = []
    self.v = []

  @snt.once
  def _initialize(self, parameters: Sequence[tf.Variable]):
    """First and second order moments are initialized to zero."""
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
    r"""Applies updates to parameters.

    Applies the OAdam update rule for each update, parameter pair:

    .. math::

       \begin{array}{ll}
       m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot update \\
       v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot update^2 \\
       \hat{m}_t = m_t / (1 - \beta_1^t) \\
       \hat{v}_t = v_t / (1 - \beta_2^t) \\
       delta = \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) \\
       param_t = param_{t-1} - delta \\
       \end{array}

    Args:
      updates: A list of updates to apply to parameters. Updates are often
        gradients as returned by :tf:`GradientTape.gradient`.
      parameters: A list of parameters.

    Raises:
      ValueError: If `updates` and `parameters` are empty, have different
        lengths, or have inconsistent types.
    """
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    self.step.assign_add(1)
    for update, param, m_var, v_var in zip(updates, parameters, self.m, self.v):
      if update is None:
        continue

      optimizer_utils.check_same_dtype(update, param)
      learning_rate = tf.cast(self.learning_rate, update.dtype)
      beta_1 = tf.cast(self.beta1, update.dtype)
      beta_2 = tf.cast(self.beta2, update.dtype)
      epsilon = tf.cast(self.epsilon, update.dtype)
      step = tf.cast(self.step, update.dtype)

      if isinstance(update, tf.IndexedSlices):
        # Sparse read our state.
        update, indices = optimizer_utils.deduplicate_indexed_slices(update)
        m = m_var.sparse_read(indices)
        v = v_var.sparse_read(indices)

        # Compute and apply a sparse update to our parameter and state.
        update, m, v = oadam_update(
            g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, t=step, m=m, v=v)
        param.scatter_sub(tf.IndexedSlices(update, indices))
        m_var.scatter_update(tf.IndexedSlices(m, indices))
        v_var.scatter_update(tf.IndexedSlices(v, indices))

      else:
        # Compute and apply a dense update to our parameter and state.
        update, m, v = oadam_update(
            g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, t=step, m=m_var, v=v_var)
        param.assign_sub(update)
        m_var.assign(m)
        v_var.assign(v)
