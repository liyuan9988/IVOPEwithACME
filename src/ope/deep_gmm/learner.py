# python3
r"""Learners of generalized method of moments with deep networks.

References:

- DeepGMMLearner:
  Andrew Bennett, Nathan Kallus, and Tobias Schnabel. Deep generalized method of
  moments forinstrumental variable analysis.  InAdvances in Neural Information
  Processing Systems, pages3559–3569, 2019.

- AdversarialSEMLearner
  L. Liao, Y. L. Chen, Z. Yang, B. Dai, Z. Wang and M. Kolar, 2020. Provably
  efficient neural estimation of structural equation model: An adversarial
  approach. arXiv preprint arXiv:2007.01290.

- AGMMLearner
  Dikkala, N., Lewis, G., Mackey, L. and Syrgkanis, V., 2020. Minimax estimation
  of conditional moment models. Advances in Neural Information Processing
  Systems, 33.
"""

import datetime
from typing import Dict, List, Optional

from acme import core
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow.compat.v2 as tf

from src.ope.deep_gmm.oadam import oadam

# Default Acme checkpoint TTL is 5 days.
_CHECKPOINT_TTL = int(datetime.timedelta(days=30).total_seconds())


def _optimizer_class(name):
  return oadam.OAdam if name == 'OAdam' else getattr(snt.optimizers, name)


def _td_error(critic_network, o_tm1, a_tm1, r_t, d_t, o_t, a_t) -> tf.Tensor:
  """Computes TD error."""
  q_tm1 = critic_network(o_tm1, a_tm1)
  q_t = critic_network(o_t, a_t)
  if q_tm1.shape != q_t.shape:
    raise ValueError(f'Shape of q_tm1 {q_tm1.shape.as_list()} does not '
                     f'match that of q_t {q_t.shape.as_list()}')
  d_t = tf.reshape(d_t, q_tm1.shape)
  g = q_tm1 - d_t * q_t
  r_t = tf.reshape(r_t, q_tm1.shape)
  td_error = r_t - g
  return td_error, q_tm1, q_t


def _orthogonal_regularization(network: snt.Module,) -> tf.Tensor:
  """Copied from third_party/py/dice_rl/estimators/neural_dice.py."""
  reg = 0.
  for w in network.trainable_variables:
    if w.name.endswith('/w:0'):
      prod = tf.matmul(tf.transpose(w), w)
      reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
  return reg


def _l2_regularization(network: snt.Module,) -> tf.Tensor:
  """Copied from third_party/py/dice_rl/estimators/neural_dice.py."""
  reg = 0.
  for w in network.trainable_variables:
    if w.name.endswith('/w:0'):
      reg += tf.reduce_sum(tf.square(w))
  return reg


class DeepGMMLearnerBase(core.Learner):
  """Deep GMM learner base class."""

  def __init__(self,
               policy_network: snt.Module,
               critic_network: snt.Module,
               f_network: snt.Module,
               discount: float,
               dataset: tf.data.Dataset,
               use_tilde_critic: bool,
               tilde_critic_network: snt.Module = None,
               tilde_critic_update_period: int = None,
               critic_optimizer_class: str = 'OAdam',
               critic_lr: float = 1e-4,
               critic_beta1: float = 0.5,
               critic_beta2: float = 0.9,
               f_optimizer_class: str = 'OAdam',
               f_lr: float = None,  # Either f_lr or f_lr_multiplier must be
                                    # None.
               f_lr_multiplier: Optional[float] = 1.0,
               f_beta1: float = 0.5,
               f_beta2: float = 0.9,
               critic_regularizer: float = 0.0,
               f_regularizer: float = 1.0,  # Ignored if use_tilde_critic = True
               critic_ortho_regularizer: float = 0.0,
               f_ortho_regularizer: float = 0.0,
               critic_l2_regularizer: float = 0.0,
               f_l2_regularizer: float = 0.0,
               checkpoint_interval_minutes: int = 10.0,
               clipping: bool = True,
               clipping_action: bool = True,
               bre_check_period: int = 0,  # Bellman residual error check.
               bre_check_num_actions: int = 0,  # Number of sampled actions.
               dev_dataset: tf.data.Dataset = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True):

    self._policy_network = policy_network
    self._critic_network = critic_network
    self._f_network = f_network
    self._discount = discount
    self._clipping = clipping
    self._clipping_action = clipping_action
    self._bre_check_period = bre_check_period
    self._bre_check_num_actions = bre_check_num_actions

    # Development dataset for hyper-parameter selection.
    self._dev_dataset = dev_dataset
    self._dev_actions_dataset = self._sample_actions()

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Necessary to track when to update tilde critic networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    self._use_tilde_critic = use_tilde_critic
    self._tilde_critic_network = tilde_critic_network
    self._tilde_critic_update_period = tilde_critic_update_period
    if use_tilde_critic and tilde_critic_update_period is None:
      raise ValueError('tilde_critic_update_period must be provided if '
                       'use_tilde_critic is True.')

    # Batch dataset and create iterator.
    self._iterator = iter(dataset)

    # Create optimizers if they aren't given.
    self._critic_optimizer = _optimizer_class(critic_optimizer_class)(
        critic_lr, beta1=critic_beta1, beta2=critic_beta2)

    if f_lr is not None:
      if f_lr_multiplier is not None:
        raise ValueError(f'Either f_lr ({f_lr}) or f_lr_multiplier '
                         f'({f_lr_multiplier}) must be None.')
    else:
      f_lr = f_lr_multiplier * critic_lr
    # Prevent unreasonable value in hyper-param search.
    f_lr = max(min(f_lr, 1e-2), critic_lr)

    self._f_optimizer = _optimizer_class(f_optimizer_class)(
        f_lr, beta1=f_beta1, beta2=f_beta2)

    # Regularization on network values.
    self._critic_regularizer = critic_regularizer
    self._f_regularizer = f_regularizer

    # Orthogonal regularization strength.
    self._critic_ortho_regularizer = critic_ortho_regularizer
    self._f_ortho_regularizer = f_ortho_regularizer

    # L2 regularization strength.
    self._critic_l2_regularizer = critic_l2_regularizer
    self._f_l2_regularizer = f_l2_regularizer

    # Expose the variables.
    self._variables = {
        'critic': self._critic_network.variables,
    }

    # Create a checkpointer object.
    self._checkpointer = None
    self._snapshotter = None

    if checkpoint:
      objects_to_save = {
          'counter': self._counter,
          'critic': self._critic_network,
          'f': self._f_network,
          'tilde_critic': self._tilde_critic_network,
          'critic_optimizer': self._critic_optimizer,
          'f_optimizer': self._f_optimizer,
          'num_steps': self._num_steps,
      }
      self._checkpointer = tf2_savers.Checkpointer(
          objects_to_save={k: v for k, v in objects_to_save.items()
                           if v is not None},
          time_delta_minutes=checkpoint_interval_minutes,
          checkpoint_ttl_seconds=_CHECKPOINT_TTL)
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={
              'critic': self._critic_network,
              'f': self._f_network,
          },
          time_delta_minutes=60.)

  def _check_bellman_residual_error(self, q_tm1, r_t, d_t, o_t):
    """Estimate of mean squared Bellman residual error."""
    # Ignore stochasticity in s'.
    # E[err(a')|s,a,s']^2 = E[\bar{err}]^2 - Var(err) / N.
    if (self._bre_check_period > 0 and
        tf.math.mod(self._num_steps, self._bre_check_period) == 0):
      q_t_sum = 0.
      q_t_sq_sum = 0.
      for _ in range(self._bre_check_num_actions):
        a_t = self._policy_network(o_t)
        if self._clipping_action:
          if not a_t.dtype.is_floating:
            raise ValueError(f'Action dtype ({a_t.dtype}) is not floating.')
          a_t = tf.clip_by_value(a_t, -1., 1.)
        q_t = self._critic_network(o_t, a_t)
        if q_tm1.shape != q_t.shape:
          raise ValueError(f'Shape of q_tm1 {q_tm1.shape.as_list()} does not '
                           f'match that of q_t {q_t.shape.as_list()}')
        q_t_sum += q_t
        q_t_sq_sum += tf.square(q_t)
      q_t_mean = q_t_sum / self._bre_check_num_actions
      q_t_var = q_t_sq_sum / self._bre_check_num_actions - tf.square(q_t_mean)
      d_t = tf.reshape(d_t, q_tm1.shape)
      r_t = tf.reshape(r_t, q_tm1.shape)
      td_error = r_t - (q_tm1 - d_t * q_t_mean)
      td_mse = tf.reduce_mean(tf.square(td_error))
      td_var = tf.reduce_mean(tf.square(d_t) * tf.reduce_mean(q_t_var))
      bre_mse = td_mse - td_var / self._bre_check_num_actions
    else:
      bre_mse = tf.constant(0., dtype=tf.float32)
    return bre_mse

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    sample = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = sample.data[:5]
    a_t = self._policy_network(o_t)
    if self._clipping_action:
      if not a_t.dtype.is_floating:
        raise ValueError(f'Action dtype ({a_t.dtype}) is not floating.')
      a_t = tf.clip_by_value(a_t, -1., 1.)

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=d_t.dtype)
    d_t = discount * d_t

    if self._use_tilde_critic:
      tilde_td_error = _td_error(
          self._tilde_critic_network, o_tm1, a_tm1, r_t, d_t, o_t, a_t)[0]
      # In the same shape as tilde_td_error.
      f_regularizer = 0.25 * tf.square(tilde_td_error)
    else:
      # Scalar.
      tilde_td_error = 0.
      f_regularizer = self._f_regularizer

    with tf.GradientTape() as tape:
      td_error, q_tm1, q_t = _td_error(
          self._critic_network, o_tm1, a_tm1, r_t, d_t, o_t, a_t)
      f = self._f_network(o_tm1, a_tm1)
      if f.shape != td_error.shape:
        raise ValueError(f'Shape of f {f.shape.as_list()} does not '
                         f'match that of td_error {td_error.shape.as_list()}')

      moment = tf.reduce_mean(f * td_error)
      f_reg_loss = tf.reduce_mean(f_regularizer * tf.square(f))
      u = moment - f_reg_loss

      # Add regularizations.

      # Regularization on critic net output values.
      if self._critic_regularizer > 0.:
        critic_reg_loss = self._critic_regularizer * (
            tf.reduce_mean(tf.square(q_tm1)) +
            tf.reduce_mean(tf.square(q_t))) / 2.
      else:
        critic_reg_loss = 0.

      # Ortho regularization on critic net.
      if self._critic_ortho_regularizer > 0.:
        critic_ortho_reg_loss = (
            self._critic_ortho_regularizer *
            _orthogonal_regularization(self._critic_network))
      else:
        critic_ortho_reg_loss = 0.

      # Ortho regularization on f net.
      if self._f_ortho_regularizer > 0.:
        f_ortho_reg_loss = (
            self._f_ortho_regularizer *
            _orthogonal_regularization(self._f_network))
      else:
        f_ortho_reg_loss = 0.

      # L2 regularization on critic net.
      if self._critic_l2_regularizer > 0.:
        critic_l2_reg_loss = (
            self._critic_l2_regularizer *
            _l2_regularization(self._critic_network))
      else:
        critic_l2_reg_loss = 0.

      # L2 regularization on f net.
      if self._f_l2_regularizer > 0.:
        f_l2_reg_loss = (
            self._f_l2_regularizer *
            _l2_regularization(self._f_network))
      else:
        f_l2_reg_loss = 0.

      loss = (u + critic_reg_loss
              + critic_ortho_reg_loss - f_ortho_reg_loss
              + critic_l2_reg_loss - f_l2_reg_loss)

    bre_mse = self._check_bellman_residual_error(q_tm1, r_t, d_t, o_t)

    # Get trainable variables.
    critic_variables = self._critic_network.trainable_variables
    f_variables = self._f_network.trainable_variables

    # Compute gradients.
    gradients = tape.gradient(loss, critic_variables + f_variables)
    critic_gradients = gradients[:len(critic_variables)]
    f_gradients = gradients[len(critic_variables):]

    # Maybe clip gradients.
    if self._clipping:
      # # clip_by_global_norm
      # critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]
      # f_gradients = tf.clip_by_global_norm(f_gradients, 40.)[0]

      # clip_by_value
      critic_gradients = [tf.clip_by_value(g, -1.0, 1.0)
                          for g in critic_gradients]
      f_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in f_gradients]

    # Apply critic gradients to minimize the loss.
    self._critic_optimizer.apply(critic_gradients, critic_variables)

    # Apply f gradients to maximize the loss.
    f_gradients = [-g for g in f_gradients]
    self._f_optimizer.apply(f_gradients, f_variables)

    if self._use_tilde_critic:
      if tf.math.mod(self._num_steps, self._tilde_critic_update_period) == 0:
        source_variables = self._critic_network.variables
        tilde_variables = self._tilde_critic_network.variables

        # Make online -> tilde network update ops.
        for src, dest in zip(source_variables, tilde_variables):
          dest.assign(src)
    self._num_steps.assign_add(1)

    # Losses to track.
    results = {
        'loss': loss,
        'u': u,
        'f_reg_loss': f_reg_loss,
        'td_mse': tf.reduce_mean(tf.square(td_error)),
        'f_ms': tf.reduce_mean(tf.square(f)),
        'moment': moment,
        'global_steps': tf.convert_to_tensor(self._num_steps),
        'bre_mse': bre_mse,
    }
    if self._use_tilde_critic:
      results.update({
          'tilde_td_mse': tf.reduce_mean(tf.square(tilde_td_error))})
    if self._critic_regularizer > 0.:
      results.update({'critic_reg_loss': critic_reg_loss})
    if self._critic_ortho_regularizer > 0.:
      results.update({'critic_ortho_reg_loss': critic_ortho_reg_loss})
    if self._f_ortho_regularizer > 0.:
      results.update({'f_ortho_reg_loss': f_ortho_reg_loss})
    if self._critic_l2_regularizer > 0.:
      results.update({'critic_l2_reg_loss': critic_l2_reg_loss})
    if self._f_l2_regularizer > 0.:
      results.update({'f_l2_reg_loss': f_l2_reg_loss})
    return results

  def num_steps(self):
    return self._num_steps.numpy()

  def step(self):
    # Run the learning step.
    fetches = self._step()

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)
    fetches.update(counts)

    # Checkpoint and attempt to write the logs.
    if self._checkpointer is not None:
      self._checkpointer.save()
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(fetches)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables[name]) for name in names]

  def _sample_actions(self):
    if self._dev_dataset is not None:
      a_ts = []
      for sample in self._dev_dataset:
        o_t = sample.data[4]
        a_t = self._policy_network(o_t)
        if self._clipping_action:
          if not a_t.dtype.is_floating:
            raise ValueError(f'Action dtype ({a_t.dtype}) is not floating.')
          a_t = tf.clip_by_value(a_t, -1., 1.)
        a_ts.append(a_t)
      return tf.data.Dataset.from_tensor_slices(a_ts)
    else:
      return None

  def dev_td_error_and_f_values(self):
    """Return TD error and f values from the dev dataset."""
    td_errors = []
    fs = []
    for sample, a_t in zip(self._dev_dataset, self._dev_actions_dataset):
      o_tm1, a_tm1, r_t, d_t, o_t = sample.data[:5]

      # Cast the additional discount to match the environment discount dtype.
      discount = tf.cast(self._discount, dtype=d_t.dtype)
      d_t = discount * d_t

      td_error, _, _ = _td_error(
          self._critic_network, o_tm1, a_tm1, r_t, d_t, o_t, a_t)
      f = self._f_network(o_tm1, a_tm1)
      if f.shape != td_error.shape:
        raise ValueError(f'Shape of f {f.shape.as_list()} does not '
                         f'match that of td_error {td_error.shape.as_list()}')

      td_errors.append(td_error)
      fs.append(f)
    td_errors = tf.concat(td_errors, axis=0)
    fs = tf.concat(fs, axis=0)
    return {
        'td_errors': td_errors,
        'fs': fs,
        'global_steps': tf.convert_to_tensor(self._num_steps),
    }


class DeepGMMLearner(DeepGMMLearnerBase):
  r"""Deep GMM learner.

  Reference:
  A. Bennett, N. Kallus, and T. Schnabel. Deep generalized method of moments for
  instrumental variable analysis. In Advances in Neural Information Processing
  Systems 32, pages 3564–3574. 2019.

  Open source code: https://github.com/CausalML/DeepGMM

  Objective (with 0.25 scaling on the original value)
  argmin_q sup_f U
  U = E[f * (R - Q_tm1 + gamma * Q_t)]
      - 0.25 * E[f^2 * (R - \tilde{Q}_tm1 + gamma * \tilde{Q}_t)^2]

  Additional optional losses include:
    - L2 regularization on Q values.
    - Orthogonal loss for Q net.
    - Orthogonal loss for f net.

  If not using the tilde Q net, one can specify a L2 regularizer for f
  values with a constant strength to replace the second term in U.
  """

  def __init__(self,
               policy_network: snt.Module,
               critic_network: snt.Module,
               f_network: snt.Module,
               discount: float,
               dataset: tf.data.Dataset,
               tilde_critic_network: snt.Module,
               tilde_critic_update_period: int = 1,
               critic_optimizer_class: str = 'OAdam',
               critic_lr: float = 1e-4,
               critic_beta1: float = 0.5,  # From open-sourced code.
               critic_beta2: float = 0.9,  # From open-sourced code.
               f_optimizer_class: str = 'OAdam',
               f_lr: float = None,  # Either f_lr or f_lr_multiplier must be
                                    # None.
               f_lr_multiplier: Optional[float] = 1.0,
               f_beta1: float = 0.5,  # From open-sourced code.
               f_beta2: float = 0.9,  # From open-sourced code.
               checkpoint_interval_minutes: int = 10.0,
               clipping: bool = True,
               clipping_action: bool = True,
               bre_check_period: int = 0,  # Bellman residual error check.
               bre_check_num_actions: int = 0,  # Number of sampled actions.
               dev_dataset: tf.data.Dataset = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True):
    super().__init__(
        policy_network=policy_network,
        critic_network=critic_network,
        f_network=f_network,
        discount=discount,
        dataset=dataset,
        use_tilde_critic=True,
        tilde_critic_network=tilde_critic_network,
        tilde_critic_update_period=tilde_critic_update_period,
        critic_optimizer_class=critic_optimizer_class,
        critic_lr=critic_lr,
        critic_beta1=critic_beta1,
        critic_beta2=critic_beta2,
        f_optimizer_class=f_optimizer_class,
        f_lr=f_lr,
        f_lr_multiplier=f_lr_multiplier,
        f_beta1=f_beta1,
        f_beta2=f_beta2,
        checkpoint_interval_minutes=checkpoint_interval_minutes,
        clipping=clipping,
        clipping_action=clipping_action,
        bre_check_period=bre_check_period,
        bre_check_num_actions=bre_check_num_actions,
        dev_dataset=dev_dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint)


class AdversarialSEMLearner(DeepGMMLearnerBase):
  """Adversarial SEM learner.

  Reference:
  L. Liao, Y. L. Chen, Z. Yang, B. Dai, Z. Wang and M. Kolar, 2020. Provably
  efficient neural estimation of structural equation model: An adversarial
  approach. arXiv preprint arXiv:2007.01290.
  """

  def __init__(self,
               policy_network: snt.Module,
               critic_network: snt.Module,
               f_network: snt.Module,
               discount: float,
               dataset: tf.data.Dataset,
               critic_optimizer_class: str = 'OAdam',
               critic_lr: float = 1e-4,
               critic_beta1: float = 0.,
               critic_beta2: float = 0.01,
               f_optimizer_class: str = 'OAdam',
               f_lr: float = None,  # Either f_lr or f_lr_multiplier must be
                                    # None.
               f_lr_multiplier: Optional[float] = 1.0,
               f_beta1: float = 0.,
               f_beta2: float = 0.01,
               critic_regularizer: float = 0.0,
               f_regularizer: float = 0.5,  # From paper.
               critic_l2_regularizer: float = 1e-4,
               f_l2_regularizer: float = 1e-4,
               checkpoint_interval_minutes: int = 10.0,
               clipping: bool = True,
               clipping_action: bool = True,
               bre_check_period: int = 0,  # Bellman residual error check.
               bre_check_num_actions: int = 0,  # Number of sampled actions.
               dev_dataset: tf.data.Dataset = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True):
    super().__init__(
        policy_network=policy_network,
        critic_network=critic_network,
        f_network=f_network,
        discount=discount,
        dataset=dataset,
        use_tilde_critic=False,
        critic_optimizer_class=critic_optimizer_class,
        critic_lr=critic_lr,
        critic_beta1=critic_beta1,
        critic_beta2=critic_beta2,
        f_optimizer_class=f_optimizer_class,
        f_lr=f_lr,
        f_lr_multiplier=f_lr_multiplier,
        f_beta1=f_beta1,
        f_beta2=f_beta2,
        critic_regularizer=critic_regularizer,
        f_regularizer=f_regularizer,
        critic_l2_regularizer=critic_l2_regularizer,
        f_l2_regularizer=f_l2_regularizer,
        checkpoint_interval_minutes=checkpoint_interval_minutes,
        clipping=clipping,
        clipping_action=clipping_action,
        bre_check_period=bre_check_period,
        bre_check_num_actions=bre_check_num_actions,
        dev_dataset=dev_dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint)


class AGMMLearner(DeepGMMLearnerBase):
  """AGMM learner.

  Reference:
  Dikkala, N., Lewis, G., Mackey, L. and Syrgkanis, V., 2020. Minimax estimation
  of conditional moment models. Advances in Neural Information Processing
  Systems, 33.

  Open source code: https://github.com/microsoft/AdversarialGMM
  """

  def __init__(self,
               policy_network: snt.Module,
               critic_network: snt.Module,
               f_network: snt.Module,
               discount: float,
               dataset: tf.data.Dataset,
               critic_optimizer_class: str = 'OAdam',
               critic_lr: float = 1e-4,
               critic_beta1: float = 0.,  # From open-sourced code.
               critic_beta2: float = 0.01,  # From open-sourced code.
               f_optimizer_class: str = 'OAdam',
               f_lr: float = None,  # Either f_lr or f_lr_multiplier must be
                                    # None.
               f_lr_multiplier: Optional[float] = 1.0,
               f_beta1: float = 0.,  # From open-sourced code.
               f_beta2: float = 0.01,  # From open-sourced code.
               f_regularizer: float = 1.0,  # From open-sourced code.
               critic_l2_regularizer: float = 1e-4,
               f_l2_regularizer: float = 1e-4,
               checkpoint_interval_minutes: int = 10.0,
               clipping: bool = True,
               clipping_action: bool = True,
               bre_check_period: int = 0,  # Bellman residual error check.
               bre_check_num_actions: int = 0,  # Number of sampled actions.
               dev_dataset: tf.data.Dataset = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True):
    super().__init__(
        policy_network=policy_network,
        critic_network=critic_network,
        f_network=f_network,
        discount=discount,
        dataset=dataset,
        use_tilde_critic=False,
        critic_optimizer_class=critic_optimizer_class,
        critic_lr=critic_lr,
        critic_beta1=critic_beta1,
        critic_beta2=critic_beta2,
        f_optimizer_class=f_optimizer_class,
        f_lr=f_lr,
        f_lr_multiplier=f_lr_multiplier,
        f_beta1=f_beta1,
        f_beta2=f_beta2,
        f_regularizer=f_regularizer,
        critic_l2_regularizer=critic_l2_regularizer,
        f_l2_regularizer=f_l2_regularizer,
        checkpoint_interval_minutes=checkpoint_interval_minutes,
        clipping=clipping,
        clipping_action=clipping_action,
        bre_check_period=bre_check_period,
        bre_check_num_actions=bre_check_num_actions,
        dev_dataset=dev_dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint)
