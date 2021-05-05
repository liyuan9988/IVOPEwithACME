# python3
"""Learner of deterministic Bellman residual minimization based on D4PG.

This method assumes a deterministic dynamics and only address confounding from
random policy actions.
"""

import datetime
from typing import Any, Dict, List

from acme import core
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree

# Default Acme checkpoint TTL is 5 days.
_CHECKPOINT_TTL = int(datetime.timedelta(days=30).total_seconds())


class DBRMLearner(core.Learner):
  """Deterministic BRM learner.

  This is the learning component of a D4PG agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(self,
               policy_network: snt.Module,
               critic_network: snt.Module,
               discount: float,
               dataset: tf.data.Dataset,
               critic_lr: float = 1e-4,
               checkpoint_interval_minutes: int = 10.0,
               clipping: bool = True,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True,
               init_observations: Any = None,
               ):

    self._policy_network = policy_network
    self._critic_network = critic_network
    self._discount = discount
    self._clipping = clipping
    self._init_observations = init_observations

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Batch dataset and create iterator.
    self._iterator = iter(dataset)

    self._critic_optimizer = snt.optimizers.Adam(critic_lr)

    # Expose the variables.
    self._variables = {
        'critic': self._critic_network.variables,
    }
    # We remove trailing dimensions to keep same output dimmension
    # as existing FQE based on D4PG. i.e.: (batch_size,).
    critic_mean = snt.Sequential(
        [self._critic_network, lambda t: tf.squeeze(t, -1)])
    self._critic_mean = critic_mean

    # Create a checkpointer object.
    self._checkpointer = None
    self._snapshotter = None

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          objects_to_save=self.state,
          time_delta_minutes=checkpoint_interval_minutes,
          checkpoint_ttl_seconds=_CHECKPOINT_TTL)
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={
              'critic': critic_mean,
          },
          time_delta_minutes=60.)

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    sample = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = sample.data[:5]
    a_t_1 = self._policy_network(o_t)
    a_t_2 = self._policy_network(o_t)

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=d_t.dtype)

    with tf.GradientTape() as tape:
      # Critic learning.
      q_tm1 = self._critic_network(o_tm1, a_tm1)
      q_t_1 = self._critic_network(o_t, a_t_1)
      q_t_2 = self._critic_network(o_t, a_t_2)

      # Critic loss.
      # Squeeze into the shape expected by the td_learning implementation.
      q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
      q_t_1 = tf.squeeze(q_t_1, axis=-1)  # [B]
      q_t_2 = tf.squeeze(q_t_2, axis=-1)  # [B]
      # critic_loss = trfl.td_learning(q_tm1, r_t, discount * d_t, q_t).loss
      critic_loss = (0.5
                     * (r_t + discount * d_t * q_t_1 - q_tm1)
                     * (r_t + discount * d_t * q_t_2 - q_tm1))

      critic_loss = tf.reduce_mean(critic_loss, axis=[0])

    # Get trainable variables.
    critic_variables = self._critic_network.trainable_variables

    # Compute gradients.
    critic_gradients = tape.gradient(critic_loss, critic_variables)

    # Maybe clip gradients.
    if self._clipping:
      # critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]
      critic_gradients = [tf.clip_by_value(g, -1.0, 1.0)
                          for g in critic_gradients]

    # Apply gradients.
    self._critic_optimizer.apply(critic_gradients, critic_variables)

    if self._init_observations is not None:
      if tf.math.mod(self._num_steps, 100) == 0:
        # init_obs = tf.convert_to_tensor(self._init_observations, tf.float32)
        init_obs = tree.map_structure(tf.convert_to_tensor,
                                      self._init_observations)
        init_actions = self._policy_network(init_obs)
        init_critic = tf.reduce_mean(self._critic_mean(init_obs, init_actions))
      else:
        init_critic = tf.constant(0.)
    else:
      init_critic = tf.constant(0.)

    self._num_steps.assign_add(1)

    # Losses to track.
    return {
        'critic_loss': critic_loss,
        'q_s0': init_critic,
    }

  def dev_critic_loss(self, dev_dataset=None):
    critic_loss_sum = 0.
    count = 0.
    for sample in dev_dataset:
      o_tm1, a_tm1, r_t, d_t, o_t = sample.data[:5]
      a_t_1 = self._policy_network(o_t)
      a_t_2 = self._policy_network(o_t)

      # Cast the additional discount to match the environment discount dtype.
      discount = tf.cast(self._discount, dtype=d_t.dtype)

      q_tm1 = self._critic_network(o_tm1, a_tm1)
      q_t_1 = self._critic_network(o_t, a_t_1)
      q_t_2 = self._critic_network(o_t, a_t_2)

      # Critic loss.
      # Squeeze into the shape expected by the td_learning implementation.
      q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
      q_t_1 = tf.squeeze(q_t_1, axis=-1)  # [B]
      q_t_2 = tf.squeeze(q_t_2, axis=-1)  # [B]
      # critic_loss = trfl.td_learning(q_tm1, r_t, discount * d_t, q_t).loss
      critic_loss = (0.5
                     * (r_t + discount * d_t * q_t_1 - q_tm1)
                     * (r_t + discount * d_t * q_t_2 - q_tm1))

      critic_loss_sum += tf.reduce_mean(critic_loss, axis=[0])
      count += 1.
    return critic_loss_sum / count

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

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'critic': self._critic_network,
        'critic_optimizer': self._critic_optimizer,
        'num_steps': self._num_steps,
        'counter': self._counter,
    }
