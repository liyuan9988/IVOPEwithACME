# Lint as: python3
"""DFIV Learner implementation."""

from typing import Dict, List

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf


class SBEEDLearner(acme.Learner, tf2_savers.TFSaveable):
    """SBEEDlearner.

    This is the learning component of a DFIV agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset. Optionally
    it takes a replay client as well to allow for updating of priorities.
    """

    def __init__(self,
                 value_func: snt.Module,
                 dual_func: snt.Module,
                 policy: snt.Module,
                 value_learning_rate: float,
                 dual_learning_rate: float,
                 policy_learning_rate: float,
                 eta: float,
                 entropy_reg: float,
                 dual_iter: int,
                 value_iter: int,
                 policy_iter: int,
                 dataset: tf.data.Dataset,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True):
        """Initializes the learner.

        Args:
          value_func: value function network
          dual_func: dual function network.
          policy_net: policy network.
          value_learning_rate: learning rate for the treatment_net update.
          dual_learning_rate: learning rate for the instrumental_net update.
          policy_learning_rate: learning rate for the policy_net update.
          eta: weight of dual loss. For SBEED, we need eta in (0,1]. Set eta=0 for PCL.
          entropy_reg: entropy regularizer for policy
          dual_iter: number of iteration for dual function
          value_iter: number of iteration for value function,
          policy_iter: number of iteration for policy,
          dataset: dataset to learn from.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.eta = eta
        self.entropy_reg = entropy_reg
        self.dual_iter = dual_iter
        self.value_iter = value_iter
        self.policy_iter = policy_iter

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.dual_func = dual_func
        self.policy = policy
        self._value_func_optimizer = snt.optimizers.Adam(value_learning_rate)
        self._dual_func_optimizer = snt.optimizers.Adam(dual_learning_rate)
        self._policy_optimizer = snt.optimizers.Adam(policy_learning_rate)

        self._variables = [
            value_func.trainable_variables,
            dual_func.trainable_variables,
            policy.trainable_variables,
        ]
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Create a snapshotter object.
        if checkpoint:
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={'value_net': value_func,
                                 'dual_net': dual_func,
                                 'policy_net': policy,
                                 }, time_delta_minutes=60.)
        else:
            self._snapshotter = None

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        dual_loss = None
        primal_loss = None
        # Pull out the data needed for updates/priorities.
        for i in range(self.dual_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
            dual_loss = self.update_dual(o_tm1, a_tm1, r_t, d_t, o_t)

        for i in range(self.policy_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
            primal_loss = self.update_policy(o_tm1, a_tm1, r_t, d_t, o_t)

        for i in range(self.value_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
            primal_loss = self.update_value(o_tm1, a_tm1, r_t, d_t, o_t)

        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'dual_loss': dual_loss, "primal_loss": primal_loss}

        return fetches

    @tf.function
    def update_dual(self, current_obs, action, reward, discount, next_obs):
        td_error = reward + discount * self.value_func(next_obs)
        target = td_error - self.entropy_reg * self.policy(current_obs).log_prob(action)

        with tf.GradientTape() as tape:
            predict = self.dual_func(current_obs, action)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(target, predict)

        gradient = tape.gradient(loss, self.dual_func.trainable_variables)
        self._dual_func_optimizer.apply(gradient, self.dual_func.trainable_variables)
        return loss

    @tf.function
    def update_value(self, current_obs, action, reward, discount, next_obs):
        dual_func_res = self.dual_func(current_obs, action)
        policy_reg = self.entropy_reg * self.policy(current_obs).log_prob(action)
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            td_target = reward + discount * (self.value_func(next_obs))
            target = td_target - policy_reg
            loss1 = mse(target, self.value_func(current_obs))
            loss2 = mse(target, dual_func_res)
            loss = loss1 - self.eta * loss2

        gradient = tape.gradient(loss, self.value_func.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_func.trainable_variables)
        return loss

    @tf.function
    def update_policy(self, current_obs, action, reward, discount, next_obs):

        dual_func_res = self.dual_func(current_obs, action)
        td_target = reward + discount * (self.value_func(next_obs))
        value_func_res = self.value_func(current_obs)
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            policy_reg = self.entropy_reg * self.policy(current_obs).log_prob(action)
            target = td_target - policy_reg
            loss1 = mse(target, value_func_res)
            loss2 = mse(target, dual_func_res)
            loss = loss1 - self.eta * loss2

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self._policy_optimizer.apply(gradient, self.policy.trainable_variables)
        return loss

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Snapshot and attempt to write logs.
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(result)


    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tf2_utils.to_numpy(self._variables)


    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'value_net': self.value_func,
            'dual_net': self.dual_func,
            'policy_net': self.policy,
            'value_opt': self._value_func_optimizer,
            'dual_opt': self._dual_func_optimizer,
            'policy_opt': self._policy_optimizer,
            'num_steps': self._num_steps
        }
