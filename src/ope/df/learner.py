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

from src.utils.tf_linear_reg_utils import fit_linear, linear_reg_loss, linear_reg_pred, add_const_col


class DFLearner(acme.Learner, tf2_savers.TFSaveable):
    """DFIVLearner.

    This is the learning component of a DFIV learner. IE it takes a dataset as
    input and implements update functionality to learn from this dataset.
    Optionally it takes a replay client as well to allow for updating of
    priorities.
    """

    def __init__(self,
                 value_func: snt.Module,
                 instrumental_feature: snt.Module,
                 policy_net: snt.Module,
                 discount: float,
                 value_learning_rate: float,
                 instrumental_learning_rate: float,
                 value_l2_reg: float,
                 instrumental_l2_reg: float,
                 stage1_reg: float,
                 stage2_reg: float,
                 instrumental_iter: int,
                 value_iter: int,
                 dataset: tf.data.Dataset,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True):
        """Initializes the learner.

        Args:
          value_feature: value function network
          instrumental_feature: dual function network.
          policy_net: policy network.
          discount: global discount.
          value_learning_rate: learning rate for the treatment_net update.
          instrumental_learning_rate: learning rate for the instrumental_net update.
          value_l2_reg: l2 reg for value feature
          instrumental_l2_reg: l2 reg for instrumental
          stage1_reg: ridge regularizer for stage 1 regression
          stage2_reg: ridge regularizer for stage 2 regression
          instrumental_iter: number of iteration for instrumental net
          value_iter: number of iteration for value function,
          dataset: dataset to learn from.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.instrumental_iter = instrumental_iter
        self.value_iter = value_iter
        self.discount = discount
        self.value_l2_reg = value_l2_reg
        self.instrumental_reg = instrumental_l2_reg

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.policy = policy_net
        self._value_func_optimizer = snt.optimizers.Adam(value_learning_rate)
        self._instrumental_func_optimizer = snt.optimizers.Adam(instrumental_learning_rate)

        self._variables = [
            value_func.trainable_variables,
            instrumental_feature.trainable_variables,
        ]
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        self.data = None

        # Create a snapshotter object.
        if checkpoint:
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={'value_func': value_func,
                                 'instrumental_feature': instrumental_feature,
                                 }, time_delta_minutes=60.)
        else:
            self._snapshotter = None

    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = None
        stage2_loss = None
        # Pull out the data needed for updates/priorities.

        if self.data is None:
            self.data = next(self._iterator)

        for i in range(self.value_iter):
            self.update_value()

        stage2_loss = self.update_final_weight()
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': 0.0, "stage2_loss": stage2_loss}

        return fetches


    def update_value(self):
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, next_obs_2nd, _ = self.data
        next_action_2nd = self.policy(next_obs_2nd)

        discount_2nd = tf.expand_dims(discount_2nd, axis=1) * self.discount

        reg = snt.regularizers.L2(self.value_l2_reg)
        with tf.GradientTape() as tape:
            next_feature = self.value_feature(obs=next_obs_2nd, action=next_action_2nd)
            current_feature = self.value_feature(obs=current_obs_2nd, action=action_2nd)
            predicted_feature = current_feature - discount_2nd * next_feature

            loss = linear_reg_loss(tf.expand_dims(reward_2nd, -1), predicted_feature, self.stage2_reg)
            loss = loss + reg(self.value_feature.trainable_variables)

        gradient = tape.gradient(loss, self.value_feature.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_feature.trainable_variables)
        return loss

    def update_final_weight(self):
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, next_obs_2nd, _ = self.data
        next_action_2nd = self.policy(next_obs_2nd)

        weight = tf.expand_dims(discount_2nd + (1.0 - discount_2nd) * 5, axis=1)
        discount_2nd = tf.expand_dims(discount_2nd, axis=1) * self.discount

        next_feature = self.value_feature(obs=next_obs_2nd, action=next_action_2nd)
        current_feature = self.value_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature = current_feature - discount_2nd * next_feature
        self.value_func._weight = fit_linear(tf.expand_dims(reward_2nd, -1)*weight, predicted_feature*weight, self.stage2_reg)
        return linear_reg_loss(tf.expand_dims(reward_2nd, -1)*weight, predicted_feature*weight, self.stage2_reg)

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
            'value_feature': self.value_feature,
            'instrumental_feature': self.instrumental_feature,
            'value_opt': self._value_func_optimizer,
            'dual_opt': self._instrumental_func_optimizer,
            'num_steps': self._num_steps
        }
