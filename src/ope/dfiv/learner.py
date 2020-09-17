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


class DFIVLearner(acme.Learner, tf2_savers.TFSaveable):
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
        for i in range(self.instrumental_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t, _ = inputs
            stage1_loss = self.update_instrumental(o_tm1, a_tm1, r_t, d_t, o_t)

        for i in range(self.value_iter):
            stage1_input = next(self._iterator)
            stage2_input = next(self._iterator)
            stage2_loss = self.update_value(stage1_input, stage2_input)

        stage1_input = next(self._iterator)
        stage2_input = next(self._iterator)
        self.update_final_weight(stage1_input, stage2_input)
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss, "stage2_loss": stage2_loss}

        return fetches

    def update_instrumental(self, current_obs, action, reward, discount, next_obs):
        next_action = self.policy(next_obs)
        target = self.value_feature(next_obs, next_action)

        with tf.GradientTape() as tape:
            feature = self.instrumental_feature(obs=current_obs, action=action)
            loss = linear_reg_loss(target, feature, self.stage1_reg)

        gradient = tape.gradient(loss, self.instrumental_feature.trainable_variables)
        self._instrumental_func_optimizer.apply(gradient, self.instrumental_feature.trainable_variables)

        return loss

    def update_value(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, reward_1st, discount_1st, next_obs_1st, _ = stage1_input
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, next_obs_2nd, _ = stage2_input
        next_action_1st = self.policy(next_obs_1st)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd)

        discount_1st = tf.expand_dims(discount_1st, axis=1) * self.discount
        with tf.GradientTape() as tape:
            target_1st = self.value_feature(obs=next_obs_1st, action=next_action_1st)
            stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
            predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
            current_feature = self.value_feature(obs=current_obs_2nd, action=action_2nd)
            predicted_feature = current_feature - discount_1st * predicted_feature
            aug_predicted_feature = tf.concat([predicted_feature, 1.0 - discount_1st], axis=1)
            loss = linear_reg_loss(tf.expand_dims(reward_2nd, -1), aug_predicted_feature, self.stage2_reg)

        gradient = tape.gradient(loss, self.value_feature.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_feature.trainable_variables)
        return loss

    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, reward_1st, discount_1st, next_obs_1st, _ = stage1_input
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, next_obs_2nd, _ = stage2_input
        next_action_1st = self.policy(next_obs_1st)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd)

        discount_1st = tf.expand_dims(discount_1st, axis=1) * self.discount
        target_1st = self.value_feature(obs=next_obs_1st, action=next_action_1st)
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        current_feature = self.value_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature = current_feature - discount_1st * predicted_feature
        aug_predicted_feature = tf.concat([predicted_feature, 1.0 - discount_1st], axis=1)
        self.value_func._weight = fit_linear(tf.expand_dims(reward_2nd, -1), aug_predicted_feature, self.stage2_reg)

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
