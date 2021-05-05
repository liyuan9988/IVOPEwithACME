# Lint as: python3
# pylint: disable=bad-indentation,line-too-long
"""DFIV Learner implementation."""

import datetime
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

# Default Acme checkpoint TTL is 5 days.
_CHECKPOINT_TTL = int(datetime.timedelta(days=30).total_seconds())


class DFIV2Learner(acme.Learner, tf2_savers.TFSaveable):
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
                 value_reg: float,
                 instrumental_reg: float,
                 stage1_reg: float,
                 stage2_reg: float,
                 instrumental_iter: int,
                 value_iter: int,
                 dataset: tf.data.Dataset,
                 d_tm1_weight: float = 1.0,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True,
                 checkpoint_interval_minutes: int = 10.0):
        """Initializes the learner.

        Args:
          value_func: value function network
          instrumental_feature: dual function network.
          policy_net: policy network.
          discount: global discount.
          value_learning_rate: learning rate for the treatment_net update.
          instrumental_learning_rate: learning rate for the instrumental_net update.
          value_reg: L2 regularizer for value net.
          instrumental_reg: L2 regularizer for instrumental net.
          stage1_reg: ridge regularizer for stage 1 regression
          stage2_reg: ridge regularizer for stage 2 regression
          instrumental_iter: number of iteration for instrumental net
          value_iter: number of iteration for value function,
          dataset: dataset to learn from.
          d_tm1_weight: weights for terminal state transitions. Ignored in this variant.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
          checkpoint_interval_minutes: checkpoint interval in minutes.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.instrumental_iter = instrumental_iter
        self.value_iter = value_iter
        self.discount = discount
        self.value_reg = value_reg
        self.instrumental_reg = instrumental_reg
        del d_tm1_weight

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.policy = policy_net
        self._value_func_optimizer = snt.optimizers.Adam(
            value_learning_rate, beta1=0.5, beta2=0.9)
        self._instrumental_func_optimizer = snt.optimizers.Adam(
            instrumental_learning_rate, beta1=0.5, beta2=0.9)

        # Define additional variables.
        self.stage1_weight = tf.Variable(
            tf.zeros((instrumental_feature.feature_dim(),
                      value_func.feature_dim()), dtype=tf.float32))
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        self._variables = [
            self.value_func.trainable_variables,
            self.instrumental_feature.trainable_variables,
            self.stage1_weight,
        ]

        # Create a checkpointer object.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                objects_to_save=self.state,
                time_delta_minutes=checkpoint_interval_minutes,
                checkpoint_ttl_seconds=_CHECKPOINT_TTL)
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={'value_func': self.value_func,
                                 'instrumental_feature': self.instrumental_feature,
                                 }, time_delta_minutes=60.)

    def update_batch(self):
        stage1_input = next(self._iterator)
        stage2_input = next(self._iterator)
        return stage1_input.data, stage2_input.data

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = None
        stage2_loss = None
        for _ in range(self.value_iter):
            # Pull out the data needed for updates/priorities.
            stage1_input, stage2_input = self.update_batch()
            for _ in range(self.instrumental_iter // self.value_iter):
                o_tm1, a_tm1, r_t, d_t, o_t = stage1_input[:5]
                stage1_loss = self.update_instrumental(o_tm1, a_tm1, r_t, d_t, o_t)

            stage2_loss = self.update_value(stage1_input, stage2_input)

        self.update_final_weight(stage1_input, stage2_input)
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss, 'stage2_loss': stage2_loss}
        return fetches

    def cal_validation_err(self, valid_input):
        """Return prediction MSE on the validation dataset."""
        stage1_weight = self.stage1_weight
        stage2_weight = self.value_func.weight
        se_sum = 0.
        se2_sum = 0.
        count = 0.
        for sample in valid_input:
            data = sample.data
            current_obs, action, reward = data[:3]
            instrumental_feature = self.instrumental_feature(obs=current_obs, action=action,
                                                             training=False)
            predicted_feature = linear_reg_pred(instrumental_feature, stage1_weight)
            predict = linear_reg_pred(predicted_feature, stage2_weight)

            sq_err = tf.square(tf.expand_dims(reward, -1) - predict)
            se_sum += tf.reduce_sum(sq_err)
            se2_sum += tf.reduce_sum(tf.square(sq_err))
            count += tf.cast(sq_err.shape[0], tf.float32)
        mse = se_sum / count
        mse_err_std = tf.sqrt((se2_sum / count - mse ** 2) / count)
        return mse, mse_err_std

    def update_instrumental(self, current_obs, action, reward, discount, next_obs):
        next_action = self.policy(next_obs)
        discount = tf.expand_dims(discount, axis=1)
        target = discount * add_const_col(self.value_feature(next_obs, next_action, training=False))
        target = add_const_col(self.value_feature(current_obs, action)) - self.discount * target
        l2 = snt.regularizers.L2(self.instrumental_reg)
        with tf.GradientTape() as tape:
            feature = self.instrumental_feature(obs=current_obs, action=action, training=True)
            loss = linear_reg_loss(target, feature, self.stage1_reg)
            loss = loss + l2(self.instrumental_feature.trainable_variables)
            loss /= action.shape[0]

        gradient = tape.gradient(loss, self.instrumental_feature.trainable_variables)
        self._instrumental_func_optimizer.apply(gradient, self.instrumental_feature.trainable_variables)

        return loss

    def update_value(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd = stage2_input[:3]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st,
                                                             training=False)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd,
                                                             training=False)
        l2 = snt.regularizers.L2(self.value_reg)
        with tf.GradientTape() as tape:
            target_1st = discount_1st * add_const_col(
                self.value_feature(obs=next_obs_1st, action=next_action_1st, training=True))
            target_1st = add_const_col(self.value_feature(obs=current_obs_1st, action=action_1st,
                                                          training=True)) - self.discount * target_1st
            stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
            predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
            loss = linear_reg_loss(tf.expand_dims(reward_2nd, -1), predicted_feature, self.stage2_reg)
            loss = loss + l2(self.value_feature.trainable_variables)
            loss /= action_2nd.shape[0]

        gradient = tape.gradient(loss, self.value_feature.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_feature.trainable_variables)
        return loss

    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd = stage2_input[:3]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st,
                                                             training=False)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd,
                                                             training=False)

        target_1st = discount_1st * add_const_col(
            self.value_feature(obs=next_obs_1st, action=next_action_1st, training=True))
        target_1st = add_const_col(self.value_feature(obs=current_obs_1st, action=action_1st,
                                                      training=True)) - self.discount * target_1st
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
        self.stage1_weight.assign(stage1_weight)
        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        stage2_weight = fit_linear(tf.expand_dims(reward_2nd, -1), predicted_feature, self.stage2_reg)
        self.value_func.weight.assign(stage2_weight)

        return stage1_weight, stage2_weight

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
          self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        if self._num_steps % 10 == 0:
            self._logger.write(result)

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tf2_utils.to_numpy(self._variables)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'value_func': self.value_func,
            'instrumental_feature': self.instrumental_feature,
            'stage1_weight': self.stage1_weight,
            'value_func_optimizer': self._value_func_optimizer,
            'instrumental_func_optimizer': self._instrumental_func_optimizer,
            'num_steps': self._num_steps,
            'counter': self._counter,
        }
