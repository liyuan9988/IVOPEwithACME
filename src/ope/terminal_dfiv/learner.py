# Lint as: python3
# pylint: disable=bad-indentation,line-too-long
"""DFIV Learner implementation."""

from typing import Dict, List

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

from src.utils.tf_linear_reg_utils import fit_linear, linear_reg_loss, linear_reg_pred, add_const_col


class TerminalDFIVLearner(acme.Learner, tf2_savers.TFSaveable):
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
                 terminate_predictor: snt.Module,
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
          d_tm1_weight: weights for terminal state transitions.
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
        self.value_reg = value_reg
        self.instrumental_reg = instrumental_reg
        self.d_tm1_weight = d_tm1_weight
        self.ignore_terminate_confounding = True

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self.stage1_input = next(self._iterator)
        self.stage2_input = next(self._iterator)
        if isinstance(self.stage1_input, reverb.ReplaySample):
            self.stage1_input = self.stage1_input.data
            self.stage2_input = self.stage2_input.data

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.terminate_predictor = terminate_predictor
        self.learn_terminate_predictor()

        self.policy = policy_net
        self._value_func_optimizer = snt.optimizers.Adam(
            value_learning_rate, beta1=0.5, beta2=0.9)
        self._instrumental_func_optimizer = snt.optimizers.Adam(
            instrumental_learning_rate, beta1=0.5, beta2=0.9)

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
                                 'terminate_predictor': terminate_predictor,
                                 }, time_delta_minutes=60.)
        else:
            self._snapshotter = None

    def learn_terminate_predictor(self):
        print("start training terminate predictor")
        opt = snt.optimizers.Adam(0.001, beta1=0.5, beta2=0.9)
        bce = tf.keras.losses.BinaryCrossentropy()
        for i in range(1000):
            with tf.GradientTape() as tape:
                o_tm1, a_tm1, _, d_t, _, _, _ = self.stage1_input[:7]
                pred = self.terminate_predictor(o_tm1, a_tm1, training=True)
                loss = bce(tf.expand_dims(d_t, axis=-1), pred)
                print(loss)
            gradient = tape.gradient(loss, self.terminate_predictor.trainable_variables)
            opt.apply(gradient, self.terminate_predictor.trainable_variables)
        print("end training terminate predictor")

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = None
        stage2_loss = None
        # Pull out the data needed for updates/priorities.

        for i in range(self.value_iter):
            for j in range(self.instrumental_iter // self.value_iter):
                o_tm1, a_tm1, r_t, d_t, o_t, _, d_tm1 = self.stage1_input[:7]
                stage1_loss = self.update_instrumental(o_tm1, a_tm1, r_t, d_t, o_t, d_tm1)

            stage2_loss = self.update_value(self.stage1_input, self.stage2_input)

        self.update_final_weight(self.stage1_input, self.stage2_input)
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss, 'stage2_loss': stage2_loss}

        return fetches

    def update_instrumental(self, current_obs, action, reward, discount, next_obs, d_tm1):
        next_action = self.policy(next_obs)
        discount = tf.expand_dims(discount, axis=1)
        target = discount * add_const_col(self.value_feature(next_obs, next_action, training=False))
        l2 = snt.regularizers.L2(self.instrumental_reg)
        with tf.GradientTape() as tape:
            feature = self.instrumental_feature(obs=current_obs, action=action, training=True)
            feature = discount * feature
            loss = linear_reg_loss(target, feature, self.stage1_reg)
            loss = loss + l2(self.instrumental_feature.trainable_variables)
            loss /= current_obs.shape[0]

        gradient = tape.gradient(loss, self.instrumental_feature.trainable_variables)
        self._instrumental_func_optimizer.apply(gradient, self.instrumental_feature.trainable_variables)

        return loss

    @tf.function
    def cal_value_weight(self, predicted_feature, current_feature, stage2_input):
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, _, _, _ = stage2_input[:7]
        discount_2nd = tf.expand_dims(discount_2nd, axis=1)
        feature = current_feature - discount_2nd * self.discount * predicted_feature
        nData, nDim = feature.shape
        nData = tf.cast(tf.shape(feature)[0], dtype=tf.float32)
        A = tf.matmul(feature, feature, transpose_a=True)
        A = A + self.stage2_reg * tf.eye(nDim) * nData
        b = tf.matmul(feature, tf.expand_dims(reward_2nd, -1), transpose_a=True)
        if not self.ignore_terminate_confounding:
            non_terminate_prob = self.terminate_predictor(current_obs_2nd, action_2nd)
            var_prob = non_terminate_prob * (1.0 - non_terminate_prob)
            A -= tf.matmul(var_prob * self.discount * predicted_feature,
                           self.discount * predicted_feature, transpose_a=True)

        weight = tf.linalg.solve(A, b)
        pred = tf.matmul(feature, weight)
        loss = tf.norm((tf.expand_dims(reward_2nd, -1) - pred)) ** 2 + self.stage2_reg * tf.norm(weight) ** 2 * nData
        if not self.ignore_terminate_confounding:
            future_q_val = tf.matmul(predicted_feature, weight) * self.discount
            loss -= tf.reduce_sum(future_q_val * future_q_val * var_prob)

        return weight, loss

    def update_value(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st, _, _ = stage1_input[:7]
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, _, _, _ = stage2_input[:7]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)
        discount_2nd = tf.expand_dims(discount_2nd, axis=1)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st,
                                                             training=False) * discount_1st
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd,
                                                             training=False) * discount_2nd
        l2 = snt.regularizers.L2(self.value_reg)
        with tf.GradientTape() as tape:
            # target_1st = discount_1st * self.value_feature(obs=next_obs_1st, action=next_action_1st, training=True)
            target_1st = discount_1st * add_const_col(
                self.value_feature(obs=next_obs_1st, action=next_action_1st, training=True))
            stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
            predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
            current_feature = add_const_col(self.value_feature(obs=current_obs_2nd, action=action_2nd, training=True))
            stage2_weight, loss = self.cal_value_weight(predicted_feature, current_feature, stage2_input)
            loss = loss + l2(self.value_feature.trainable_variables)
            loss /= current_obs_2nd.shape[0]

        gradient = tape.gradient(loss, self.value_feature.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_feature.trainable_variables)
        return loss

    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st, _, _ = stage1_input[:7]
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd, _, _, _ = stage2_input[:7]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)
        discount_2nd = tf.expand_dims(discount_2nd, axis=1)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st,
                                                             training=False) * discount_1st
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd,
                                                             training=False) * discount_2nd
        target_1st = discount_1st * add_const_col(
            self.value_feature(obs=next_obs_1st, action=next_action_1st, training=True))
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        current_feature = add_const_col(self.value_feature(obs=current_obs_2nd, action=action_2nd, training=True))
        stage2_weight, loss = self.cal_value_weight(predicted_feature, current_feature, stage2_input)
        self.value_func._weight.assign(stage2_weight)

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Snapshot and attempt to write logs.
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
            'value_feature': self.value_feature,
            'instrumental_feature': self.instrumental_feature,
            'value_opt': self._value_func_optimizer,
            'dual_opt': self._instrumental_func_optimizer,
            'num_steps': self._num_steps
        }
