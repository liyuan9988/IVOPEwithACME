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
                 ignore_terminate_confounding: bool,
                 dataset: tf.data.Dataset,
                 d_tm1_weight: float = 1.0,
                 n_terminate_predictor_iter: int = 1000,
                 terminate_predictor_learning_rate: float = 0.001,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True,
                 checkpoint_interval_minutes: int = 10.0):
        """Initializes the learner.

        Args:
          value_func: value function network
          instrumental_feature: dual function network.
          policy_net: policy network.
          terminate_predictor: terminate predictor network.
          discount: global discount.
          value_learning_rate: learning rate for the treatment_net update.
          instrumental_learning_rate: learning rate for the instrumental_net update.
          value_reg: L2 regularizer for value net.
          instrumental_reg: L2 regularizer for instrumental net.
          stage1_reg: ridge regularizer for stage 1 regression
          stage2_reg: ridge regularizer for stage 2 regression
          instrumental_iter: number of iteration for instrumental net
          value_iter: number of iteration for value function,
          ignore_terminate_confounding: whether to ignore the confounding term caused by terminal state,
          dataset: dataset to learn from.
          d_tm1_weight: weights for terminal state transitions.
          n_terminate_predictor_iter: number of iteration for terminate predictor,
          terminate_predictor_learning_rate: learning rate for terminate predictor,
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
        self.d_tm1_weight = d_tm1_weight
        self.ignore_terminate_confounding = ignore_terminate_confounding

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.terminate_predictor = terminate_predictor
        self._bce = tf.keras.losses.BinaryCrossentropy()

        # self.learn_terminate_predictor(terminate_predictor_learning_rate,
        #                                n_terminate_predictor_iter)

        self.policy = policy_net
        self._value_func_optimizer = snt.optimizers.Adam(
            value_learning_rate, beta1=0.5, beta2=0.9)
        self._instrumental_func_optimizer = snt.optimizers.Adam(
            instrumental_learning_rate, beta1=0.5, beta2=0.9)

        self._terminate_predictor_optimizer = snt.optimizers.Adam(
            terminate_predictor_learning_rate)
        self._n_terminate_predictor_iter = n_terminate_predictor_iter

        # Define additional variables.
        self.stage1_weight = tf.Variable(
            tf.zeros((instrumental_feature.feature_dim(),
                      value_func.feature_dim()), dtype=tf.float32))
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        self._variables = [
            self.value_func.trainable_variables,
            self.instrumental_feature.trainable_variables,
            self.terminate_predictor.trainable_variables,
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
                                 'terminate_predictor': self.terminate_predictor,
                                 }, time_delta_minutes=60.)

    def update_batch(self):
        stage1_input = next(self._iterator)
        stage2_input = next(self._iterator)
        return stage1_input.data, stage2_input.data

    # def learn_terminate_predictor(self, stage1_input, lr, niter):
    #     print('start training terminate predictor')
    #     opt = snt.optimizers.Adam(lr, beta1=0.5, beta2=0.9)
    #     bce = tf.keras.losses.BinaryCrossentropy()
    #     for _ in range(niter):
    #         with tf.GradientTape() as tape:
    #             o_tm1, a_tm1, _, d_t = stage1_input[:4]
    #             pred = self.terminate_predictor(o_tm1, a_tm1, training=True)
    #             loss = bce(tf.expand_dims(d_t, axis=-1), pred)
    #             print(loss)
    #         gradient = tape.gradient(loss, self.terminate_predictor.trainable_variables)
    #         opt.apply(gradient, self.terminate_predictor.trainable_variables)
    #     print('end training terminate predictor')

    @tf.function
    def learn_terminate_predictor_step(self, stage1_input):
        print('start training terminate predictor')
        with tf.GradientTape() as tape:
            o_tm1, a_tm1, _, d_t = stage1_input[:4]
            pred = self.terminate_predictor(o_tm1, a_tm1, training=True)
            loss = self._bce(tf.expand_dims(d_t, axis=-1), pred)
            print(loss)
        gradient = tape.gradient(loss, self.terminate_predictor.trainable_variables)
        self._terminate_predictor_optimizer.apply(
            gradient, self.terminate_predictor.trainable_variables)
        print('end training terminate predictor')
        return loss

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = 0.
        stage2_loss = 0.
        pred_loss = 0.
        val_loss = 0.

        if self._num_steps < self._n_terminate_predictor_iter:
            stage1_input, _ = self.update_batch()
            pred_loss = self.learn_terminate_predictor_step(stage1_input)
        else:
            for _ in range(self.value_iter):
                stage1_input, stage2_input = self.update_batch()
                for _ in range(self.instrumental_iter // self.value_iter):
                    o_tm1, a_tm1, r_t, d_t, o_t, _, d_tm1 = stage1_input[:7]
                    stage1_loss = self.update_instrumental(o_tm1, a_tm1, r_t, d_t, o_t, d_tm1)
                stage2_loss = self.update_value(stage1_input, stage2_input)

            self.update_final_weight(stage1_input, stage2_input)
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss,
                   'stage2_loss': stage2_loss,
                   'pred_loss': pred_loss,
                   'val_loss': val_loss
                   }

        return fetches

    def cal_validation_err(self, valid_input):
        """Return prediction MSE on the validation dataset."""
        stage1_weight = self.stage1_weight
        stage2_weight = self.value_func.weight
        loss_sum = 0.
        count = 0.
        for sample in valid_input:
            data = sample.data
            current_obs, action, reward, discount = data[:4]
            discount = tf.expand_dims(discount, axis=1)
            instrumental_feature = self.instrumental_feature(obs=current_obs, action=action,
                                                             training=False) * discount
            predicted_feature = linear_reg_pred(instrumental_feature, stage1_weight)
            current_feature = add_const_col(self.value_feature(obs=current_obs, action=action, training=False))
            non_terminate_prob = self.terminate_predictor(current_obs, action)
            var_prob = non_terminate_prob * (1.0 - non_terminate_prob)

            feature = current_feature - discount * self.discount * predicted_feature
            pred = tf.matmul(feature, stage2_weight)
            loss = tf.norm((tf.expand_dims(reward, -1) - pred)) ** 2
            if not self.ignore_terminate_confounding:
                future_q = tf.matmul(predicted_feature, stage2_weight) * self.discount
                loss -= tf.reduce_sum(future_q * future_q * var_prob)
            loss_sum += loss / action.shape[0]
            count += 1.
        return loss_sum / count

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
            loss /= action.shape[0]

        gradient = tape.gradient(loss, self.instrumental_feature.trainable_variables)
        self._instrumental_func_optimizer.apply(gradient, self.instrumental_feature.trainable_variables)

        return loss

    @tf.function
    def cal_value_weight(self, predicted_feature, current_feature, stage2_input):
        current_obs_2nd, action_2nd, reward_2nd, discount_2nd = stage2_input.data[:4]
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
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, _, discount_2nd = stage2_input[:4]
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
            _, loss = self.cal_value_weight(predicted_feature, current_feature, stage2_input)
            loss = loss + l2(self.value_feature.trainable_variables)
            loss /= action_2nd.shape[0]

        gradient = tape.gradient(loss, self.value_feature.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_feature.trainable_variables)
        return loss

    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, _, discount_2nd = stage2_input[:4]
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
        self.stage1_weight.assign(stage1_weight)
        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        current_feature = add_const_col(self.value_feature(obs=current_obs_2nd, action=action_2nd, training=True))
        stage2_weight, _ = self.cal_value_weight(predicted_feature, current_feature, stage2_input)
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
            'terminate_predictor': self.terminate_predictor,
            'stage1_weight': self.stage1_weight,
            'value_func_optimizer': self._value_func_optimizer,
            'instrumental_func_optimizer': self._instrumental_func_optimizer,
            'terminate_predictor_optimizer': self._terminate_predictor_optimizer,
            'num_steps': self._num_steps
        }
