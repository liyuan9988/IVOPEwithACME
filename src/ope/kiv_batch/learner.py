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
import sonnet as snt
import tensorflow as tf

from src.utils.tf_linear_reg_utils import fit_linear, linear_reg_loss, linear_reg_pred, add_const_col


class KIVLearner(acme.Learner, tf2_savers.TFSaveable):
    """KIVLearner.

    This is the learning component of aKIV learner. IE it takes a dataset as
    input and implements update functionality to learn from this dataset.
    Optionally it takes a replay client as well to allow for updating of
    priorities.
    """

    def __init__(self,
                 value_func: snt.Module,
                 instrumental_feature: snt.Module,
                 policy_net: snt.Module,
                 discount: float,
                 stage1_reg: float,
                 stage2_reg: float,
                 stage1_batch: int,
                 stage2_batch: int,
                 dataset: tf.data.Dataset,
                 valid_dataset: tf.data.Dataset,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True):
        """Initializes the learner.

        Args:
          value_func: value function network
          instrumental_feature: dual function network.
          policy_net: policy network.
          discount: global discount.
          stage1_reg: ridge regularizer for stage 1 regression
          stage2_reg: ridge regularizer for stage 2 regression
          stage1_batch: number of mini-batches for stage 1 regression
          stage2_batch: number of mini-batches for stage 2 regression
          dataset: dataset to learn from.
          valid_dataset: validation dataset to compute score.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.discount = discount

        self.stage1_weight = None
        self.stage2_weight = None

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._valid_dataset = valid_dataset

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.policy = policy_net

        self.stage1_batch = stage1_batch
        self.stage2_batch = stage2_batch

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

    # @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss, stage2_loss = self.update_final_weight()
        self._num_steps.assign_add(1)

        fetches = {'stage1_loss': stage1_loss, 'stage2_loss': stage2_loss}
        return fetches

    def cal_stage1_weights(self, stage1_input):
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input.data[:5]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)
        target_1st = discount_1st * self.value_feature(obs=next_obs_1st, action=next_action_1st)
        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st)

        nData, nDim = instrumental_feature_1st.shape
        nData = tf.cast(tf.shape(instrumental_feature_1st)[0], dtype=tf.float32)
        A = tf.matmul(instrumental_feature_1st, instrumental_feature_1st, transpose_a=True)
        A = A + self.stage1_reg * tf.eye(nDim) * nData
        b = tf.matmul(instrumental_feature_1st, target_1st, transpose_a=True)
        return A / nData, b / nData

    def cal_stage1_loss(self):
        loss_sum = 0.
        count = 0.
        for sample in self._valid_dataset:
            loss_sum += self.cal_stage1_loss_one_batch(sample)
            count += 1.
        return loss_sum / count

    def cal_stage1_loss_one_batch(self, stage1_input):
        assert self.stage1_weight is not None
        current_obs_1st, action_1st, _, discount_1st, next_obs_1st = stage1_input.data[:5]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)
        target_1st = discount_1st * self.value_feature(obs=next_obs_1st, action=next_action_1st)
        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st)
        pred = linear_reg_pred(instrumental_feature_1st, self.stage1_weight)
        return tf.reduce_mean((pred - target_1st) ** 2).numpy()

    def cal_stage2_weight(self, stage2_input, stage1_weight):
        current_obs_2nd, action_2nd, reward_2nd, _, _ = stage2_input.data[:5]
        reward_2nd = tf.expand_dims(reward_2nd, axis=1)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature_2nd = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        current_feature_2nd = self.value_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature_2nd = current_feature_2nd - self.discount * predicted_feature_2nd

        nData, nDim = predicted_feature_2nd.shape
        nData = tf.cast(tf.shape(predicted_feature_2nd)[0], dtype=tf.float32)
        A = tf.matmul(predicted_feature_2nd, predicted_feature_2nd, transpose_a=True)
        A = A + self.stage2_reg * tf.eye(nDim) * nData
        b = tf.matmul(predicted_feature_2nd, reward_2nd, transpose_a=True)
        return A / nData, b / nData

    def cal_stage2_loss(self):
        loss_sum = 0.
        count = 0.
        for sample in self._valid_dataset:
            loss_sum += self.cal_stage2_loss_one_batch(sample)
            count += 1.
        return loss_sum / count

    def cal_stage2_loss_one_batch(self, sample):
        assert self.stage1_weight is not None
        assert self.stage2_weight is not None

        current_obs_2nd, action_2nd, reward_2nd, _, _ = sample.data[:5]
        reward_2nd = tf.expand_dims(reward_2nd, axis=1)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature_2nd = linear_reg_pred(instrumental_feature_2nd, self.stage1_weight)
        current_feature_2nd = self.value_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature_2nd = current_feature_2nd - self.discount * predicted_feature_2nd

        pred = linear_reg_pred(predicted_feature_2nd, self.stage2_weight)
        return tf.reduce_mean((pred - reward_2nd) ** 2).numpy()

    def update_final_weight(self):
        # calculate stage1 weights
        instrumental_feature_dim = self.instrumental_feature.feature_dim()
        value_feature_dim = self.value_feature.feature_dim()
        A = tf.zeros((instrumental_feature_dim, instrumental_feature_dim))
        b = tf.zeros((instrumental_feature_dim, value_feature_dim))
        for _ in range(self.stage1_batch):
            data = next(self._iterator)
            A_new, b_new = self.cal_stage1_weights(data)
            A = A + A_new
            b = b + b_new

        self.stage1_weight = tf.linalg.solve(A, b)
        # calculate training loss for the last batch
        # it may be replaced to validation data
        stage1_loss = None
        if self._valid_dataset is not None:
            stage1_loss = self.cal_stage1_loss()

        # calculate stage2 weights
        A = tf.zeros((value_feature_dim, value_feature_dim))
        b = tf.zeros((value_feature_dim, 1))
        for _ in range(self.stage2_batch):
            data = next(self._iterator)
            A_new, b_new = self.cal_stage2_weight(data, self.stage1_weight)
            A = A + A_new
            b = b + b_new

        self.stage2_weight = tf.linalg.solve(A, b)
        # calculate training loss for the last batch
        # it may be replaced to validation data
        stage2_loss = None
        if self._valid_dataset is not None:
            stage2_loss = self.cal_stage2_loss()

        self.value_func.weight.assign(self.stage2_weight)
        return stage1_loss, stage2_loss

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

        return result

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tf2_utils.to_numpy(self._variables)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'value_feature': self.value_feature,
            'instrumental_feature': self.instrumental_feature,
            'num_steps': self._num_steps
        }
