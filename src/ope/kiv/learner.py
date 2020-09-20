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
                 dataset: tf.data.Dataset,
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
          dataset: dataset to learn from.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.discount = discount

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.policy = policy_net

        self.stage1_input = None
        self.stage2_input = None

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

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = None
        stage2_loss = None
        # Pull out the data needed for updates/priorities.
        if self.stage2_input is None:
            self.stage1_input = next(self._iterator)
            self.stage2_input = next(self._iterator)

        o_tm1, a_tm1, r_t, d_t, o_t, _ = self.stage1_input
        stage1_loss, stage2_loss = self.update_final_weight(self.stage1_input, self.stage2_input)
        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss, "stage2_loss": stage2_loss}

        return fetches


    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, reward_1st, discount_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd = stage2_input[:3]
        next_action_1st = self.policy(next_obs_1st)
        discount_1st = tf.expand_dims(discount_1st, axis=1)

        instrumental_feature_1st = self.instrumental_feature(obs=current_obs_1st, action=action_1st)
        instrumental_feature_2nd = self.instrumental_feature(obs=current_obs_2nd, action=action_2nd)

        target_1st = discount_1st * self.value_feature(obs=next_obs_1st, action=next_action_1st)
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
        stage1_loss = linear_reg_loss(target_1st, instrumental_feature_1st, self.stage1_reg)
        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        current_feature = self.value_feature(obs=current_obs_2nd, action=action_2nd)
        predicted_feature = current_feature - self.discount * predicted_feature
        self.value_func._weight.assign(
            fit_linear(tf.expand_dims(reward_2nd, -1), predicted_feature, self.stage2_reg))
        stage2_loss = linear_reg_loss(tf.expand_dims(reward_2nd, -1), predicted_feature, self.stage2_reg)
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