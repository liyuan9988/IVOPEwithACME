# Lint as: python3
"""DeepIV Learner implementation."""

from typing import Dict, List

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf

from src.utils.tf_linear_reg_utils import fit_linear, linear_reg_loss, linear_reg_pred


class DeepIVLearner(acme.Learner, tf2_savers.TFSaveable):
    """DeepIVLearner.

    This is the learning component of a DeepIV learner. IE it takes a dataset as
    input and implements update functionality to learn from this dataset.
    Optionally it takes a replay client as well to allow for updating of
    priorities.
    """

    def __init__(self,
                 value_func: snt.Module,
                 mixture_density: snt.Module,
                 policy_net: snt.Module,
                 discount: float,
                 value_learning_rate: float,
                 density_learning_rate: float,
                 n_sampling: int,
                 density_iter: int,
                 value_iter: int,
                 dataset: tf.data.Dataset,
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True):
        """Initializes the learner.

        Args:
          value_feature: value function network
          mixture_density: mixture density function network.
          policy_net: policy network.
          discount: global discount.
          value_learning_rate: learning rate for the treatment_net update.
          density_learning_rate: learning rate for the mixture_density update.
          n_sampling: number of samples generated in stage 2,
          density_iter: number of iteration for mixture_density function,
          value_iter: number of iteration for value function,
          dataset: dataset to learn from.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        self.density_iter = density_iter
        self.value_iter = value_iter
        self.n_sampling = n_sampling
        self.discount = discount

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self.value_func = value_func
        self.mixture_density = mixture_density
        self.policy = policy_net
        self._value_func_optimizer = snt.optimizers.Adam(value_learning_rate)
        self._mixture_density_optimizer = snt.optimizers.Adam(density_learning_rate)

        self._variables = [
            value_func.trainable_variables,
            mixture_density.trainable_variables,
        ]
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Create a snapshotter object.
        if checkpoint:
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={'value_func': value_func,
                                 'mixture_density': mixture_density,
                                 }, time_delta_minutes=60.)
        else:
            self._snapshotter = None

    def _step(self) -> Dict[str, tf.Tensor]:
        stage1_loss = None
        stage2_loss = None
        # Pull out the data needed for updates/priorities.
        for i in range(self.density_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t, _ = inputs.data
            stage1_loss = self.update_density(o_tm1, a_tm1, r_t, d_t, o_t)

        for i in range(self.value_iter):
            inputs = next(self._iterator)
            o_tm1, a_tm1, r_t, d_t, o_t, _ = inputs.data
            stage2_loss = self.update_value(o_tm1, a_tm1, r_t, d_t, o_t)

        # Compute the global norm of the gradients for logging.
        fetches = {'stage1_loss': stage1_loss, "stage2_loss": stage2_loss}

        return fetches

    def update_density(self, current_obs, action, reward, discount, next_obs):
        target = tf2_utils.batch_concat(next_obs)
        with tf.GradientTape() as tape:
            density = self.mixture_density(current_obs, action)
            loss = tf.reduce_mean(-density.log_prob(target))

        gradient = tape.gradient(loss, self.mixture_density.trainable_variables)
        self._mixture_density_optimizer.apply(gradient, self.mixture_density.trainable_variables)

        return loss

    def obtain_sampled_value_function(self, current_obs, action):
        res_list = []
        for i in range(self.n_sampling):
            sampled_value = self.mixture_density.obtain_sampled_value_function(current_obs, action, self.policy,
                                                                               self.value_func)
            res_list.append(sampled_value)
        return tf.reduce_mean(tf.concat(res_list, axis=0), axis=0)

    def update_value(self, current_obs, action, reward, discount, next_obs):
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            next_value = self.mixture_density.obtain_sampled_value_function(current_obs, action, self.policy,
                                                                            self.value_func)
            current_value = self.value_func(current_obs, action)
            pred = current_value - discount * next_value * self.discount
            loss = mse(y_pred=pred, y_true=reward)

        gradient = tape.gradient(loss, self.value_func.trainable_variables)
        self._value_func_optimizer.apply(gradient, self.value_func.trainable_variables)
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
            'value_function': self.value_func,
            'mixture_density': self.mixture_density,
            'value_opt': self._value_func_optimizer,
            'density_opt': self._mixture_density_optimizer,
            'num_steps': self._num_steps
        }
