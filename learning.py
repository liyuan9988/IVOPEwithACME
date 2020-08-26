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


class DFIVLearner(acme.Learner, tf2_savers.TFSaveable):
  """DFIV learner.

  This is the learning component of a DFIV agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(self,
               treatment_net: snt.Module,
               instrumental_net: snt.Module,
               policy_net: snt.Module,
               treatment_learning_rate: float,
               instrumental_learning_rate: float,
               policy_learning_rate: float,
               dataset: tf.data.Dataset,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True):
    """Initializes the learner.

    Args:
      treatment_net: treatment network.
      instrumental_net: instrumental network.
      policy_net: policy network.
      treatment_learning_rate: learning rate for the treatment_net update.
      instrumental_learning_rate: learning rate for the instrumental_net update.
      policy_learning_rate: learning rate for the policy_net update.
      dataset: dataset to learn from.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Get an iterator over the dataset.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    # TODO(b/155086959): Fix type stubs and remove.

    self._treatment_net = treatment_net
    self._instrumental_net = instrumental_net
    self._policy_net = policy_net
    self._treatment_optimizer = snt.optimizers.Adam(treatment_learning_rate)
    self._instrumental_optimizer = snt.optimizers.Adam(
        instrumental_learning_rate)
    self._policy_optimizer = snt.optimizers.Adam(policy_learning_rate)

    self._variables = [
        treatment_net.trainable_variables,
        instrumental_net.trainable_variables,
        policy_net.trainable_variables,
    ]
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Create a snapshotter object.
    if checkpoint:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'treatment_net': treatment_net,
                           'instrumental_net': instrumental_net,
                           'policy_net': policy_net,
                           }, time_delta_minutes=60.)
    else:
      self._snapshotter = None

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""
    # TODO(liyuan): add the learning algorithm in this method.

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    # del r_t, d_t, o_t
    #
    # with tf.GradientTape() as tape:
    #   # Evaluate our networks.
    #   logits = self._network(o_tm1)
    #   cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #   loss = cce(a_tm1, logits)
    #
    # gradients = tape.gradient(loss, self._network.trainable_variables)
    # self._optimizer.apply(gradients, self._network.trainable_variables)

    self._num_steps.assign_add(1)

    # # Compute the global norm of the gradients for logging.
    # global_gradient_norm = tf.linalg.global_norm(gradients)
    # fetches = {'loss': loss, 'gradient_norm': global_gradient_norm}

    return fetches

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
        'treatment_net': self._treatment_net,
        'instrumental_net': self._instrumental_net,
        'policy_net': self._policy_net,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }
