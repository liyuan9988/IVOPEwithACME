"""Functions for shared evaluation among all methods."""

from acme.tf import utils as tf2_utils

import numpy as np
import tensorflow as tf


def ope_evaluation(
    test_data, value_func, policy_net, environment, logger, num_init_samples):
  """Run OPE evaluation."""

  # Compute Bellman residual error from test data.
  current_obs, action, reward, discount, next_obs, _ = test_data.data
  next_action = policy_net(next_obs)
  target = (tf.expand_dims(reward, axis=1) +
            tf.expand_dims(discount, axis=1) *
            value_func(next_obs, next_action))
  mse = tf.reduce_mean((target - value_func(current_obs, action)) ** 2)

  # Compute policy value from initial distribution.
  q_0s = []
  for _ in range(num_init_samples):
    timestep = environment.reset()
    observation = tf2_utils.add_batch_dim(timestep.observation)
    action = policy_net(observation)
    q_0s.append(value_func(observation, action).numpy().squeeze())

  results = {
      'Bellman_Residual_MSE': mse,
      'Q0_mean': np.mean(q_0s),
      'Q0_std_err': np.std(q_0s, ddof=0) / np.sqrt(len(q_0s)),
  }
  logger.write(results)
