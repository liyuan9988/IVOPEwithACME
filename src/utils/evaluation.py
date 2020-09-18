"""Functions for shared evaluation among all methods."""

from acme.tf import utils as tf2_utils
from acme.agents.tf import actors

import numpy as np
import tensorflow as tf


def cal_mse(value_func, policy_net, environment, mse_samples, discount):
    sample_count = 0
    actor = actors.FeedForwardActor(policy_network=policy_net)
    timestep = environment.reset()
    actor.observe_first(timestep)
    mse = 0.0
    while sample_count < mse_samples:
        current_obs = timestep.observation
        action = actor.select_action(current_obs)
        timestep = environment.step(action)
        actor.observe(action, next_timestep=timestep)
        next_obs = timestep.observation
        reward = timestep.reward

        if timestep.last():
            timestep = environment.reset()
            actor.observe_first(timestep)
            current_obs = tf2_utils.add_batch_dim(current_obs)
            action = tf2_utils.add_batch_dim(action)
            mse_one = (reward - value_func(current_obs, action)) ** 2
            print(value_func(current_obs, action))
            print(f"reward = {reward}")
            print("=====End Episode=====")

        else:
            next_action = tf2_utils.add_batch_dim(actor.select_action(next_obs))
            action = tf2_utils.add_batch_dim(action)
            current_obs = tf2_utils.add_batch_dim(current_obs)
            next_obs = tf2_utils.add_batch_dim(next_obs)
            mse_one = (reward + discount * value_func(next_obs, next_action) - value_func(current_obs, action)) ** 2
            print(value_func(current_obs, action))
        mse = mse + mse_one.numpy()
        sample_count += 1
    return mse / mse_samples


def ope_evaluation(value_func, policy_net, environment, logger, num_init_samples, mse_samples=0, discount=0.99):
    """Run OPE evaluation."""
    mse = -1
    if mse_samples > 0:
        mse = cal_mse(value_func, policy_net, environment, mse_samples, discount)

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
