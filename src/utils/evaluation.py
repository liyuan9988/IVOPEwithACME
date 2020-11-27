# pylint: disable=bad-indentation,line-too-long
"""Functions for shared evaluation among all methods."""

from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
import numpy as np
import tensorflow as tf
import tree


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
            print(value_func(current_obs, action).numpy().squeeze())
            print(f'reward = {reward}')
            print('=====End Episode=====')

        else:
            next_action = tf2_utils.add_batch_dim(actor.select_action(next_obs))
            action = tf2_utils.add_batch_dim(action)
            current_obs = tf2_utils.add_batch_dim(current_obs)
            next_obs = tf2_utils.add_batch_dim(next_obs)
            mse_one = (reward + discount * value_func(next_obs, next_action) - value_func(current_obs, action)) ** 2
            print(value_func(current_obs, action).numpy().squeeze())
        mse = mse + mse_one.numpy()
        sample_count += 1
    return mse.squeeze() / mse_samples


def ope_evaluation(value_func, policy_net, environment, num_init_samples,
                   mse_samples=0, discount=0.99, counter=None, logger=None):
    """Run OPE evaluation."""
    mse = -1
    if mse_samples > 0:
        mse = cal_mse(value_func, policy_net, environment, mse_samples, discount)

    # Compute policy value from initial distribution.
    # q_0s = []
    # for _ in range(num_init_samples):
    #     timestep = environment.reset()
    #     observation = tf2_utils.add_batch_dim(timestep.observation)
    #     action = policy_net(observation)
    #     q_0s.append(value_func(observation, action).numpy().squeeze())

    init_obs = []
    for _ in range(num_init_samples):
        timestep = environment.reset()
        init_obs.append(timestep.observation)
    init_obs = tf2_utils.stack_sequence_fields(init_obs)
    init_obs = tree.map_structure(tf.convert_to_tensor, init_obs)
    init_actions = policy_net(init_obs)
    q_0s = value_func(init_obs, init_actions).numpy().squeeze()

    results = {
        'Bellman_Residual_MSE': mse,
        'Q0_mean': np.mean(q_0s),
        'Q0_std_err': np.std(q_0s, ddof=0) / np.sqrt(len(q_0s)),
    }
    if counter is not None:
        counts = counter.increment(steps=1)
        results.update(counts)
    if logger is not None:
        logger.write(results)
    return results
