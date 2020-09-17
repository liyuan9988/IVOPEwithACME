from acme.tf import utils as tf2_utils
from acme.agents.tf import actors
import tensorflow as tf

def generate_train_data(policy_net, environment, n_samples):
    sample_count = 0
    actor = actors.FeedForwardActor(policy_network=policy_net)
    timestep = environment.reset()
    actor.observe_first(timestep)

    current_obs_list = []
    action_list = []
    next_obs_list = []
    reward_list = []
    discount_list = []
    while sample_count < n_samples:
        current_obs = timestep.observation
        action = actor.select_action(current_obs)
        timestep = environment.step(action)
        actor.observe(action, next_timestep=timestep)
        next_obs = timestep.observation
        reward = timestep.reward
        discount = 1.0
        if timestep.last():
            discount = 0.0

        current_obs_list.append(tf2_utils.add_batch_dim(current_obs))
        action_list.append(tf2_utils.add_batch_dim(action))
        reward_list.append(tf2_utils.add_batch_dim(reward))
        discount_list.append(tf2_utils.add_batch_dim(discount))
        next_obs_list.append(tf2_utils.add_batch_dim(next_obs))

        if timestep.last():
            timestep = environment.reset()
            actor.observe_first(timestep)

        sample_count += 1

    current_obs_data = tf.concat(current_obs_list, axis=0)
    action_data = tf.concat(action_list, axis=0)
    next_obs_data = tf.concat(next_obs_list, axis=0)
    reward_data = tf.concat(reward_list, axis=0)
    discount_data = tf.concat(discount_list, axis=0)

    return tf.data.Dataset.from_tensor_slices((current_obs_data,
                                               action_data,
                                               reward_data,
                                               discount_data,
                                               next_obs_data,
                                               action_data))
