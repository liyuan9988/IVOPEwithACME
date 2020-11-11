# pylint: disable=bad-indentation,line-too-long
from absl import logging
from acme import specs
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from .load_data import load_policy_net
import numpy as np
import reverb
import tensorflow as tf


def generate_train_data(task_name,
                        behavior_policy_param,
                        dataset_path,
                        environment,
                        dataset_size,
                        batch_size,
                        shuffle,
                        include_terminal=False,  # Include terminal absorbing state.
                        ignore_d_tm1=False  # Set d_tm1 as constant 1.0 if True.
                        ):
    environment_spec = specs.make_environment_spec(environment)
    with tf.device('CPU'):
        behavior_policy_net = load_policy_net(task_name=task_name,
                                              params=behavior_policy_param,
                                              environment_spec=environment_spec,
                                              dataset_path=dataset_path)

        logging.info('start generating transitions')
        dataset = _generate_data(
            behavior_policy_net, environment, dataset_size, batch_size, shuffle,
            include_terminal=include_terminal,
            ignore_d_tm1=ignore_d_tm1)
        logging.info('end generating transitions')
    return dataset


def _generate_data(policy_net, environment, n_samples, batch_size, shuffle,
                   include_terminal=False,  # Include terminal absorbing state.
                   ignore_d_tm1=False  # Set d_tm1 as constant 1.0 if True.
                   ):
    sample_count = 0
    actor = actors.FeedForwardActor(policy_network=policy_net)
    timestep = environment.reset()
    actor.observe_first(timestep)

    current_obs_list = []
    action_list = []
    next_obs_list = []
    reward_list = []
    discount_list = []
    nonterminal_list = []
    while sample_count < n_samples:
        current_obs = timestep.observation
        action = actor.select_action(current_obs)
        timestep = environment.step(action)
        actor.observe(action, next_timestep=timestep)
        next_obs = timestep.observation
        reward = timestep.reward
        discount = np.array(1.0, dtype=np.float32)
        if timestep.last() and not include_terminal:
            discount = np.array(0.0, dtype=np.float32)

        current_obs_list.append(tf2_utils.add_batch_dim(current_obs))
        action_list.append(tf2_utils.add_batch_dim(action))
        reward_list.append(tf2_utils.add_batch_dim(reward))
        discount_list.append(tf2_utils.add_batch_dim(discount))
        next_obs_list.append(tf2_utils.add_batch_dim(next_obs))
        nonterminal_list.append(tf2_utils.add_batch_dim(np.array(1.0, dtype=np.float32)))

        if timestep.last():
            if include_terminal:
                # Make another transition tuple from s, a -> s, a with 0 reward.
                current_obs = next_obs
                # action = actor.select_action(current_obs)
                reward = np.zeros_like(timestep.reward)
                discount = np.array(1.0, dtype=np.float32)
                next_obs = current_obs

                if ignore_d_tm1:
                    d_tm1 = np.array(1.0, dtype=np.float32)
                else:
                    d_tm1 = np.array(0.0, dtype=np.float32)

                for i in range(environment.action_spec().num_values):
                    action_ = np.array(i, dtype=action.dtype).reshape(action.shape)

                    current_obs_list.append(tf2_utils.add_batch_dim(current_obs))
                    action_list.append(tf2_utils.add_batch_dim(action_))
                    reward_list.append(tf2_utils.add_batch_dim(reward))
                    discount_list.append(tf2_utils.add_batch_dim(discount))
                    next_obs_list.append(tf2_utils.add_batch_dim(next_obs))
                    nonterminal_list.append(tf2_utils.add_batch_dim(d_tm1))

            timestep = environment.reset()
            actor.observe_first(timestep)

        sample_count += 1

    current_obs_data = tf.concat(current_obs_list, axis=0)
    action_data = tf.concat(action_list, axis=0)
    next_obs_data = tf.concat(next_obs_list, axis=0)
    reward_data = tf.concat(reward_list, axis=0)
    discount_data = tf.concat(discount_list, axis=0)
    nonterminal_data = tf.concat(nonterminal_list, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((current_obs_data,
                                                  action_data,
                                                  reward_data,
                                                  discount_data,
                                                  next_obs_data,
                                                  # The last action is not valid
                                                  # and should not be used.
                                                  action_data,
                                                  nonterminal_data))

    def _reverb_sample(*data_tuple):
        info = reverb.SampleInfo(key=tf.constant(0, tf.uint64),
                                 probability=tf.constant(1.0, tf.float64),
                                 table_size=tf.constant(0, tf.int64),
                                 priority=tf.constant(1.0, tf.float64))
        return reverb.ReplaySample(info=info, data=data_tuple)
    dataset = dataset.map(_reverb_sample,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
