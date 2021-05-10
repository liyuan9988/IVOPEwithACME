# pylint: disable=bad-indentation,line-too-long,missing-function-docstring
"""
When near_policy_dataset = True, the behavior and target policy is trained in an
environment with noise_level = 0, and run_id = 1.
Otherwise, the target policy is trained in an environment with the same
noise_level as that to be evaluated.
"""

from absl import logging
import functools
import os
from typing import Any, Dict
from pathlib import Path
from acme import specs
from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import trfl
import sonnet as snt

from src.utils import load_offline_bsuite_dataset
from src.utils import load_offline_dm_control_dataset
from src.utils import acme_utils


TASK_SHARD_MAP = {
    "bsuite_catch": 1,
    "bsuite_mountain_car": 1,
    "bsuite_cartpole": 1,
    "dm_control_cartpole_swingup": 1,
    "dm_control_cheetah_run": 1,
    "dm_control_walker_walk": 1,
    "dm_control_humanoid_run": 10,
}


TASK_VALID_SHARD_MAP = {
    "bsuite_catch": 1,
    "bsuite_mountain_car": 1,
    "bsuite_cartpole": 1,
    "dm_control_cartpole_swingup": 1,
    "dm_control_cheetah_run": 1,
    "dm_control_walker_walk": 1,
    "dm_control_humanoid_run": 10,
}


def bsuite_near_policy_dataset_dir(bsuite_id: str,
                                   noise_level: float,
                                   dataset_path: str):
    # Policy training config.
    policy_train_env_noise_level = 0.0
    run_id = 1

    # Data generation config.
    # Policy action noise.
    policy_noise_level = 0.3
    # Environment noise level.
    prob_noise_level = noise_level

    dataset_path = Path(dataset_path)
    path = str(dataset_path.joinpath(
        f"bsuite_near_policy/transitions/{bsuite_id}/"
        f"policy_train_env_noise_{policy_train_env_noise_level}_run_{run_id}/"
        f"policy_noise_{policy_noise_level}/"
        f"env_noise_{prob_noise_level}")) + "/"
    return path


def bsuite_offline_dataset_dir(bsuite_id: str,
                               noise_level: float,
                               dataset_path: str):
    run_id = 0
    dataset_path = Path(dataset_path)
    path = str(dataset_path.joinpath(
        f"bsuite/{bsuite_id}_{noise_level}/{run_id}_full"))
    return path


def dm_control_offline_dataset_dir(dm_control_task_name: str,
                                   noise_level: float,
                                   dataset_path: str):
    run_id = 0
    dataset_path = Path(dataset_path)
    root_path = str(dataset_path.joinpath(
        "dm_control_suite_stochastic/transitions/"
        f"{dm_control_task_name}_{noise_level}/"))
    data_path = f"{run_id}_full"
    return root_path, data_path


def load_data_and_env(task_name: str,
                      noise_level: float,
                      dataset_path: str,
                      batch_size: int,
                      near_policy_dataset: bool = False,
                      valid_batch_size: int = 1024,
                      shuffle: bool = True,  # Shuffle training dataset.
                      repeat: bool = True,  # Repeat training dataset.
                      max_dev_size: int = None):
    """Load train/valid dataset and environment."""
    num_shards = TASK_SHARD_MAP.get(task_name, 1)
    num_valid_shards = TASK_VALID_SHARD_MAP.get(task_name, 1)
    if task_name.startswith("bsuite"):
        # BSuite tasks.
        bsuite_id = task_name[len("bsuite_"):] + "/0"
        if near_policy_dataset:
            # Near-policy dataset.
            path = bsuite_near_policy_dataset_dir(
                bsuite_id, noise_level, dataset_path)
        else:
            # Pure offline dataset.
            path = bsuite_offline_dataset_dir(
                bsuite_id, noise_level, dataset_path)
        logging.info("Dataset path: %s", path)
        train_dataset, valid_dataset, environment = load_offline_bsuite_dataset(
            bsuite_id=bsuite_id,
            random_prob=noise_level,
            path=path,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            num_shards=num_shards,
            num_valid_shards=num_valid_shards,
            shuffle=shuffle,
            repeat=repeat)
    elif task_name.startswith("dm_control"):
        # DM Control tasks.
        if near_policy_dataset:
            raise ValueError(
                "Near-policy dataset is not available for dm_control tasks.")
        dm_control_task_name = task_name[len("dm_control_"):]
        root_path, data_path = dm_control_offline_dataset_dir(
            dm_control_task_name, noise_level, dataset_path)
        logging.info("Dataset root path: %s", root_path)
        logging.info("Dataset file path: %s", data_path)
        train_dataset, valid_dataset, environment = load_offline_dm_control_dataset(
            task_name=dm_control_task_name,
            noise_std=noise_level,
            root_path=root_path,
            data_path=data_path,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            num_shards=num_shards,
            num_valid_shards=num_valid_shards,
            shuffle=shuffle,
            repeat=repeat)
    else:
        raise ValueError(f"task name {task_name} is unsupported.")
    if max_dev_size is not None:
        valid_dataset = valid_dataset.take(
            (max_dev_size + valid_batch_size - 1) // valid_batch_size)
    return train_dataset, valid_dataset, environment


def bsuite_policy_path(bsuite_id: str,
                       noise_level: float,
                       near_policy_dataset: bool,
                       dataset_path: str):

    if near_policy_dataset:
        env_noise_level = 0.0  # params["env_noise_level"]
        run_id = 1  # params["run_id"]
    else:
        env_noise_level = noise_level  # params["env_noise_level"]
        run_id = 1  # params["run_id"]
    # policy_noise_level = 0.1  # params["policy_noise_level"]
    dataset_path = Path(dataset_path)
    path = str(dataset_path.joinpath(
        "bsuite_near_policy/snapshots/"
        f"{bsuite_id}_{env_noise_level}/{run_id}_full"))
    return path


def dm_control_policy_path(dm_control_task: str,
                           noise_level: float,
                           dataset_path: str):
    env_noise_level = noise_level
    run_id = 1
    dataset_path = Path(dataset_path)
    path = str(dataset_path.joinpath(
        "dm_control_suite_stochastic/snapshots/"
        f"{dm_control_task}_{env_noise_level}/{run_id}_full"))
    return path


def load_policy_net(
    task_name: str,
    noise_level: float,
    dataset_path: str,
    environment_spec: specs.EnvironmentSpec,
    near_policy_dataset: bool = False,
    ):
    dataset_path = Path(dataset_path)
    if task_name.startswith("bsuite"):
        # BSuite tasks.
        bsuite_id = task_name[len("bsuite_"):] + "/0"
        path = bsuite_policy_path(
            bsuite_id, noise_level, near_policy_dataset, dataset_path)
        logging.info("Policy path: %s", path)
        policy_net = tf.saved_model.load(path)

        policy_noise_level = 0.1  # params["policy_noise_level"]
        observation_network = tf2_utils.to_sonnet_module(functools.partial(
            tf.reshape, shape=(-1,) + environment_spec.observations.shape))
        policy_net = snt.Sequential([
            observation_network,
            policy_net,
            # Uncomment this line to add action noise to the target policy.
            lambda q: trfl.epsilon_greedy(q, epsilon=policy_noise_level).sample(),
        ])
    elif task_name.startswith("dm_control"):
        # DM Control tasks.
        if near_policy_dataset:
            raise ValueError(
                "Near-policy dataset is not available for dm_control tasks.")
        dm_control_task = task_name[len("dm_control_"):]
        path = dm_control_policy_path(
            dm_control_task, noise_level, dataset_path)
        logging.info("Policy path: %s", path)
        policy_net = tf.saved_model.load(path)

        policy_noise_level = 0.2  # params["policy_noise_level"]
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)
        policy_net = snt.Sequential([
            observation_network,
            policy_net,
            # Uncomment these two lines to add action noise to target policy.
            acme_utils.GaussianNoise(policy_noise_level),
            networks.ClipToSpec(environment_spec.actions),
        ])
    else:
        raise ValueError(f"task name {task_name} is unsupported.")
    return policy_net
