# pylint: disable=bad-indentation,line-too-long,missing-function-docstring

import functools
from typing import Any, Dict
from pathlib import Path
from acme import specs
from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import trfl
import sonnet as snt

from src.utils import load_offline_dm_control_dataset, load_offline_bsuite_dataset
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


def load_data_and_env(task_name: str,
                      params: Dict[str, Any],
                      dataset_path: str,
                      batch_size: int,
                      valid_batch_size: int = 1024,
                      shuffle: bool = True,  # Shuffle training dataset.
                      repeat: bool = True,  # Repeat training dataset.
                      max_dev_size: int = None):
    """Load train/valid dataset and environment."""
    dataset_path = Path(dataset_path)
    num_shards = TASK_SHARD_MAP.get(task_name, 1)
    num_valid_shards = TASK_VALID_SHARD_MAP.get(task_name, 1)
    if task_name.startswith("bsuite"):
        # BSuite tasks.
        bsuite_id = task_name[len("bsuite_"):] + "/0"
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = dataset_path.joinpath(f"bsuite/transitions/{bsuite_id}_{noise_level}/{run_id}_full")
        train_dataset, valid_dataset, environment = load_offline_bsuite_dataset(
            bsuite_id=bsuite_id,
            random_prob=noise_level,
            path=str(path),
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            num_shards=num_shards,
            num_valid_shards=num_valid_shards,
            shuffle=shuffle,
            repeat=repeat)
    elif task_name.startswith("dm_control"):
        # DM Control tasks.
        dm_control_task_name = task_name[len("dm_control_"):]
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        root_path = dataset_path.joinpath(f"dm_control_suite/transitions/{dm_control_task_name}_{noise_level}/")
        data_path = f"{run_id}_full"
        train_dataset, valid_dataset, environment = load_offline_dm_control_dataset(
            task_name=dm_control_task_name,
            noise_std=noise_level,
            root_path=str(root_path),
            data_path=str(data_path),
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            num_shards=num_shards,
            num_valid_shards=num_valid_shards,
            shuffle=shuffle,
            repeat=repeat)
    else:
        raise ValueError(f"task name {task_name} is unknown")
    if max_dev_size is not None:
      valid_dataset = valid_dataset.take(
          (max_dev_size + valid_batch_size - 1) // valid_batch_size)
    return train_dataset, valid_dataset, environment


def load_policy_net(
    task_name: str,
    params: Dict[str, Any],
    dataset_path: str,
    environment_spec: specs.EnvironmentSpec = None,
    ):
    dataset_path = Path(dataset_path)
    if task_name.startswith("bsuite"):
        # BSuite tasks.
        bsuite_id = task_name[len("bsuite_"):] + "/0"
        env_noise_level = params["env_noise_level"]
        policy_noise_level = params["policy_noise_level"]
        run_id = params["run_id"]
        path = dataset_path.joinpath(f"bsuite/snapshots/{bsuite_id}_{env_noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))
        observation_network = tf2_utils.to_sonnet_module(functools.partial(
            tf.reshape, shape=(-1,) + environment_spec.observations.shape))
        policy_net = snt.Sequential([
            observation_network,
            policy_net,
            lambda q: trfl.epsilon_greedy(q, epsilon=policy_noise_level).sample(),
        ])
    elif task_name.startswith("dm_control"):
        # DM Control tasks.
        dm_control_task = task_name[len("dm_control_"):]
        env_noise_level = params["env_noise_level"]
        policy_noise_level = params["policy_noise_level"]
        run_id = params["run_id"]
        path = dataset_path.joinpath(f"dm_control_suite/snapshots/{dm_control_task}_{env_noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))

        # act_spec = environment_spec.actions
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)
        policy_net = snt.Sequential([
            observation_network,
            policy_net,
            # Uncomment these two lines to add action noise to target policy.
            acme_utils.GaussianNoise(policy_noise_level),
            networks.ClipToSpec(environment_spec.actions),
        ])
    else:
        raise ValueError(f"task name {task_name} is unknown")
    return policy_net
