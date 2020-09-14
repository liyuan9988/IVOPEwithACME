from pathlib import Path
from acme import specs
from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import trfl
import sonnet as snt

from src.utils import load_offline_dm_control_dataset, load_offline_bsuite_dataset
from src.utils import acme_utils

DATA_PATH = Path(__file__).resolve().parent.parent.joinpath("offline_dataset").joinpath("stochastic")


def load_data_and_env(task_name: str, params: dict):
    if task_name == "bsuite_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"bsuite/transitions/cartpole_swingup/0_{noise_level}/{run_id}_full")
        dataset, environment = load_offline_bsuite_dataset(
            bsuite_id="cartpole_swingup/0",
            random_prob=noise_level,
            path=str(path))
    elif task_name == "dm_control_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        root_path = DATA_PATH.joinpath(f"dm_control_suite/transitions/cartpole_swingup_{noise_level}/")
        data_path = f"{run_id}_full"
        dataset, environment = load_offline_dm_control_dataset(
            task_name="cartpole_swingup",
            noise_std=noise_level,
            root_path=str(root_path),
            data_path=str(data_path))
    else:
        raise ValueError(f"task name {task_name} is unknown")
    return dataset, environment


def load_policy_net(
    task_name: str,
    params: dict,
    environment_spec: specs.EnvironmentSpec = None,
    ):
    if task_name == "bsuite_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"bsuite/snapshots/cartpole_swingup/0_{noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))
        policy_net = snt.Sequential([
            policy_net,
            lambda q: trfl.epsilon_greedy(q, epsilon=0.0).sample(),
        ])
    elif task_name == "dm_control_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"dm_control_suite/snapshots/cartpole_swingup_{noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))

        # act_spec = environment_spec.actions
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)
        policy_net = snt.Sequential([
            observation_network,
            policy_net,
            # Uncomment these two lines to add action noise to target policy.
            # acme_utils.GaussianNoise(0.0),
            # networks.ClipToSpec(act_spec),
        ])
    else:
        raise ValueError(f"task name {task_name} is unknown")
    return policy_net