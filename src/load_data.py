from pathlib import Path
import tensorflow as tf

from src.utils import load_offline_dm_control_dataset, load_offline_bsuite_dataset

DATA_PATH = Path(__file__).resolve().parent.parent.joinpath("offline_dataset").joinpath("stochastic")


def load_data_and_env(task_name: str, params: dict):
    if task_name == "bsuite_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"bsuite/transitions/cartpole_swingup/0_{noise_level}/{run_id}_full")
        dataset, environment = load_offline_bsuite_dataset(bsuite_id="cartpole_swingup", path=str(path))
    elif task_name == "dm_control_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        root_path = DATA_PATH.joinpath(f"dm_control_suite/transitions/cartpole_swingup_{noise_level}/")
        data_path = f"{run_id}_full"
        dataset, environment = load_offline_dm_control_dataset(task_name="cartpole_swingup",
                                                               root_path=str(root_path),
                                                               data_path=str(data_path))
    else:
        raise ValueError(f"task name {task_name} is unknown")
    return dataset, environment


def load_policy_net(task_name: str, params: dict):
    if task_name == "bsuite_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"bsuite/snapshots/cartpole_swingup/0_{noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))
    elif task_name == "dm_control_cartpole_swingup":
        noise_level = params["noise_level"]
        run_id = params["run_id"]
        path = DATA_PATH.joinpath(f"dm_control_suite/snapshots/cartpole_swingup_{noise_level}/{run_id}_full")
        policy_net = tf.saved_model.load(str(path))
    else:
        raise ValueError(f"task name {task_name} is unknown")
    return policy_net
