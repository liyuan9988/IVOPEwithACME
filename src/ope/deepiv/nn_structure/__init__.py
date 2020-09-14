from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
import sonnet as snt
from typing import Tuple
from .cartpole_swingup import make_value_func_cartpole
from .bsuite_cartpole_swingup import bsuite_make_value_func_cartpole


def make_ope_networks(task_id: str, environment_spec: EnvironmentSpec) -> Tuple[snt.Module, snt.Module]:

    if task_id == "cartpole_swingup" or  task_id == "dm_control_cartpole_swingup":
        value_func, mixture_density = make_value_func_cartpole()
    elif task_id == "bsuite_cartpole_swingup":
        value_func, mixture_density = bsuite_make_value_func_cartpole()
    else:
        raise ValueError(f"task id {task_id} not known")

    tf2_utils.create_variables(mixture_density,
                               [environment_spec.observations,
                                environment_spec.actions])

    tf2_utils.create_variables(value_func, [environment_spec.observations,
                                            environment_spec.actions])

    return value_func, mixture_density