from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
import sonnet as snt
from typing import Tuple
from .cartpole_swingup import make_value_func_cartpole
from .bsuite_network import make_value_func_bsuite

def make_ope_networks(task_id: str, environment_spec: EnvironmentSpec, instrumental_predict_terminating: bool) -> Tuple[snt.Module, snt.Module]:

    if task_id == "dm_control_cartpole_swingup":
        value_func, instrumental_feature = make_value_func_cartpole()
    elif task_id.startswith("bsuite"):
        value_func, instrumental_feature = make_value_func_bsuite(environment_spec, instrumental_predict_terminating)
    else:
        raise ValueError(f"task id {task_id} not known")

    tf2_utils.create_variables(instrumental_feature,
                               [environment_spec.observations,
                                environment_spec.actions])

    tf2_utils.create_variables(value_func, [environment_spec.observations,
                                            environment_spec.actions])

    return value_func, instrumental_feature
