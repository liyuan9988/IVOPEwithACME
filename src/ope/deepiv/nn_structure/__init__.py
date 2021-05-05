# pylint: disable=bad-indentation,missing-class-docstring,missing-function-docstring
from typing import Any, Dict, Tuple

from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
import sonnet as snt
from .bsuite_network import make_value_func_bsuite
from .dm_control_network import make_value_func_dm_control


def make_ope_networks(task_id: str,
                      environment_spec: EnvironmentSpec,
                      **network_params: Dict[str, Any],
                      ) -> Tuple[snt.Module, snt.Module]:
    if task_id.startswith("dm_control"):
        value_func, mixture_density = make_value_func_dm_control(
            environment_spec, **network_params)
    elif task_id.startswith("bsuite"):
        value_func, mixture_density = make_value_func_bsuite(
            environment_spec, **network_params)
    else:
        raise ValueError(f"task id {task_id} not known")

    tf2_utils.create_variables(mixture_density,
                               [environment_spec.observations,
                                environment_spec.actions])

    tf2_utils.create_variables(value_func, [environment_spec.observations,
                                            environment_spec.actions])

    return value_func, mixture_density
