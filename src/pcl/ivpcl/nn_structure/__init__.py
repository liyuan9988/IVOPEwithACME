from acme.specs import EnvironmentSpec
import sonnet as snt
from typing import Tuple

from acme.tf import utils as tf2_utils

from .bsuite import make_policy_network_bsuite


def make_policy_network(task_id: str, environment_spec: EnvironmentSpec) -> Tuple[snt.Module, snt.Module, snt.Module]:

    if task_id == "bsuite":
        value_feature, instrumental_feature, policy_net = make_policy_network_bsuite(environment_spec)
    else:
        raise ValueError(f"task id {task_id} unknown")

    tf2_utils.create_variables(policy_net, [environment_spec.observations])
    tf2_utils.create_variables(value_feature, [environment_spec.observations])
    tf2_utils.create_variables(instrumental_feature, [environment_spec.observations, environment_spec.actions])
    return value_feature, instrumental_feature, policy_net
