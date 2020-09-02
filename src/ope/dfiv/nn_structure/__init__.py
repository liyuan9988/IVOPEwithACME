from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
import sonnet as snt
from typing import Tuple
# from .bsuite import make_policy_network_bsuite


def make_ope_networks(task_id: str, environment_spec: EnvironmentSpec) -> Tuple[snt.Module, snt.Module]:

  # TODO(liyuan): implement value_feature, instrumental_feature
  # if task_id == "bsuite":
  #     value_feature, instrumental_feature, policy_net = make_policy_network_bsuite(environment_spec)
  # else:
  #     raise ValueError(f"task id {task_id} unknown")

  tf2_utils.create_variables(value_feature, [environment_spec.observations,
                                             environment_spec.actions])
  tf2_utils.create_variables(instrumental_feature,
                             [environment_spec.observations,
                              environment_spec.actions])
  return value_feature, instrumental_feature
