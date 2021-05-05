# pylint: disable=bad-indentation,missing-module-docstring,missing-function-docstring
import functools
from typing import Tuple

from acme.specs import EnvironmentSpec
from acme.tf import networks
import sonnet as snt
import tensorflow as tf


def make_value_func_bsuite(environment_spec: EnvironmentSpec,
                           value_layer_sizes: str = '50,50',
                           adversarial_layer_sizes: str = '50,50',
                           ) -> Tuple[snt.Module, snt.Module]:
    action_network = functools.partial(
        tf.one_hot, depth=environment_spec.actions.num_values)

    layer_sizes = list(map(int, value_layer_sizes.split(',')))
    value_function = snt.Sequential([
        networks.CriticMultiplexer(action_network=action_network),
        snt.nets.MLP(layer_sizes, activate_final=True),
        snt.Linear(1)])

    layer_sizes = list(map(int, adversarial_layer_sizes.split(',')))
    advsarial_function = snt.Sequential([
        networks.CriticMultiplexer(action_network=action_network),
        snt.nets.MLP(layer_sizes, activate_final=True),
        snt.Linear(1)])

    return value_function, advsarial_function
