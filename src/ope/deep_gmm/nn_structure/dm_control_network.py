# pylint: disable=bad-indentation,missing-module-docstring,missing-function-docstring
from typing import Tuple

from acme.tf import networks
import sonnet as snt


def make_value_func_dm_control(value_layer_sizes: str = '512,512,256',
                               adversarial_layer_sizes: str = '512,512,256',
                               ) -> Tuple[snt.Module, snt.Module]:
    layer_sizes = list(map(int, value_layer_sizes.split(',')))
    value_function = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(layer_sizes, activate_final=True),
        snt.Linear(1)])

    layer_sizes = list(map(int, adversarial_layer_sizes.split(',')))
    advsarial_function = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(layer_sizes, activate_final=True),
        snt.Linear(1)])

    return value_function, advsarial_function
