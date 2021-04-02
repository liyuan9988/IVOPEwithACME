# pylint: disable=bad-indentation,missing-module-docstring,missing-class-docstring,missing-function-docstring
from acme.tf import networks
import sonnet as snt


def make_value_func_dm_control(layer_sizes: str = '512,512,256') -> snt.Module:
    layer_sizes = list(map(int, layer_sizes.split(',')))
    value_function = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(layer_sizes, activate_final=True),
        snt.Linear(1)])
    return value_function
