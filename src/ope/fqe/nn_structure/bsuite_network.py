# pylint: disable=bad-indentation,missing-module-docstring,missing-function-docstring
import functools

from acme.specs import EnvironmentSpec
from acme.tf import networks
import sonnet as snt
import tensorflow as tf


def make_value_func_bsuite(environment_spec: EnvironmentSpec,
                           distributional: bool = True,
                           layer_sizes: str = '50,50',
                           vmin: float = 0.,
                           vmax: float = 100.,
                           num_atoms: int = 21,
                           ) -> snt.Module:
    layer_sizes = list(map(int, layer_sizes.split(',')))
    action_network = functools.partial(
        tf.one_hot, depth=environment_spec.actions.num_values)
    if distributional:
      head = networks.DiscreteValuedHead(vmin, vmax, num_atoms)
    else:
      head = snt.Linear(1)
    value_function = snt.Sequential([
        networks.CriticMultiplexer(action_network=action_network),
        snt.nets.MLP(layer_sizes, activate_final=True),
        head])
    return value_function
