# pylint: disable=bad-indentation,missing-module-docstring,missing-class-docstring,missing-function-docstring
from acme.tf import networks
import sonnet as snt


def make_value_func_dm_control(distributional: bool = True,
                               layer_sizes: str = '512,512,256',
                               vmin: float = 0.,
                               vmax: float = 100.,
                               num_atoms: int = 21,
                               ) -> snt.Module:
    layer_sizes = list(map(int, layer_sizes.split(',')))
    if distributional:
      head = networks.DiscreteValuedHead(vmin, vmax, num_atoms)
    else:
      head = snt.Linear(1)
    value_function = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(layer_sizes, activate_final=True),
        head])
    return value_function
