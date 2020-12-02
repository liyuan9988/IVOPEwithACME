# pylint: disable=bad-indentation,missing-class-docstring,missing-function-docstring
from typing import Sequence, Tuple

from acme.specs import EnvironmentSpec
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class MixtureDensity(snt.Module):

    def __init__(self,
                 environment_spec: EnvironmentSpec,
                 layer_sizes: Sequence[int],
                 num_cat=10):
        super(MixtureDensity, self).__init__()

        if isinstance(environment_spec.observations, dict):
            obs_size = 0
            for obs_spec in environment_spec.observations.values():
                obs_size += int(np.prod(obs_spec.shape))
        else:
            obs_size = int(np.prod(environment_spec.observations.shape))
        self.obs_size = obs_size
        self.num_cat = num_cat

        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(layer_sizes, activate_final=True)])

        self._discount_logits = snt.Linear(1)
        self._mix_logits = snt.Linear(num_cat)
        self._locs = snt.Linear(num_cat * self.obs_size)
        self._scales = snt.Linear(num_cat * self.obs_size)

    def __call__(self, obs, action):
        tfd = tfp.distributions
        feature = self._net(obs, action)
        mix_logits = self._mix_logits(feature)
        locs = tf.reshape(self._locs(feature),
                          [-1, self.num_cat, self.obs_size])
        scales = tf.exp(tf.reshape(self._scales(feature),
                                   [-1, self.num_cat, self.obs_size]))
        obs_distr = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=mix_logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=locs, scale_diag=scales))
        discount_logits = self._discount_logits(feature)
        # Reshape to a 1D tensor to match the batch shape of obs_distr.
        discount_logits = tf.squeeze(discount_logits, axis=1)
        discount_distr = tfd.Bernoulli(logits=discount_logits)
        return obs_distr, discount_distr


class ValueFunction(snt.Module):

    def __init__(self, layer_sizes: Sequence[int]):
        super(ValueFunction, self).__init__()
        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(layer_sizes, activate_final=True),
            snt.Linear(1)])

    def __call__(self, obs, action):
        return self._net(obs, action)


def make_value_func_dm_control(environment_spec: EnvironmentSpec,
                               value_layer_sizes: str = '512,512,256',
                               density_layer_sizes: str = '512,512,256',
                               num_cat: int = 10,
                               ) -> Tuple[snt.Module, snt.Module]:
    layer_sizes = list(map(int, value_layer_sizes.split(',')))
    value_function = ValueFunction(layer_sizes=layer_sizes)

    layer_sizes = list(map(int, density_layer_sizes.split(',')))
    mixture_density = MixtureDensity(environment_spec,
                                     layer_sizes=layer_sizes,
                                     num_cat=num_cat)
    return value_function, mixture_density
