# pylint: disable=bad-indentation,missing-class-docstring,missing-function-docstring
import functools
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

        action_network = functools.partial(
            tf.one_hot, depth=environment_spec.actions.num_values)
        self._net = snt.Sequential([
            networks.CriticMultiplexer(action_network=action_network),
            snt.nets.MLP(layer_sizes, activate_final=True)])

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

    def __init__(self,
                 environment_spec: EnvironmentSpec,
                 layer_sizes: Sequence[int]):
        super(ValueFunction, self).__init__()

        action_network = functools.partial(
            tf.one_hot, depth=environment_spec.actions.num_values)
        self._net = snt.Sequential([
            networks.CriticMultiplexer(action_network=action_network),
            snt.nets.MLP(layer_sizes, activate_final=True),
            snt.Linear(1)])

    def __call__(self, obs, action):
        return self._net(obs, action)


def make_value_func_bsuite(environment_spec: EnvironmentSpec,
                           value_layer_sizes: str = '50,50',
                           density_layer_sizes: str = '50,50',
                           num_cat: int = 10,
                           ) -> Tuple[snt.Module, snt.Module]:
    layer_sizes = list(map(int, value_layer_sizes.split(',')))
    value_function = ValueFunction(environment_spec, layer_sizes=layer_sizes)

    layer_sizes = list(map(int, density_layer_sizes.split(',')))
    mixture_density = MixtureDensity(environment_spec,
                                     layer_sizes=layer_sizes,
                                     num_cat=num_cat)
    return value_function, mixture_density
