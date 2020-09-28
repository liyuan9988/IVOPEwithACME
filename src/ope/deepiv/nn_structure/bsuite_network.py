import numpy as np
from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec
import tensorflow_probability as tfp


class MixtureDensity(snt.Module):

    def __init__(self, environment_spec, num_cat=10):
        super(MixtureDensity, self).__init__()
        self.obs_shape = environment_spec.observations.shape
        self.n_action = environment_spec.actions.num_values
        self.obs_size = np.prod(self.obs_shape)
        self.num_cat = num_cat

        # self._net = snt.nets.MLP([150, 100, 50], activate_final=True)
        self._net = snt.nets.MLP([50, 50], activate_final=True)
        self._discount_logits = snt.Linear(1)
        self._mix_logits = snt.Linear(num_cat)
        self._locs = snt.Linear(num_cat * self.obs_size)
        self._scales = snt.Linear(num_cat * self.obs_size)

        self.flat = snt.Flatten()

    def __call__(self, obs, action):
        tfd = tfp.distributions
        action_aug = tf.one_hot(action, depth=self.n_action)
        data = tf.concat([self.flat(obs), action_aug], axis=1)
        feature = self._net(data)

        mix_logits = self._mix_logits(feature)
        locs = tf.reshape(self._locs(feature), [-1, self.num_cat, self.obs_size])
        scales = tf.exp(tf.reshape(self._scales(feature), [-1, self.num_cat, self.obs_size]))
        obs_distr = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=mix_logits),
                                          components_distribution=tfd.MultivariateNormalDiag(loc=locs,
                                                                                             scale_diag=scales)
                                          )
        discount_logits = self._discount_logits(feature)
        discount_distr = tfd.Bernoulli(logits=discount_logits)
        return obs_distr, discount_distr


class ValueFunction(snt.Module):

    def __init__(self, environment_spec):
        super(ValueFunction, self).__init__()
        self._net = snt.nets.MLP([50, 50, 1])
        self.n_action = environment_spec.actions.num_values
        self.flat = snt.Flatten()

    def __call__(self, obs, action):
        action_aug = tf.one_hot(action, depth=self.n_action)
        return self._net(tf.concat([self.flat(obs), action_aug], axis=1))


def bsuite_make_value_func(environment_spec, num_cat) -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction(environment_spec)
    mixture_density = MixtureDensity(environment_spec, num_cat)
    return value_function, mixture_density
