from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec
import numpy as np

from src.utils.tf_linear_reg_utils import outer_prod, add_const_col

@snt.allow_empty_variables
class RandomFourierFeature(snt.Module):

    def __init__(self, n_component, gamma=1.0):
        super(RandomFourierFeature, self).__init__()
        self.n_components = n_component
        self.gamma = gamma

    @snt.once
    def _initialize(self, x):
        n_features = x.shape[1]
        self.random_weights_ = np.sqrt(2 * self.gamma) * tf.random.normal(
            shape=(n_features, self.n_components))

        self.random_offset_ = tf.random.uniform(minval=0, maxval=2 * np.pi, shape=(1, self.n_components))

    def __call__(self, x):
        self._initialize(x)
        z_vec = tf.cos(tf.matmul(x, self.random_weights_) + self.random_offset_) / np.sqrt(self.n_components)
        return z_vec

@snt.allow_empty_variables
class InstrumentalFeature(snt.Module):

    def __init__(self, environment_spec):
        super(InstrumentalFeature, self).__init__()
        self.flat = snt.Flatten()
        self.rff = RandomFourierFeature(n_component=100, gamma=10)
        self.n_action = environment_spec.actions.num_values
        self.last_flat = snt.Flatten()

    def __call__(self, obs, action):
        action_aug = tf.one_hot(action, depth=self.n_action)
        feature = self.rff(self.flat(obs))
        return self.last_flat(outer_prod(feature, action_aug))


@snt.allow_empty_variables
class ValueFeature(snt.Module):

    def __init__(self, environment_spec):
        super(ValueFeature, self).__init__()
        self._net = snt.Sequential([snt.Flatten(),
                                    RandomFourierFeature(n_component=100, gamma=10)])
        self.n_action = environment_spec.actions.num_values
        self.last_flat = snt.Flatten()

    def __call__(self, obs, action):
        action_aug = tf.one_hot(action, depth=self.n_action)
        feature = self._net(obs)
        return self.last_flat(outer_prod(add_const_col(feature), action_aug))


@snt.allow_empty_variables
class ValueFunction(snt.Module):

    def __init__(self, environment_spec):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature(environment_spec)
        self.n_action = environment_spec.actions.num_values
        self._weight = tf.Variable(
            tf.zeros((51 * self.n_action, 1), dtype=tf.float32))

    def __call__(self, obs, action):
        return tf.matmul(self._feature(obs, action), self._weight)


def make_value_func_bsuite(environment_spec) -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction(environment_spec)
    instrumental_feature = InstrumentalFeature(environment_spec)
    return value_function, instrumental_feature
