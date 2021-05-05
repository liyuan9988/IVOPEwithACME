# pylint: disable=bad-indentation,missing-class-docstring,missing-function-docstring
from typing import Tuple

from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf


@snt.allow_empty_variables
class RandomFourierFeature(snt.Module):

    def __init__(self, n_component, gamma):
        super(RandomFourierFeature, self).__init__()
        self.n_components = n_component
        self.gamma = gamma

    @snt.once
    def _initialize(self, x):
        n_features = x.shape[1]
        self.random_weights_ = np.sqrt(2 * self.gamma) * tf.random.normal(
            shape=(n_features, self.n_components))

        self.random_offset_ = tf.random.uniform(
            minval=0, maxval=2 * np.pi, shape=(1, self.n_components))

    def __call__(self, x):
        self._initialize(x)
        z_vec = (
            tf.cos(tf.matmul(x, self.random_weights_) + self.random_offset_) /
            np.sqrt(self.n_components))
        return z_vec


@snt.allow_empty_variables
class InstrumentalFeature(snt.Module):

    def __init__(self, n_component, gamma):
        super(InstrumentalFeature, self).__init__()
        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            RandomFourierFeature(n_component=n_component, gamma=gamma)])
        self._feature_dim = n_component

    def __call__(self, obs, action):
        feature = self._net(obs, action)
        return feature

    def feature_dim(self):
        return self._feature_dim


@snt.allow_empty_variables
class ValueFeature(snt.Module):

    def __init__(self, n_component, gamma):
        super(ValueFeature, self).__init__()
        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            RandomFourierFeature(n_component=n_component, gamma=gamma)])
        self._feature_dim = n_component

    def __call__(self, obs, action):
        feature = self._net(obs, action)
        return feature

    def feature_dim(self):
        return self._feature_dim


@snt.allow_empty_variables
class ValueFunction(snt.Module):

    def __init__(self, n_component, gamma):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature(n_component=n_component, gamma=gamma)
        self._weight = tf.Variable(tf.zeros((n_component, 1), dtype=tf.float32))

    def __call__(self, obs, action):
        return tf.matmul(self._feature(obs, action), self._weight)

    @property
    def weight(self):
        return self._weight


def make_value_func_dm_control(n_component=100,
                               gamma=10.0) -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction(
        n_component=n_component, gamma=gamma)
    instrumental_feature = InstrumentalFeature(
        n_component=n_component, gamma=gamma)
    return value_function, instrumental_feature
