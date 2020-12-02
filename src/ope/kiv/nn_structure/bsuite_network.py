from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec
import numpy as np
from scipy.spatial.distance import cdist

from src.utils.tf_linear_reg_utils import outer_prod, add_const_col


def get_median(X):
    A = X.numpy()
    dist_mat = cdist(A, A, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


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

        self.random_offset_ = tf.random.uniform(minval=0, maxval=2 * np.pi, shape=(1, self.n_components))

    def __call__(self, x):
        self._initialize(x)
        z_vec = tf.cos(tf.matmul(x, self.random_weights_) + self.random_offset_) / np.sqrt(self.n_components)
        return z_vec


@snt.allow_empty_variables
class InstrumentalFeature(snt.Module):

    def __init__(self, environment_spec, n_component, gamma=None):
        super(InstrumentalFeature, self).__init__()
        self.rff = None
        self.n_component = n_component
        self.gamma = None
        self.n_action = environment_spec.actions.num_values
        self.flat = snt.Flatten()

    @snt.once
    def _initialize(self, x):
        if self.gamma is None:
            self.gamma = 1.0 / get_median(x)
        self._net = RandomFourierFeature(n_component=self.n_component, gamma=self.gamma)

    def __call__(self, obs, action):
        action_aug = tf.one_hot(action, depth=self.n_action)
        inputs = tf.concat([self.flat(obs), action_aug], axis=1)
        self._initialize(inputs)
        return self.rff(inputs)


@snt.allow_empty_variables
class ValueFeature(snt.Module):

    def __init__(self, environment_spec, n_component, gamma=None):
        super(ValueFeature, self).__init__()
        self._net = None
        self.gamma = gamma
        self.n_component = n_component
        self.n_action = environment_spec.actions.num_values
        self.flat = snt.Flatten()

    @snt.once
    def _initialize(self, x):
        if self.gamma is None:
            self.gamma = 1.0 / get_median(x)
        self._net = RandomFourierFeature(n_component=self.n_component, gamma=self.gamma)

    def __call__(self, obs, action):
        action_aug = tf.one_hot(action, depth=self.n_action)
        inputs = tf.concat([self.flat(obs), action_aug], axis=1)
        self._initialize(inputs)
        return self._net(inputs)


@snt.allow_empty_variables
class ValueFunction(snt.Module):

    def __init__(self, environment_spec, n_component, gamma=None):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature(environment_spec, n_component=n_component, gamma=gamma)
        self.n_action = environment_spec.actions.num_values
        self._weight = tf.Variable(tf.zeros((n_component, 1), dtype=tf.float32))

    def __call__(self, obs, action):
        return tf.matmul(self._feature(obs, action), self._weight)


def make_value_func_bsuite(environment_spec, n_component=100, gamma=None) -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction(environment_spec, n_component=n_component, gamma=gamma)
    instrumental_feature = InstrumentalFeature(environment_spec, n_component=n_component, gamma=gamma)
    return value_function, instrumental_feature
