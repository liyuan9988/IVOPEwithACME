from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec

from src.utils.tf_linear_reg_utils import outer_prod, add_const_col


class InstrumentalFeature(snt.Module):

    def __init__(self, environment_spec):
        super(InstrumentalFeature, self).__init__()
        self._net = snt.Sequential([snt.Flatten(),
                                    snt.nets.MLP([150, 100, 50], activate_final=True)])
        self.discount_pred = snt.Sequential([snt.Flatten(),
                                             snt.nets.MLP([50, 50, 1])])

        self.n_action = environment_spec.actions.num_values
        self.flat = snt.Flatten()

    def __call__(self, obs, action, training=False):
        action_aug = tf.one_hot(action, depth=self.n_action)
        feature = self._net(tf.concat([self.flat(obs), action_aug], axis=1))
        discount_pred = tf.sigmoid(self.discount_pred(tf.concat([self.flat(obs), action_aug], axis=1)))
        return feature * discount_pred


class ValueFeature(snt.Module):

    def __init__(self, environment_spec):
        super(ValueFeature, self).__init__()
        self._net = snt.Sequential([snt.Flatten(),
                                    snt.nets.MLP([50, 50], activate_final=True)])
        self.n_action = environment_spec.actions.num_values
        self.flat = snt.Flatten()


    def __call__(self, obs, action, training=False):
        action_aug = tf.one_hot(action, depth=self.n_action)
        feature = self._net(tf.concat([self.flat(obs), action_aug], axis=1))
        return feature


class ValueFunction(snt.Module):

    def __init__(self, environment_spec):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature(environment_spec)
        self.n_action = environment_spec.actions.num_values
        self._weight = tf.Variable(
          tf.zeros((51, 1), dtype=tf.float32))

    def __call__(self, obs, action, training=False):
        return tf.matmul(add_const_col(self._feature(obs, action, training)), self._weight)


def make_value_func_bsuite(environment_spec) -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction(environment_spec)
    instrumental_feature = InstrumentalFeature(environment_spec)
    return value_function, instrumental_feature
