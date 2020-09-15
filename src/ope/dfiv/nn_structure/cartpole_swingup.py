from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec

from src.utils.tf_linear_reg_utils import add_const_col

class InstrumentalFeature(snt.Module):

    def __init__(self):
        super(InstrumentalFeature, self).__init__()
        self._net = snt.Sequential([networks.LayerNormMLP([1024,1024,512], activate_final=True)])

    def __call__(self, obs, action):
        data = tf.concat([tf2_utils.batch_concat(obs), action], axis=1)
        return self._net(data)

class ValueFeature(snt.Module):

    def __init__(self):
        super(ValueFeature, self).__init__()
        self._net = snt.Sequential([snt.Sequential([networks.LayerNormMLP([512, 512, 512], activate_final=True)])])

    def __call__(self, obs, action):
        data = tf.concat([tf2_utils.batch_concat(obs), action], axis=1)
        return self._net(data)


class ValueFunction(snt.Module):

    def __init__(self):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature()
        self._weight = tf.random.uniform((513, 1))

    def __call__(self, obs, action):
        return tf.matmul(add_const_col(self._feature(obs, action)), self._weight)


def make_value_func_cartpole() -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction()
    instrumental_feature = InstrumentalFeature()
    return value_function, instrumental_feature
