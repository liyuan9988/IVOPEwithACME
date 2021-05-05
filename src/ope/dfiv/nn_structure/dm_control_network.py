# pylint: disable=bad-indentation,missing-class-docstring,missing-function-docstring
from typing import Sequence, Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec

from src.utils.tf_linear_reg_utils import outer_prod, add_const_col


class InstrumentalFeature(snt.Module):

    def __init__(self, layer_sizes: Sequence[int]):
        super(InstrumentalFeature, self).__init__()
        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(layer_sizes, activate_final=True)])
        self._feature_dim = layer_sizes[-1] + 1

    def __call__(self, obs, action, training=False):
        feature = self._net(obs, action)
        feature = add_const_col(feature)
        return feature

    def feature_dim(self):
        return self._feature_dim


class ValueFeature(snt.Module):

    def __init__(self, layer_sizes: Sequence[int]):
        super(ValueFeature, self).__init__()
        self._net = snt.Sequential([
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(layer_sizes, activate_final=True)])

    def __call__(self, obs, action, training=False):
        feature = self._net(obs, action)
        return feature


class ValueFunction(snt.Module):

    def __init__(self, layer_sizes: Sequence[int]):
        super(ValueFunction, self).__init__()
        self._feature = ValueFeature(layer_sizes)
        self._weight = tf.Variable(
            tf.zeros((layer_sizes[-1] + 1, 1), dtype=tf.float32))

    def __call__(self, obs, action, training=False):
        feature = self._feature(obs, action, training)
        return tf.matmul(add_const_col(feature), self._weight)

    def feature_dim(self):
        return self._weight.shape[0]

    @property
    def weight(self):
        return self._weight


def make_value_func_dm_control(value_layer_sizes: str = '512,512,256',
                               instrumental_layer_sizes: str = '512,512,256',
                               ) -> Tuple[snt.Module, snt.Module]:
    layer_sizes = list(map(int, value_layer_sizes.split(',')))
    value_function = ValueFunction(layer_sizes=layer_sizes)

    layer_sizes = list(map(int, instrumental_layer_sizes.split(',')))
    instrumental_feature = InstrumentalFeature(layer_sizes=layer_sizes)
    return value_function, instrumental_feature
