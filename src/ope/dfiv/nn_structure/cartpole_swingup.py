from typing import Tuple
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme.specs import EnvironmentSpec


class InstrumentalFeature(snt.Module):

    def __init__(self):
        super(InstrumentalFeature, self).__init__()
        self._net = snt.Sequential([snt.Linear(128),
                                    tf.nn.relu,
                                    snt.Linear(64),
                                    tf.nn.relu,
                                    snt.Linear(32),
                                    tf.nn.relu])

        self.flat = tf.keras.layers.Flatten()

    def __call__(self, obs, action):
        data = tf.concat([self.flat(obs), action], axis=1)
        return self._net(data)


class ValueFunction(snt.Module):

    def __init__(self):
        super(ValueFunction, self).__init__()
        self._feature = snt.Sequential([snt.Linear(128),
                                        tf.nn.relu,
                                        snt.Linear(64),
                                        tf.nn.relu,
                                        snt.Linear(32),
                                        tf.nn.relu])

        self.flat = tf.keras.layers.Flatten()
        self._weight = tf.random.uniform((32, 1))

    def __call__(self, obs, action):
        data = tf.concat([self.flat(obs), action], axis=1)
        return tf.matmul(self._feature(data), self._weight)


def make_value_func_cartpole() -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction()
    instrumental_feature = InstrumentalFeature()
    return value_function, instrumental_feature
