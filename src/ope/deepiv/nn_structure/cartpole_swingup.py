from typing import Tuple
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.specs import EnvironmentSpec
import tensorflow_probability as tfp


class MixtureDensity(snt.Module):

    def __init__(self, num_cat=10):
        super(MixtureDensity, self).__init__()
        self._net = snt.Sequential([networks.LayerNormMLP([1024, 1024, 512], activate_final=True)])
        self._logits = snt.Linear(num_cat)
        self._locs_weight = tf.Variable(tf.random.normal([512, num_cat, 5]))
        self._locs_bias = tf.Variable(tf.random.normal([num_cat, 5]))
        self._scale_weight = tf.Variable(tf.random.normal([512, num_cat, 5]))
        self._scale_bias = tf.Variable(tf.random.normal([num_cat, 5]))
        self.num_cat = num_cat

    def __call__(self, obs, action):
        tfd = tfp.distributions
        data = tf.concat([tf2_utils.batch_concat(obs), action], axis=1)
        feature = self._net(data)
        logits = self._logits(feature)
        locs = tf.einsum("ij,jkl->ikl", feature, self._locs_weight) + self._locs_bias
        scales = tf.exp(tf.einsum("ij,jkl->ikl", feature, self._scale_weight) + self._scale_bias)
        res = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits),
                                    components_distribution=tfd.MultivariateNormalDiag(loc=locs,
                                                                                       scale_diag=scales)
                                    )
        return res


class ValueFunction(snt.Module):

    def __init__(self):
        super(ValueFunction, self).__init__()
        self._net = snt.Sequential([snt.Sequential([networks.LayerNormMLP([512, 512, 512], activate_final=True),
                                                    snt.Linear(1)])])

    def __call__(self, obs, action):
        if isinstance(obs, tf.Tensor):
            data = tf.concat([obs, action], axis=1)
        else:
            data = tf.concat([tf2_utils.batch_concat(obs), action], axis=1)
        return self._net(data)


def make_value_func_cartpole() -> Tuple[snt.Module, snt.Module]:
    value_function = ValueFunction()
    mixture_density = MixtureDensity()
    return value_function, mixture_density
