from typing import Tuple
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme.specs import EnvironmentSpec

from src.utils.tf_linear_reg_utils import outer_prod


class DiscretePolicy(snt.Module):

    def __init__(self, action_spec):
        super(DiscretePolicy, self).__init__()
        self.logit_network = snt.Sequential([snt.Flatten(),
                                             snt.Linear(32),
                                             tf.nn.relu,
                                             snt.Linear(action_spec.num_values)])

    def __call__(self, obs):
        logits = self.logit_network(obs)
        return tfp.distributions.Categorical(logits=logits)


class InstrumentalFeatureNet(snt.Module):

    def __init__(self, action_spec):
        super(InstrumentalFeatureNet, self).__init__()
        self._net = snt.Sequential([snt.Flatten(),
                                    snt.Linear(128),
                                    tf.nn.relu,
                                    snt.Linear(64),
                                    tf.nn.relu,
                                    snt.Linear(64)])
        self.num_action = action_spec.num_values
        self.flat = snt.Flatten()

    def __call__(self, obs, action):
        one_hot = tf.one_hot(action, depth=self.num_action)
        pred = self._net(obs)
        return self.flat(outer_prod(one_hot, pred))


def make_policy_network_bsuite(environment_spec: EnvironmentSpec) -> Tuple[snt.Module, snt.Module, snt.Module]:
    value_feature = snt.Sequential([
        snt.Flatten(),
        snt.Linear(64),
        tf.nn.relu,
        snt.Linear(64),
        tf.nn.relu
    ])

    instrumental_feature = InstrumentalFeatureNet(environment_spec.actions)
    policy_net = DiscretePolicy(environment_spec.actions)
    return value_feature, instrumental_feature, policy_net
