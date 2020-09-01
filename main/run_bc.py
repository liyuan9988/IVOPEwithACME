# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example BC running on BSuite."""

import functools


from absl import app
from absl import flags
import acme


from acme.agents.tf import actors
from acme.agents.tf.bc import learning

from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme import specs

import sonnet as snt
import trfl

from src.utils import generate_dataset

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')

# Agent flags
flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_float('epsilon', 0., 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def make_policy_network(action_spec: specs.DiscreteArray) -> snt.Module:
    return snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([64, 64, action_spec.num_values]),
    ])


def main(_):
    dataset, environment = generate_dataset(FLAGS)
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    policy_network = make_policy_network(environment_spec.actions)

    # If the agent is non-autoregressive use epsilon=0 which will be a greedy
    # policy.
    evaluator_network = snt.Sequential([
        policy_network,
        lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
    ])

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(policy_network, [environment_spec.observations])

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # Create the actor which defines how we take actions.
    evaluation_network = actors.FeedForwardActor(evaluator_network)

    eval_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=evaluation_network,
        counter=counter,
        logger=loggers.TerminalLogger('evaluation', time_delta=1.))

    # The learner updates the parameters (and initializes them).
    learner = learning.BCLearner(
        network=policy_network,
        learning_rate=FLAGS.learning_rate,
        dataset=dataset,
        counter=learner_counter)

    # Run the environment loop.
    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        learner_counter.increment(learner_steps=FLAGS.evaluate_every)
        eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
    app.run(main)
