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

from absl import app
from absl import flags
import acme

from acme.agents.tf import actors
from acme.agents.tf.bc import learning
from acme.tf import networks as acme_nets
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme import specs

import sonnet as snt
import trfl

import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_PATH)

from src.utils import generate_dataset  # noqa: E402
from src.pcl.ivpcl import IVPCLLearner, make_policy_network  # noqa: E402

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')

# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_learning_rate', 2e-5, 'learning rate for the instrumental_net update')
flags.DEFINE_float('policy_learning_rate', 2e-5, 'learning rate for the policy_net update')
flags.DEFINE_float('stage1_reg', 1e-3, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-3, 'ridge regularizer for stage 2 regression')
flags.DEFINE_float('entropy_reg', 0.1, 'entropy regularizer for policy')
flags.DEFINE_integer('instrumental_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('value_iter', 1, 'number of iteration for value function')
flags.DEFINE_integer('policy_iter', 5, 'number of iteration for policy')

flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def main(_):
    dataset, environment = generate_dataset(FLAGS)
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    value_feature, instrumental_feature, policy_net = make_policy_network("bsuite", environment_spec)

    # If the agent is non-autoregressive use epsilon=0 which will be a greedy
    # policy.
    evaluator_network = snt.Sequential([
        policy_net,
        acme_nets.StochasticSamplingHead()
    ])

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
    learner = IVPCLLearner(
        value_feature=value_feature,
        instrumental_feature=instrumental_feature,
        policy_net=policy_net,
        value_learning_rate=FLAGS.value_learning_rate,
        instrumental_learning_rate=FLAGS.instrumental_learning_rate,
        policy_learning_rate=FLAGS.policy_learning_rate,
        entropy_reg=FLAGS.entropy_reg,
        stage1_reg=FLAGS.stage1_reg,
        stage2_reg=FLAGS.stage2_reg,
        instrumental_iter=FLAGS.instrumental_iter,
        value_iter=FLAGS.value_iter,
        policy_iter=FLAGS.policy_iter,
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
