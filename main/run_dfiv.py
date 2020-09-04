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
import tensorflow.compat.v2 as tf
import trfl

import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_PATH)

from src.utils import generate_rl_unplugged_dataset  # noqa: E402
from src.ope.dfiv import DFIVLearner, make_ope_networks  # noqa: E402

flags.DEFINE_string('dataset_path', '/tmp/dataset', 'Path to dataset.')
flags.DEFINE_string('task_name', 'cartpole_swingup', 'Task name.')
flags.DEFINE_enum('task_class', 'control_suite',
                  ['humanoid', 'rodent', 'control_suite'],
                  'Task class.')
flags.DEFINE_string('target_policy_path', '', 'Path to target policy snapshot')


# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_learning_rate', 2e-5, 'learning rate for the instrumental_net update')
flags.DEFINE_float('policy_learning_rate', 2e-5, 'learning rate for the policy_net update')
flags.DEFINE_float('stage1_reg', 1e-3, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-3, 'ridge regularizer for stage 2 regression')
flags.DEFINE_integer('instrumental_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('value_iter', 1, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def main(_):
  # Load the offline dataset and environment.
  dataset, environment = generate_rl_unplugged_dataset(
      FLAGS.task_class, FLAGS.task_name, FLAGS.dataset_path)
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  value_func, instrumental_feature = make_ope_networks(environment_spec)

  # Load pretrained target policy network.
  policy_net = tf.saved_model.load(FLAGS.target_policy_path)

  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')

  # The learner updates the parameters (and initializes them).
  learner = DFIVLearner(
      value_func=value_func,
      instrumental_feature=instrumental_feature,
      policy_net=policy_net,
      value_learning_rate=FLAGS.value_learning_rate,
      instrumental_learning_rate=FLAGS.instrumental_learning_rate,
      stage1_reg=FLAGS.stage1_reg,
      stage2_reg=FLAGS.stage2_reg,
      instrumental_iter=FLAGS.instrumental_iter,
      value_iter=FLAGS.value_iter,
      dataset=dataset,
      counter=learner_counter)


if __name__ == '__main__':
  app.run(main)
