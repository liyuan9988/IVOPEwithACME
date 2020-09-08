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
import tensorflow as tf
import trfl

import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_PATH)

from src.utils import generate_rl_unplugged_dataset  # noqa: E402
from src.ope.deepiv import DeepIVLearner, make_ope_networks  # noqa: E402

flags.DEFINE_string('dataset_path', '/tmp/dataset', 'Path to dataset.')
flags.DEFINE_string('task_name', 'cartpole_swingup', 'Task name.')
flags.DEFINE_enum('task_class', 'control_suite',
                  ['humanoid', 'rodent', 'control_suite'],
                  'Task class.')
flags.DEFINE_string('target_policy_path', '', 'Path to target policy snapshot')

# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('density_learning_rate', 2e-5, 'learning rate for the mixture density net update')
flags.DEFINE_integer('density_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('n_sampling', 10, 'number of samples generated in stage 2')
flags.DEFINE_integer('value_iter', 10, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 2000, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 10, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS

def eval_model(test_data, value_func, policy):
    current_obs, action, reward, discount, next_obs, _ = test_data.data
    next_action = policy(tf2_utils.batch_concat(next_obs))
    target = tf.expand_dims(reward, axis=1) + tf.expand_dims(discount, axis=1) * value_func(next_obs, next_action)
    return tf.norm(target - value_func(current_obs, action)) ** 2

def main(_):
    # Load the offline dataset and environment.
    full_dataset, environment = generate_rl_unplugged_dataset(
        FLAGS.task_class, FLAGS.task_name, FLAGS.dataset_path)
    environment_spec = specs.make_environment_spec(environment)

    full_dataset = full_dataset.shuffle(10000)
    test_data = full_dataset.take(1000)
    train_data = full_dataset.skip(1000)
    train_data = train_data.shuffle(20000)

    test_data = test_data.batch(1000)
    test_data = next(iter(test_data))

    dataset = train_data.batch(FLAGS.batch_size)



    # Create the networks to optimize.
    value_func, mixture_density = make_ope_networks("cartpole_swingup", environment_spec)

    # Load pretrained target policy network.
    policy_net = tf.saved_model.load(FLAGS.target_policy_path)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DeepIVLearner(
        value_func=value_func,
        mixture_density=mixture_density,
        policy_net=policy_net,
        value_learning_rate=FLAGS.value_learning_rate,
        density_learning_rate=FLAGS.density_learning_rate,
        n_sampling=FLAGS.n_sampling,
        density_iter=FLAGS.density_iter,
        value_iter=FLAGS.value_iter,
        dataset=dataset,
        counter=learner_counter)

    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        print(eval_model(test_data, value_func, policy_net))


if __name__ == '__main__':
    app.run(main)
