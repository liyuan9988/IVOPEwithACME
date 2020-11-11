# python3
# pylint: disable=bad-indentation,line-too-long

from absl import app
from absl import flags
from absl import logging
import acme

from acme import specs
from acme.agents.tf import actors
from acme.agents.tf.bc import learning
from acme.tf import networks as acme_nets
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers

import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl

import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src.ope.dfiv import DFIVLearner
from src.ope.dfiv import make_ope_networks
from src.utils import generate_train_data
from src.utils import load_data_and_env
from src.utils import load_policy_net
from src.utils import ope_evaluation

flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')
flags.DEFINE_string('task_name', 'cartpole_swingup', 'Task name.')
flags.DEFINE_enum('task_class', 'control_suite',
                  ['humanoid', 'rodent', 'control_suite'],
                  'Task class.')
flags.DEFINE_string('target_policy_path', '', 'Path to target policy snapshot')

# Agent flags
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')

flags.DEFINE_float('value_learning_rate', 1e-4, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_learning_rate', 1e-3, 'learning rate for the instrumental_net update')
flags.DEFINE_float('stage1_reg', 1e-5, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-5, 'ridge regularizer for stage 2 regression')
flags.DEFINE_float('instrumental_reg', 1e-5, 'ridge regularizer instrumental')
flags.DEFINE_float('value_reg', 1e-5, 'ridge regularizer for value_reg')

flags.DEFINE_integer('instrumental_iter', 10, 'number of iteration for instrumental function')
flags.DEFINE_integer('value_iter', 10, 'number of iteration for value function')


flags.DEFINE_integer('evaluate_every', 1, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 100, 'Number of initial samples for evaluation.')

flags.DEFINE_integer('max_steps', 100000, 'Max number of steps.')
flags.DEFINE_float('d_tm1_weight', 0.01,  # 0.01 for cartpole, 0.03 for catch and mountain_car.
                   'Weights of terminal states.')
flags.DEFINE_boolean('include_terminal', False, 'Generate dataset with terminal absorbing state.')
flags.DEFINE_boolean('ignore_d_tm1', False, 'Always set d_tm1 = 1.0 if True.')


FLAGS = flags.FLAGS


def main(_):
    # Load the offline dataset and environment.
    problem_config = {
        'task_name': 'bsuite_cartpole',
        'prob_param': {
            'noise_level': 0.2,
            'run_id': 0
        },
        'target_policy_param': {
            'env_noise_level': 0.2,
            'policy_noise_level': 0.0,
            'run_id': 1
        },
        'behavior_policy_param': {
            'env_noise_level': 0.2,
            'policy_noise_level': 0.2,
            'run_id': 1
        },
        'behavior_dataset_size': 180000,
        'discount': 0.99,
    }
    dataset, environment = load_data_and_env(
        problem_config['task_name'], problem_config['prob_param'],
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.batch_size)
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    value_func, instrumental_feature = make_ope_networks(
        problem_config['task_name'], environment_spec)


    # Load pretrained target policy network.
    target_policy_net = load_policy_net(task_name=problem_config['task_name'],
                                        params=problem_config['target_policy_param'],
                                        environment_spec=environment_spec,
                                        dataset_path=FLAGS.dataset_path)

    if problem_config['behavior_dataset_size'] > 0:
      # Use behavior policy to generate an off-policy dataset and replace
      # the pre-generated offline dataset.
      logging.warning('Ignore offline dataset')
      dataset = generate_train_data(
          task_name=problem_config['task_name'],
          behavior_policy_param=problem_config['behavior_policy_param'],
          dataset_path=FLAGS.dataset_path,
          environment=environment,
          dataset_size=problem_config['behavior_dataset_size'],
          batch_size=problem_config['behavior_dataset_size'] // 2,
          shuffle=False,
          include_terminal=FLAGS.include_terminal,
          ignore_d_tm1=FLAGS.ignore_d_tm1)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DFIVLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy_net=target_policy_net,
        discount=problem_config['discount'],
        value_learning_rate=FLAGS.value_learning_rate,
        instrumental_learning_rate=FLAGS.instrumental_learning_rate,
        stage1_reg=FLAGS.stage1_reg,
        stage2_reg=FLAGS.stage2_reg,
        value_reg=FLAGS.value_reg,
        instrumental_reg=FLAGS.instrumental_reg,
        instrumental_iter=FLAGS.instrumental_iter,
        value_iter=FLAGS.value_iter,
        dataset=dataset,
        d_tm1_weight=FLAGS.d_tm1_weight,
        counter=learner_counter)

    eval_logger = loggers.TerminalLogger('eval')

    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        ope_evaluation(value_func=value_func,
                       policy_net=target_policy_net,
                       environment=environment,
                       logger=eval_logger,
                       num_init_samples=FLAGS.evaluate_init_samples,
                       mse_samples=18,
                       discount=problem_config['discount'])
        if learner.state['num_steps'] >= FLAGS.max_steps:
            break


if __name__ == '__main__':
    app.run(main)
