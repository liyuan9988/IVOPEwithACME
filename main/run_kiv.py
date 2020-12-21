# python3
# pylint: disable=bad-indentation,line-too-long

from absl import app
from absl import flags
from absl import logging

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

from src.ope.kiv_batch import KIVLearner
from src.ope.kiv_batch import make_ope_networks
from src.utils import generate_train_data
from src.utils import load_data_and_env
from src.utils import load_policy_net
from src.utils import ope_evaluation
from src.utils import get_bsuite_median

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
flags.DEFINE_integer('max_dev_size', 10*1024, 'Maximum dev dataset size.')

flags.DEFINE_float('stage1_reg', 1e-5, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-5, 'ridge regularizer for stage 2 regression')
flags.DEFINE_integer('n_component', 512, 'Number of random Fourier features.')
flags.DEFINE_float('gamma', None, 'Gamma in Gaussian kernel.')

flags.DEFINE_integer('evaluate_init_samples', 100, 'Number of initial samples for evaluation.')

flags.DEFINE_integer('max_steps', 1, 'Max number of steps.')

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
            'env_noise_level': 0.0,
            'policy_noise_level': 0.0,
            'run_id': 1
        },
        'behavior_policy_param': {
            'env_noise_level': 0.0,
            'policy_noise_level': 0.2,
            'run_id': 1
        },
        'behavior_dataset_size': 180000,
        'discount': 0.99,
    }

    # Load the offline dataset and environment.
    dataset, dev_dataset, environment = load_data_and_env(
        problem_config['task_name'], problem_config['prob_param'],
        dataset_path=FLAGS.dataset_path,
        max_dev_size=FLAGS.max_dev_size,
        shuffle=False,
        repeat=False)
    environment_spec = specs.make_environment_spec(environment)

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
          batch_size=problem_config['behavior_dataset_size'] // 4,
          shuffle=False)
      dev_dataset = None

    """
        task_gamma_map = {
            'bsuite_catch': 0.25,
            'bsuite_mountain_car': 0.5,
            'bsuite_cartpole': 0.44,
        }
        gamma = FLAGS.gamma or task_gamma_map[problem_config['task_name']]
    """

    gamma = utils.get_median(
        problem_config['task_name'], environment_spec, dataset)

    # Create the networks to optimize.
    value_func, instrumental_feature = make_ope_networks(
        problem_config['task_name'], environment_spec,
        n_component=FLAGS.n_component, gamma=gamma)

    # Load pretrained target policy network.
    target_policy_net = load_policy_net(
        task_name=problem_config['task_name'],
        params=problem_config['target_policy_param'],
        environment_spec=environment_spec,
        dataset_path=FLAGS.dataset_path)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    num_batches = len(dataset)
    stage1_batch = num_batches // 2
    stage2_batch = num_batches - stage1_batch
    learner = KIVLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy_net=target_policy_net,
        discount=problem_config['discount'],
        stage1_reg=FLAGS.stage1_reg,
        stage2_reg=FLAGS.stage2_reg,
        stage1_batch=stage1_batch,
        stage2_batch=stage2_batch,
        dataset=dataset,
        valid_dataset=dev_dataset,
        counter=learner_counter)

    eval_logger = loggers.TerminalLogger('eval')

    while True:
        learner.step()
        results = {'gamma': gamma}
        results.update(ope_evaluation(
            value_func=value_func,
            policy_net=target_policy_net,
            environment=environment,
            num_init_samples=FLAGS.evaluate_init_samples,
            mse_samples=18,
            discount=problem_config['discount']))
        eval_logger.write(results)
        if learner.state['num_steps'] >= FLAGS.max_steps:
            break


if __name__ == '__main__':
    app.run(main)
