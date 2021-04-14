# python3
# pylint: disable=bad-indentation,line-too-long

from absl import app
from absl import flags
from absl import logging

from acme import specs
from acme.utils import counting
from acme.utils import loggers
import ml_collections as collections
from ml_collections.config_flags import config_flags

import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src.ope.deepiv import DeepIVLearner
from src.ope.deepiv import make_ope_networks
from src.utils import generate_train_data
from src.utils import load_policy_net
from src.utils import load_data_and_env
from src.utils import ope_evaluation


# Network flags.
flags.DEFINE_string('density_layer_sizes', '50,50',
                    'Density network hidden layer sizes.')
flags.DEFINE_string('value_layer_sizes', '50,50',
                    'Density network hidden layer sizes.')
flags.DEFINE_integer('num_cat', 10, 'number of mixture components.')

# Agent flags
flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('density_learning_rate', 2e-5, 'learning rate for the mixture density net update')
flags.DEFINE_integer('density_iter', 100000, 'number of iteration for instrumental function')
flags.DEFINE_integer('n_sampling', 10, 'number of samples generated in stage 2')
flags.DEFINE_integer('value_iter', 100000, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 100, 'Number of initial samples for evaluation.')


def get_problem_config():
    """Problem config."""
    problem_config = collections.ConfigDict({
        'task_name': 'bsuite_cartpole',
        'prob_param': {
            'noise_level': 0.2,
            'run_id': 0
        },
        'target_policy_param': {
            'env_noise_level': 0.2,
            'policy_noise_level': 0.1,
            'run_id': 1
        },
        'behavior_policy_param': {
            'env_noise_level': 0.2,
            'policy_noise_level': 0.3,
            'run_id': 1
        },
        'behavior_dataset_size': 0,  # 180000
        'discount': 0.99,
    })
    return problem_config


config_flags.DEFINE_config_dict('problem_config', get_problem_config(),
                                'ConfigDict instance for problem config.')
FLAGS = flags.FLAGS


def main(_):
    problem_config = FLAGS.problem_config

    # Load the offline dataset and environment.
    dataset, _, environment = load_data_and_env(
        problem_config['task_name'], problem_config['prob_param'],
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.batch_size)
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    value_func, mixture_density = make_ope_networks(
        problem_config['task_name'],
        environment_spec=environment_spec,
        density_layer_sizes=FLAGS.density_layer_sizes,
        value_layer_sizes=FLAGS.value_layer_sizes,
        num_cat=FLAGS.num_cat)

    # Load pretrained target policy network.
    target_policy_net = load_policy_net(
        task_name=problem_config['task_name'],
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
          batch_size=FLAGS.batch_size,
          shuffle=True)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DeepIVLearner(
        value_func=value_func,
        mixture_density=mixture_density,
        policy_net=target_policy_net,
        discount=problem_config['discount'],
        value_learning_rate=FLAGS.value_learning_rate,
        density_learning_rate=FLAGS.density_learning_rate,
        n_sampling=FLAGS.n_sampling,
        density_iter=FLAGS.density_iter,
        dataset=dataset,
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
                       discount=problem_config['discount'])
        if learner.state['num_steps'] >= FLAGS.density_iter + FLAGS.value_iter:
          break


if __name__ == '__main__':
    app.run(main)
