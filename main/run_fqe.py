# python3
# pylint: disable=line-too-long

import copy

from absl import app
from absl import flags
from absl import logging

from acme import specs
from acme.utils import counting
from acme.utils import loggers
from ml_collections.config_flags import config_flags

import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src import utils
from src.ope import fqe

flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')

# Network flags.
flags.DEFINE_string('layer_sizes', '50,50', 'Network hidden layer sizes.')
flags.DEFINE_boolean('distributional', True, 'Distributional critic if True.')
flags.DEFINE_float('vmin', 0., 'Minimum value of the distributional critic')
flags.DEFINE_float('vmax', 100., 'Minimum value of the distributional critic')
flags.DEFINE_integer('num_atoms', 21,
                     'Number of discrete values of the distributional critic')

# Agent flags
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate for the network update')
flags.DEFINE_integer('target_update_period', 100, 'Target net update period.')

flags.DEFINE_integer('max_dev_size', 10*1024, 'Maximum dev dataset size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 1000, 'Number of initial samples for evaluation.')

flags.DEFINE_integer('max_steps', 100000, 'Max number of steps.')


config_flags.DEFINE_config_dict('problem_config', utils.get_problem_config(),
                                'ConfigDict instance for problem config.')
FLAGS = flags.FLAGS


def main(_):
  problem_config = FLAGS.problem_config

  # Load the offline dataset and environment.
  dataset, dev_dataset, environment = utils.load_data_and_env(
      task_name=problem_config['task_name'],
      noise_level=problem_config['noise_level'],
      near_policy_dataset=problem_config['near_policy_dataset'],
      dataset_path=FLAGS.dataset_path,
      batch_size=FLAGS.batch_size,
      max_dev_size=FLAGS.max_dev_size)
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  value_func = fqe.make_ope_networks(
      problem_config['task_name'], environment_spec,
      distributional=FLAGS.distributional,
      layer_sizes=FLAGS.layer_sizes,
      vmin=FLAGS.vmin,
      vmax=FLAGS.vmax,
      num_atoms=FLAGS.num_atoms)
  target_value_func = copy.deepcopy(value_func)

  # Load pretrained target policy network.
  target_policy_net = utils.load_policy_net(
      task_name=problem_config['task_name'],
      noise_level=problem_config['noise_level'],
      near_policy_dataset=problem_config['near_policy_dataset'],
      dataset_path=FLAGS.dataset_path,
      environment_spec=environment_spec)

  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')
  logger = loggers.TerminalLogger('learner')

  # The learner updates the parameters (and initializes them).
  learner = fqe.FQELearner(
      policy_network=target_policy_net,
      critic_network=value_func,
      target_critic_network=target_value_func,
      discount=problem_config['discount'],
      target_update_period=FLAGS.target_update_period,
      vmin=FLAGS.vmin,
      vmax=FLAGS.vmax,
      dataset=dataset,
      distributional=FLAGS.distributional,
      critic_lr=FLAGS.learning_rate,
      counter=learner_counter,
      logger=logger)

  eval_counter = counting.Counter(counter, 'eval')
  eval_logger = loggers.TerminalLogger('eval')

  while True:
    learner.step()
    steps = learner.state['num_steps'].numpy()

    if steps % FLAGS.evaluate_every == 0:
      eval_results = {}
      if dev_dataset is not None:
        eval_results = {'dev_loss': learner.dev_critic_loss(dev_dataset)}
      eval_results.update(utils.ope_evaluation(
          value_func=learner.critic_mean,
          policy_net=target_policy_net,
          environment=environment,
          num_init_samples=FLAGS.evaluate_init_samples,
          discount=problem_config['discount'],
          counter=eval_counter))
      eval_logger.write(eval_results)

    if steps >= FLAGS.max_steps:
      break


if __name__ == '__main__':
  app.run(main)
