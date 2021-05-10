# python3
# pylint: disable=bad-indentation
"""Compute the ground-truth policy value of a pretrained policy net."""

import pathlib
import sys
import time

from absl import app
from absl import flags
from acme import specs
from acme.agents.tf import actors
from acme.utils import loggers
from ml_collections.config_flags import config_flags
import numpy as np
import tree

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src import utils

# Agent flags
flags.DEFINE_integer('num_episodes', 100, 'number of episodes to evaluate.')
flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')


config_flags.DEFINE_config_dict('problem_config', utils.get_problem_config(),
                                'ConfigDict instance for problem config.')
FLAGS = flags.FLAGS


def main(_):
  problem_config = FLAGS.problem_config

  # Load the offline dataset and environment.
  _, _, environment = utils.load_data_and_env(
      task_name=problem_config['task_name'],
      noise_level=problem_config['noise_level'],
      near_policy_dataset=problem_config['near_policy_dataset'],
      dataset_path=FLAGS.dataset_path,
      batch_size=1)
  environment_spec = specs.make_environment_spec(environment)

  # Load pretrained target policy network.
  policy_net = utils.load_policy_net(
      task_name=problem_config['task_name'],
      noise_level=problem_config['noise_level'],
      near_policy_dataset=problem_config['near_policy_dataset'],
      dataset_path=FLAGS.dataset_path,
      environment_spec=environment_spec)

  actor = actors.FeedForwardActor(policy_network=policy_net)

  logger = loggers.TerminalLogger('ground_truth')

  discount = problem_config['discount']

  returns = []
  lengths = []

  t_start = time.time()
  timestep = environment.reset()
  actor.observe_first(timestep)
  cur_return = 0.
  cur_step = 0
  while len(returns) < FLAGS.num_episodes:

    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    # Have the agent observe the timestep and let the actor update itself.
    actor.observe(action, next_timestep=timestep)

    cur_return += pow(discount, cur_step) * timestep.reward
    cur_step += 1

    if timestep.last():
      # Append return of the current episode, and reset the environment.
      returns.append(cur_return)
      lengths.append(cur_step)
      timestep = environment.reset()
      actor.observe_first(timestep)
      cur_return = 0.
      cur_step = 0

      if len(returns) % (FLAGS.num_episodes // 10) == 0:
        print(f'Run time {time.time() - t_start:0.0f} secs, '
              f'evaluated episode {len(returns)} / {FLAGS.num_episodes}')

  # Returned data include problem configs.
  results = {
      '_'.join(keys): value
      for keys, value in tree.flatten_with_path(problem_config)
  }

  # And computed results.
  results.update({
      'metric_value': np.mean(returns),
      'metric_std_dev': np.std(returns, ddof=0),
      'metric_std_err': np.std(returns, ddof=0) / np.sqrt(len(returns)),
      'length_mean': np.mean(lengths),
      'length_std': np.std(lengths, ddof=0),
      'num_episodes': len(returns),
  })
  logger.write(results)


if __name__ == '__main__':
  app.run(main)
