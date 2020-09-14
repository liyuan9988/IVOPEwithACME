# python3
"""Compute the ground-truth policy value of a pretrained policy net.

Results (Mean estimate +- 1-sigma) with discount = 0.9 with discount = 0.99.

DM Control Suite:
cartpole_swingup: 32.383 +- 0.009

BSuite:
catch/0: 0.923 +- 0.000
cartpole/0: 99.996 +- 0.000
"""

import pathlib
import sys
import time

from absl import app
from absl import flags
from acme import specs
from acme.agents.tf import actors
from acme.utils import loggers
import numpy as np
import tree

ROOT_PATH = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_PATH)
from src.load_data import load_policy_net, load_data_and_env
from src.ope.deepiv import DeepIVLearner, make_ope_networks  # noqa: E402

# Agent flags
flags.DEFINE_integer('num_episodes', 100, 'number of episodes to evaluate.')
FLAGS = flags.FLAGS


def main(_):
  # problem_config = {
  #     'task_name': 'dm_control_cartpole_swingup',
  #     'prob_param': {
  #         'noise_level': 0.0,
  #         'run_id': 0
  #     },
  #     'policy_param': {
  #         'noise_level': 0.0,
  #         'run_id': 1
  #     },
  #     'discount': 0.99,
  # }

  problem_config = {
      # 'task_name': 'bsuite_cartpole_swingup',
      # 'task_name': 'bsuite_catch',
      'task_name': 'bsuite_cartpole',
      'prob_param': {
          'noise_level': 0.0,
          'run_id': 0
      },
      'policy_param': {
          'noise_level': 0.0,
          'run_id': 1
      },
      'discount': 0.99,
  }

  # Load the offline dataset and environment.
  _, environment = load_data_and_env(problem_config['task_name'],
                                     problem_config['prob_param'])
  environment_spec = specs.make_environment_spec(environment)

  # Load pretrained target policy network.
  policy_net = load_policy_net(task_name=problem_config['task_name'],
                               params=problem_config['policy_param'],
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
