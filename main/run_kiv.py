# python3
# pylint: disable=line-too-long

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
from src.ope import kiv_batch

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

config_flags.DEFINE_config_dict('problem_config', utils.get_problem_config(),
                                'ConfigDict instance for problem config.')
FLAGS = flags.FLAGS


def main(_):
  problem_config = FLAGS.problem_config

  # Load the offline dataset and environment.
  dataset, dev_dataset, environment = utils.load_data_and_env(
      problem_config['task_name'], problem_config['prob_param'],
      dataset_path=FLAGS.dataset_path,
      batch_size=FLAGS.batch_size,
      max_dev_size=FLAGS.max_dev_size,
      shuffle=False,
      repeat=False)
  environment_spec = specs.make_environment_spec(environment)

  if problem_config['use_near_policy_dataset']:
    # Use a behavior policy to generate a near-policy dataset and replace
    # the pure offline dataset.
    logging.info('Using the near-policy dataset.')
    dataset, dev_dataset = utils.load_near_policy_data(
        task_name=problem_config['task_name'],
        prob_param=problem_config['prob_param'],
        policy_param=problem_config['behavior_policy_param'],
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.batch_size,
        valid_batch_size=FLAGS.batch_size,
        max_dev_size=FLAGS.max_dev_size,
        shuffle=False,
        repeat=False)

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
  value_func, instrumental_feature = kiv_batch.make_ope_networks(
      problem_config['task_name'], environment_spec,
      n_component=FLAGS.n_component, gamma=gamma)

  # Load pretrained target policy network.
  target_policy_net = utils.load_policy_net(
      task_name=problem_config['task_name'],
      params=problem_config['target_policy_param'],
      environment_spec=environment_spec,
      dataset_path=FLAGS.dataset_path)

  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')
  logger = loggers.TerminalLogger('learner')

  # The learner updates the parameters (and initializes them).
  num_batches = 0
  for _ in dataset:
    num_batches += 1
  stage1_batch = num_batches // 2
  stage2_batch = num_batches - stage1_batch
  learner = kiv_batch.KIVLearner(
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
      counter=learner_counter,
      logger=logger,
      checkpoint=False)

  eval_counter = counting.Counter(counter, 'eval')
  eval_logger = loggers.TerminalLogger('eval')

  while True:
    results = {'gamma': gamma,
               'stage1_batch': stage1_batch,
               'stage2_batch': stage2_batch,
               }
    # Include learner results in eval results for ease of analysis.
    results.update(learner.step())
    results.update(utils.ope_evaluation(
        value_func=value_func,
        policy_net=target_policy_net,
        environment=environment,
        num_init_samples=FLAGS.evaluate_init_samples,
        discount=problem_config['discount'],
        counter=eval_counter))
    eval_logger.write(results)
    if learner.state['num_steps'] >= FLAGS.max_steps:
      break


if __name__ == '__main__':
  app.run(main)
