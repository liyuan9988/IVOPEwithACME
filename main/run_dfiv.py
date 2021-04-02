# python3
# pylint: disable=bad-indentation,line-too-long

from absl import app
from absl import flags
from absl import logging

from acme import specs
from acme.utils import counting
from acme.utils import loggers

import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src import utils
from src.ope import dfiv

flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')

# Network flags.
flags.DEFINE_string('value_layer_sizes', '50,50',
                    'Value net hidden layer sizes.')
flags.DEFINE_string('instrumental_layer_sizes', '50,50',
                    'Instrumental net layer sizes.')

# Agent flags
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('value_learning_rate', 1e-4, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_learning_rate', 1e-3, 'learning rate for the instrumental_net update')
flags.DEFINE_float('stage1_reg', 1e-5, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-5, 'ridge regularizer for stage 2 regression')
flags.DEFINE_float('instrumental_reg', 1e-5, 'ridge regularizer instrumental')
flags.DEFINE_float('value_reg', 1e-5, 'ridge regularizer for value_reg')

flags.DEFINE_integer('instrumental_iter', 1, 'number of iteration for instrumental function')
flags.DEFINE_integer('value_iter', 1, 'number of iteration for value function')


flags.DEFINE_integer('max_dev_size', 10*1024, 'Maximum dev dataset size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 1000, 'Number of initial samples for evaluation.')

flags.DEFINE_integer('max_steps', 100000, 'Max number of steps.')
flags.DEFINE_float('d_tm1_weight', 0.01,  # 0.01 for cartpole, 0.03 for catch and mountain_car.
                   'Weights of terminal states.')
flags.DEFINE_boolean('include_terminal', False, 'Generate dataset with terminal absorbing state.')
flags.DEFINE_boolean('ignore_d_tm1', False, 'Always set d_tm1 = 1.0 if True.')

flags.DEFINE_boolean('learner2', False, 'Run Learner2 that learns '
                     'curr_feature - discount * next_feature in the 1st stage.')


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
        'behavior_dataset_size': 0,  # 180000
        'discount': 0.99,
    }

    # Load the offline dataset and environment.
    dataset, dev_dataset, environment = utils.load_data_and_env(
        problem_config['task_name'], problem_config['prob_param'],
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.batch_size,
        max_dev_size=FLAGS.max_dev_size)
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    value_func, instrumental_feature = dfiv.make_ope_networks(
        problem_config['task_name'], environment_spec,
        value_layer_sizes=FLAGS.value_layer_sizes,
        instrumental_layer_sizes=FLAGS.instrumental_layer_sizes)

    # Load pretrained target policy network.
    target_policy_net = utils.load_policy_net(
        task_name=problem_config['task_name'],
        params=problem_config['target_policy_param'],
        environment_spec=environment_spec,
        dataset_path=FLAGS.dataset_path)

    if problem_config['behavior_dataset_size'] > 0:
      # Use behavior policy to generate an off-policy dataset and replace
      # the pre-generated offline dataset.
      logging.warning('Ignore offline dataset')
      dataset = utils.generate_train_data(
          task_name=problem_config['task_name'],
          behavior_policy_param=problem_config['behavior_policy_param'],
          dataset_path=FLAGS.dataset_path,
          environment=environment,
          dataset_size=problem_config['behavior_dataset_size'],
          batch_size=problem_config['behavior_dataset_size'] // 2,
          shuffle=True,
          include_terminal=FLAGS.include_terminal,
          ignore_d_tm1=FLAGS.ignore_d_tm1)
      dev_dataset = None

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')
    logger = loggers.TerminalLogger('learner')

    # The learner updates the parameters (and initializes them).
    learner_cls = dfiv.DFIVLearner
    if FLAGS.learner2:
      learner_cls = dfiv.DFIV2Learner
    learner = learner_cls(
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
          eval_results = {'dev_mse': learner.cal_validation_err(dev_dataset)}
        eval_results.update(utils.ope_evaluation(
            value_func=value_func,
            policy_net=target_policy_net,
            environment=environment,
            num_init_samples=FLAGS.evaluate_init_samples,
            mse_samples=18,
            discount=problem_config['discount'],
            counter=eval_counter))
        eval_logger.write(eval_results)

      if steps >= FLAGS.max_steps:
        break


if __name__ == '__main__':
    app.run(main)
