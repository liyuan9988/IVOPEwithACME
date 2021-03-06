# python3
# pylint: disable=bad-indentation,line-too-long

from absl import app
from absl import flags
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

import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_PATH)

from src.load_data import load_policy_net, load_data_and_env  # noqa: E402
from src.ope.df.learner import DFLearner
from src.ope.dfiv import make_ope_networks  # noqa: E402
from src.utils import ope_evaluation, generate_train_data

# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_learning_rate', 2e-5, 'learning rate for the instrumental_net update')
flags.DEFINE_float('value_l2_reg', 1e-3, 'learning rate for the treatment_net update')
flags.DEFINE_float('instrumental_l2_reg', 1e-3, 'learning rate for the treatment_net update')
flags.DEFINE_float('stage1_reg', 1e-3, 'ridge regularizer for stage 1 regression')
flags.DEFINE_float('stage2_reg', 1e-3, 'ridge regularizer for stage 2 regression')
flags.DEFINE_integer('instrumental_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('value_iter', 10, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 10000, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 1, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 100, 'Number of initial samples for evaluation.')

FLAGS = flags.FLAGS


def main(_):
    # Load the offline dataset and environment.
    problem_config = {
        'task_name': 'bsuite_cartpole',
        'prob_param': {
            'noise_level': 0.0,
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
        'discount': 0.99,
    }
    _, _, environment = load_data_and_env(problem_config['task_name'], problem_config['prob_param'])
    environment_spec = specs.make_environment_spec(environment)

    # Create the networks to optimize.
    value_func, instrumental_feature = make_ope_networks(
        problem_config['task_name'], environment_spec)

    # Load pretrained target policy network.
    target_policy_net = load_policy_net(task_name=problem_config['task_name'],
                                        params=problem_config['target_policy_param'],
                                        environment_spec=environment_spec)

    behavior_policy_net = load_policy_net(task_name=problem_config['task_name'],
                                        params=problem_config['behavior_policy_param'],
                                        environment_spec=environment_spec)


    print("start generating transitions")
    dataset = generate_train_data(behavior_policy_net, environment, 5000)
    print("end generating transitions")
    dataset = dataset.batch(5000)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DFLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy_net=target_policy_net,
        discount=problem_config["discount"],
        value_learning_rate=FLAGS.value_learning_rate,
        instrumental_learning_rate=FLAGS.instrumental_learning_rate,
        value_l2_reg = FLAGS.value_l2_reg,
        instrumental_l2_reg = FLAGS.instrumental_l2_reg,
        stage1_reg=FLAGS.stage1_reg,
        stage2_reg=FLAGS.stage2_reg,
        instrumental_iter=FLAGS.instrumental_iter,
        value_iter=FLAGS.value_iter,
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
                       mse_samples=18,
                       discount=problem_config["discount"])


if __name__ == '__main__':
    app.run(main)
