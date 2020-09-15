# python3
# pylint: disable=bad-indentation,line-too-long

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

from src.load_data import load_policy_net, load_data_and_env
from src.ope.deepiv import DeepIVLearner, make_ope_networks  # noqa: E402
from src.utils import ope_evaluation


# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('density_learning_rate', 2e-5, 'learning rate for the mixture density net update')
flags.DEFINE_integer('density_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('n_sampling', 10, 'number of samples generated in stage 2')
flags.DEFINE_integer('value_iter', 10, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 2000, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 10, 'Evaluation period.')
flags.DEFINE_integer('evaluate_init_samples', 100, 'Number of initial samples for evaluation.')

FLAGS = flags.FLAGS


def main(_):
    problem_config = {
        'task_name': 'dm_control_cartpole_swingup',
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
    dataset, environment = load_data_and_env(problem_config['task_name'], problem_config['prob_param'])
    environment_spec = specs.make_environment_spec(environment)


    dataset = dataset.batch(FLAGS.batch_size)

    # Create the networks to optimize.
    value_func, mixture_density = make_ope_networks(problem_config['task_name'], environment_spec)

    # Load pretrained target policy network.
    policy_net = load_policy_net(task_name=problem_config['task_name'],
                                 params=problem_config['policy_param'],
                                 environment_spec=environment_spec)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DeepIVLearner(
        value_func=value_func,
        mixture_density=mixture_density,
        policy_net=policy_net,
        discount=problem_config["discount"],
        value_learning_rate=FLAGS.value_learning_rate,
        density_learning_rate=FLAGS.density_learning_rate,
        n_sampling=FLAGS.n_sampling,
        density_iter=FLAGS.density_iter,
        value_iter=FLAGS.value_iter,
        dataset=dataset,
        counter=learner_counter)

    eval_logger = loggers.TerminalLogger('eval')

    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        ope_evaluation(value_func=value_func,
                       policy_net=policy_net,
                       environment=environment,
                       logger=eval_logger,
                       num_init_samples=FLAGS.evaluate_init_samples,
                       mse_samples=1000,
                       discount=problem_config["discount"])


if __name__ == '__main__':
    app.run(main)
