# python3

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


# Agent flags
flags.DEFINE_float('value_learning_rate', 2e-5, 'learning rate for the treatment_net update')
flags.DEFINE_float('density_learning_rate', 2e-5, 'learning rate for the mixture density net update')
flags.DEFINE_integer('density_iter', 20, 'number of iteration for instrumental function')
flags.DEFINE_integer('n_sampling', 10, 'number of samples generated in stage 2')
flags.DEFINE_integer('value_iter', 10, 'number of iteration for value function')

flags.DEFINE_integer('batch_size', 2000, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 10, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS

def eval_model(test_data, value_func, policy):
    current_obs, action, reward, discount, next_obs, _ = test_data.data
    next_action = policy(next_obs)
    target = tf.expand_dims(reward, axis=1) + tf.expand_dims(discount, axis=1) * value_func(next_obs, next_action)
    return tf.norm(target - value_func(current_obs, action)) ** 2

def main(_):
    problem_config = {
        "task_name": "dm_control_cartpole_swingup",
        "prob_param": {
            "noise_level": 0.0,
            "run_id": 0
        },
        "policy_param": {
            "noise_level": 0.0,
            "run_id": 1
        },
        'discount': 0.99,
    }

    # Load the offline dataset and environment.
    full_dataset, environment = load_data_and_env(problem_config["task_name"], problem_config["prob_param"])
    environment_spec = specs.make_environment_spec(environment)

    full_dataset = full_dataset.shuffle(10000)
    test_data = full_dataset.take(1000)
    train_data = full_dataset.skip(1000)
    train_data = train_data.shuffle(20000)

    test_data = test_data.batch(1000)
    test_data = next(iter(test_data))

    dataset = train_data.batch(FLAGS.batch_size)



    # Create the networks to optimize.
    value_func, mixture_density = make_ope_networks(problem_config["task_name"], environment_spec)

    # Load pretrained target policy network.
    policy_net = load_policy_net(task_name=problem_config["task_name"],
                                 params=problem_config["policy_param"],
                                 environment_spec=environment_spec)

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # The learner updates the parameters (and initializes them).
    learner = DeepIVLearner(
        value_func=value_func,
        mixture_density=mixture_density,
        policy_net=policy_net,
        value_learning_rate=FLAGS.value_learning_rate,
        density_learning_rate=FLAGS.density_learning_rate,
        n_sampling=FLAGS.n_sampling,
        density_iter=FLAGS.density_iter,
        value_iter=FLAGS.value_iter,
        dataset=dataset,
        counter=learner_counter)

    while True:
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        print(eval_model(test_data, value_func, policy_net))


if __name__ == '__main__':
    app.run(main)
