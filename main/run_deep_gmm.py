# python3
# pylint: disable=bad-indentation,line-too-long
"""Run GMM for OPE experiments.

The type of algorithm is determined by config.learner_class.
"""

import copy
import functools

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme.tf import networks as acme_nets
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import ml_collections as collections
from ml_collections.config_flags import config_flags
import sonnet as snt
import tensorflow.compat.v2 as tf

import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from src import utils
from src.ope import deep_gmm

flags.DEFINE_string(
    'dataset_path',
    str(ROOT_PATH.joinpath('offline_dataset').joinpath('stochastic')),
    'Path to offline dataset directory.')


def get_config():
  """Algorithm config."""
  config = collections.ConfigDict()

  config.problem_config = collections.ConfigDict(dict(
      task_name='bsuite_cartpole',
      prob_param=dict(
          noise_level=0.2,  # Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
          run_id=0),
      target_policy_param=dict(
          env_noise_level=0.2,  # Has to be the same as
                                # prob_param.env_noise_level.
          policy_noise_level=0.1,
          run_id=1),
      behavior_policy_param=dict(
          env_noise_level=0.2,
          policy_noise_level=0.3,
          run_id=1),
      behavior_dataset_size=0,  # 180000,
      discount=0.99))

  config.agent_config = collections.ConfigDict(dict(
      batch_size=1024,
      evaluate_every=100,  # Evaluation period.
      evaluate_init_samples=1000,  # Number of initial samples for evaluation.
      compute_dev_every=10000,  # Period to compute dev set TD error and f for
                                # hyper-parameter selection, if not None.
      max_dev_size=10*1024,  # Take at most max_dev_size data items for dev set.
      max_steps=200000,
      ))

  config.network_config = collections.ConfigDict(dict(
      critic_layer_sizes='50,50',  # Critic net hidden layer sizes.
      f_layer_sizes='50,50',  # Adversarial net hidden layer sizes.
      ))

  config.learner_class = 'DeepGMMLearner'
  config.learner_config = collections.ConfigDict(dict(
      DeepGMMLearner=collections.ConfigDict(dict(
          tilde_critic_update_period=1,
          critic_lr=3e-4,
          critic_beta1=0.0,
          critic_beta2=0.01,
          f_lr=None,
          f_lr_multiplier=1.0,
          f_beta1=0.0,
          f_beta2=0.01,
          clipping=True,
          clipping_action=False,
          bre_check_period=300,  # Bellman residual error check.
          bre_check_num_actions=20,  # Number of sampled actions.
          )),
      AdversarialSEMLearner=collections.ConfigDict(dict(
          critic_lr=1e-4,
          critic_beta1=0.,
          critic_beta2=0.01,
          f_lr=None,
          f_lr_multiplier=1.0,
          f_beta1=0.,
          f_beta2=0.01,
          critic_regularizer=0.5,
          f_regularizer=0.5,
          critic_l2_regularizer=1e-4,
          f_l2_regularizer=1e-4,
          clipping=True,
          clipping_action=False,
          bre_check_period=300,  # Bellman residual error check.
          bre_check_num_actions=20,  # Number of sampled actions.
          )),
      AGMMLearner=collections.ConfigDict(dict(
          critic_lr=1e-4,
          critic_beta1=0.,
          critic_beta2=0.01,
          f_lr=None,
          f_lr_multiplier=1.0,
          f_beta1=0.,
          f_beta2=0.01,
          f_regularizer=1.0,
          critic_l2_regularizer=1e-4,
          f_l2_regularizer=1e-4,
          clipping=True,
          clipping_action=False,
          bre_check_period=300,  # Bellman residual error check.
          bre_check_num_actions=20,  # Number of sampled actions.
          )),
      ))
  return config


config_flags.DEFINE_config_dict('config', get_config(), 'ConfigDict instance.')
FLAGS = flags.FLAGS


def make_networks(task_name, action_spec, **net_params):
  """Make networks."""
  def make_bsuite_networks(
      action_spec: specs.DiscreteArray,
      critic_layer_sizes: str = '50,50',
      f_layer_sizes: str = '50,50',
      ):
    critic_layer_sizes = list(map(int, critic_layer_sizes.split(',')))
    f_layer_sizes = list(map(int, f_layer_sizes.split(',')))
    action_network = functools.partial(tf.one_hot, depth=action_spec.num_values)
    critic_network = snt.Sequential([
        acme_nets.CriticMultiplexer(action_network=action_network),
        snt.nets.MLP(critic_layer_sizes, activate_final=True),
        snt.Linear(1)])
    f_network = snt.Sequential([
        acme_nets.CriticMultiplexer(action_network=action_network),
        snt.nets.MLP(f_layer_sizes, activate_final=True),
        snt.Linear(1)])
    return {'critic': critic_network,
            'f': f_network}

  def make_dm_control_networks(
      critic_layer_sizes: str = '512,512,256',
      f_layer_sizes: str = '512,512,256'):
    critic_layer_sizes = list(map(int, critic_layer_sizes.split(',')))
    f_layer_sizes = list(map(int, f_layer_sizes.split(',')))
    critic_network = snt.Sequential([
        acme_nets.CriticMultiplexer(),
        acme_nets.LayerNormMLP(critic_layer_sizes, activate_final=True),
        snt.Linear(1)])
    f_network = snt.Sequential([
        acme_nets.CriticMultiplexer(),
        acme_nets.LayerNormMLP(f_layer_sizes, activate_final=True),
        snt.Linear(1)])
    return {'critic': critic_network,
            'f': f_network}

  if task_name.startswith('bsuite'):
    return make_bsuite_networks(
        action_spec=action_spec,
        critic_layer_sizes=net_params['critic_layer_sizes'],
        f_layer_sizes=net_params['f_layer_sizes'])
  elif task_name.startswith('dm_control'):
    return make_dm_control_networks(
        critic_layer_sizes=net_params['critic_layer_sizes'],
        f_layer_sizes=net_params['f_layer_sizes'])
  else:
    raise ValueError(f'Unsupported task_name {task_name}')


def main(_):
  config = FLAGS.config
  problem_config = config.problem_config
  agent_config = config.agent_config
  network_config = config.network_config
  learner_config = config.learner_config[config.learner_class]

  # Load the offline dataset and environment.
  dataset, dev_dataset, environment = utils.load_data_and_env(
      problem_config.task_name, problem_config.prob_param,
      dataset_path=FLAGS.dataset_path,
      batch_size=agent_config.batch_size,
      max_dev_size=agent_config.max_dev_size)
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  online_networks = make_networks(task_name=problem_config.task_name,
                                  action_spec=environment_spec.actions,
                                  **network_config)
  critic_network = online_networks['critic']
  f_network = online_networks['f']
  if config.learner_class == 'DeepGMMLearner':
    tilde_networks = copy.deepcopy(online_networks)
    tilde_critic_network = tilde_networks['critic']
  else:
    tilde_critic_network = None

  # Load pretrained target policy network.
  target_policy_network = utils.load_policy_net(
      task_name=problem_config.task_name,
      params=problem_config.target_policy_param,
      environment_spec=environment_spec,
      dataset_path=FLAGS.dataset_path)

  if problem_config.behavior_dataset_size > 0:
    # Use behavior policy to generate an off-policy dataset and replace
    # the pre-generated offline dataset.
    logging.warning('Ignore offline dataset')
    dataset = utils.generate_train_data(
        task_name=problem_config.task_name,
        behavior_policy_param=problem_config.behavior_policy_param,
        dataset_path=FLAGS.dataset_path,
        environment=environment,
        dataset_size=problem_config.behavior_dataset_size,
        batch_size=agent_config.batch_size,
        shuffle=True)
    dev_dataset = None
    if agent_config.compute_dev_every is not None:
      raise ValueError('generate_train_data does not support computing'
                       'hyper-parameter selection.')

  # Create variables.
  action_spec = environment_spec.actions
  obs_spec = environment_spec.observations
  tf2_utils.create_variables(critic_network, [obs_spec, action_spec])
  tf2_utils.create_variables(f_network, [obs_spec, action_spec])
  if tilde_critic_network is not None:
    tf2_utils.create_variables(tilde_critic_network, [obs_spec, action_spec])

  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')
  logger = loggers.TerminalLogger('learner')

  if agent_config.compute_dev_every is not None:
    dev_logger = loggers.TerminalLogger('hyper')
  else:
    dev_dataset = None
    dev_logger = None

  # The learner updates the parameters (and initializes them).
  learner_kwargs = dict(
      policy_network=target_policy_network,
      critic_network=critic_network,
      f_network=f_network,
      discount=problem_config.discount,
      dataset=dataset,
      dev_dataset=dev_dataset,
      counter=learner_counter,
      logger=logger)
  if tilde_critic_network is not None:
    learner_kwargs.update(dict(tilde_critic_network=tilde_critic_network))
  learner_kwargs.update(learner_config)
  learner = getattr(deep_gmm, config.learner_class)(**learner_kwargs)

  eval_logger = loggers.TerminalLogger('eval')
  eval_counter = counting.Counter(counter, 'eval')

  while True:
    learner.step()
    steps = learner.num_steps()

    if steps % agent_config.evaluate_every == 0:
      utils.ope_evaluation(
          value_func=critic_network,
          policy_net=target_policy_network,
          environment=environment,
          logger=eval_logger,
          num_init_samples=agent_config.evaluate_init_samples,
          discount=problem_config.discount,
          counter=eval_counter)

    if (agent_config.compute_dev_every is not None and
        steps % agent_config.compute_dev_every == 0):
      dev_results = learner.dev_td_error_and_f_values()
      dev_results.update(utils.ope_evaluation(
          value_func=critic_network,
          policy_net=target_policy_network,
          environment=environment,
          num_init_samples=agent_config.evaluate_init_samples,
          mse_samples=18,
          discount=problem_config.discount))
      dev_results.update(learner_counter.get_counts())
      dev_results.update(eval_counter.get_counts())
      if dev_logger is not None:
        # TODO(yutianc): write full fs and td_errors array to disk for hyper
        # parameter selection.
        dev_logger.write(dev_results)

    if (agent_config.max_steps is not None and
        steps >= agent_config.max_steps):
      break


if __name__ == '__main__':
    app.run(main)
