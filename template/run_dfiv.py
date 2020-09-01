# python3

"""An example DFIV running script."""

# import functools
import operator

from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme.agents.tf import actors
# from acme.agents.tf.dqfd import bsuite_demonstrations
from acme.tf import networks as acme_nets
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
# from acme.wrappers import single_precision
# import bsuite
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from src.models.DFIV.acme import learning

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')

# Agent flags
# TODO(liyuan): Add necessary flags for DFIV method.
flags.DEFINE_float('treatment_learning_rate', 1e-3,
                   'Learning rate for treatment network.')
flags.DEFINE_float('instrumental_learning_rate', 1e-3,
                   'Learning rate for instrumental network.')
flags.DEFINE_float('policy_learning_rate', 1e-3,
                   'Learning rate for policy network.')
# flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def make_networks(action_spec: specs.DiscreteArray) -> Dict[str, snt.Module]:
  """Make networks."""

  # TODO(liyuan): make networks using sonnet and return a dictionary.
  # The policy network should be a distribution in the class of tfp.Distribution
  # e.g. tfp.distributions.Normal or tfp.distributions.TruncatedNormal.

  # return snt.Sequential([
  #     snt.Flatten(),
  #     snt.nets.MLP([64, 64, action_spec.num_values]),
  # ])

  return {'treatment_net': treatment_net,
          'instrumental_net': instrumental_net,
          'policy_net': policy_net,
          }


def _n_step_transition_from_episode(observations: types.NestedTensor,
                                    actions: tf.Tensor, rewards: tf.Tensor,
                                    discounts: tf.Tensor, n_step: int,
                                    additional_discount: float):
  """Produce Reverb-like N-step transition from a full episode.
  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.
  Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    n_step: number of steps to squash into a single transition.
    additional_discount: discount to use for TD updates.
  Returns:
    (o_t, a_t, r_t, d_t, o_tp1) tuple.
  """

  max_index = tf.shape(rewards)[0] - 1
  first = tf.random.uniform(
      shape=(), minval=0, maxval=max_index - 1, dtype=tf.int32)
  last = tf.minimum(first + n_step, max_index)

  o_t = tree.map_structure(operator.itemgetter(first), observations)
  a_t = tree.map_structure(operator.itemgetter(first), actions)
  o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

  # 0, 1, ..., n-1.
  discount_range = tf.cast(tf.range(last - first), tf.float32)
  # 1, g, ..., g^{n-1}.
  additional_discounts = tf.pow(additional_discount, discount_range)
  # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
  discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last - 1])], 0)
  # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
  discounts *= additional_discounts
  # r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
  # We have to shift rewards by one so last=max_index corresponds to transitions
  # that include the last reward.
  r_t = tf.reduce_sum(rewards[first + 1:last + 1] * discounts)

  # g^{n-1} * d_{t} * ... * d_{t+n-1}.
  d_t = discounts[-1]

  # Reverb requires every sample to be given a key and priority.
  # In the supervised learning case for BC, neither of those will be used.
  # We set the key to `0` and the priorities probabilities to `1`, but that
  # should not matter much.
  key = tf.constant(0, tf.uint64)
  probability = tf.constant(1.0, tf.float64)
  table_size = tf.constant(1, tf.int64)
  priority = tf.constant(1.0, tf.float64)
  info = reverb.SampleInfo(
      key=key,
      probability=probability,
      table_size=table_size,
      priority=priority,
  )

  return reverb.ReplaySample(info=info, data=(o_t, a_t, r_t, d_t, o_tp1))


def main(_):
  # TODO(yutian): Create environment.
  # # Create an environment and grab the spec.
  # raw_environment = bsuite.load_and_record_to_csv(
  #     bsuite_id=FLAGS.bsuite_id,
  #     results_dir=FLAGS.results_dir,
  #     overwrite=FLAGS.overwrite,
  # )
  # environment = single_precision.SinglePrecisionWrapper(raw_environment)
  # environment_spec = specs.make_environment_spec(environment)

  # TODO(yutian): Create dataset.
  # Build the dataset.
  # if hasattr(raw_environment, 'raw_env'):
  #   raw_environment = raw_environment.raw_env
  #
  # batch_dataset = bsuite_demonstrations.make_dataset(raw_environment)
  # # Combine with demonstration dataset.
  # transition = functools.partial(
  #     _n_step_transition_from_episode, n_step=1, additional_discount=1.)
  #
  # dataset = batch_dataset.map(transition)
  #
  # # Batch and prefetch.
  # dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  # Create the networks to optimize.
  networks = make_networks(environment_spec.actions)
  treatment_net = networks['treatment_net']
  instrumental_net = networks['instrumental_net']
  policy_net = networks['policy_net']

  # If the agent is non-autoregressive use epsilon=0 which will be a greedy
  # policy.
  evaluator_net = snt.Sequential([
      policy_net,
      # Sample actions.
      acme_nets.StochasticSamplingHead()
  ])

  # Ensure that we create the variables before proceeding (maybe not needed).
  tf2_utils.create_variables(policy_net, [environment_spec.observations])
  # TODO(liyuan): set the proper input spec using environment_spec.observations
  # and environment_spec.actions.
  tf2_utils.create_variables(treatment_net, [environment_spec.observations])
  tf2_utils.create_variables(instrumental_net, [environment_spec.observations,
                                                environment_spec.actions])

  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')

  # Create the actor which defines how we take actions.
  evaluator_net = actors.FeedForwardActor(evaluator_net)

  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=evaluator_net,
      counter=counter,
      logger=loggers.TerminalLogger('evaluation', time_delta=1.))

  # The learner updates the parameters (and initializes them).
  learner = learning.DFIVLearner(
      treatment_net=treatment_net,
      instrumental_net=instrumental_net,
      policy_net=policy_net,
      treatment_learning_rate=FLAGS.treatment_learning_rate,
      instrumental_learning_rate=FLAGS.instrumental_learning_rate,
      policy_learning_rate=FLAGS.policy_learning_rate,
      dataset=dataset,
      counter=learner_counter)

  # Run the environment loop.
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    learner_counter.increment(learner_steps=FLAGS.evaluate_every)
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
