import tensorflow as tf
from acme import types
import tree
import operator
import reverb
import functools
from typing import Tuple
from acme.agents.tf.dqfd import bsuite_demonstrations
import bsuite
from acme.wrappers import single_precision
import bsuite
from acme import specs
import dm_env


def generate_dataset(FLAGS) -> Tuple[tf.data.Dataset, dm_env.Environment]:
    # Create an environment and grab the spec.
    raw_environment = bsuite.load_and_record_to_csv(
        bsuite_id=FLAGS.bsuite_id,
        results_dir=FLAGS.results_dir,
        overwrite=FLAGS.overwrite,
    )
    environment = single_precision.SinglePrecisionWrapper(raw_environment)

    # Build demonstration dataset.
    if hasattr(raw_environment, 'raw_env'):
        raw_environment = raw_environment.raw_env
    batch_dataset = bsuite_demonstrations.make_dataset(raw_environment)
    # Combine with demonstration dataset.
    transition = functools.partial(
        n_step_transition_from_episode, n_step=1, additional_discount=1.)

    dataset = batch_dataset.map(transition)

    # Batch and prefetch.
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, environment


def n_step_transition_from_episode(observations: types.NestedTensor,
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
