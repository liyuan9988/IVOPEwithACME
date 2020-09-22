"""Offline bsuite datasets."""

import functools
from typing import Dict, Tuple

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree


def _parse_seq_tf_example(example, shapes, dtypes):
    """Parse tf.Example containing one or two episode steps."""

    def to_feature(shape, dtype):
        if np.issubdtype(dtype, np.floating):
            return tf.io.FixedLenSequenceFeature(
                shape=shape, dtype=tf.float32, allow_missing=True)
        elif dtype == np.bool or np.issubdtype(dtype, np.integer):
            return tf.io.FixedLenSequenceFeature(
                shape=shape, dtype=tf.int64, allow_missing=True)
        else:
            raise ValueError(f'Unsupported type {dtype} to convert from TF Example.')

    feature_map = {}
    for k, v in shapes.items():
        feature_map[k] = to_feature(v, dtypes[k])

    parsed = tf.io.parse_single_example(example, features=feature_map)

    restructured = {}
    for k, v in parsed.items():
        dtype = tf.as_dtype(dtypes[k])
        if v.dtype == dtype:
            restructured[k] = parsed[k]
        else:
            restructured[k] = tf.cast(parsed[k], dtype)

    return restructured


def _build_sequence_example(sequences):
    """Convert raw sequences into a Reverb sequence sample."""
    o = sequences['observation']
    a = sequences['action']
    r = sequences['reward']
    p = sequences['discount']

    info = reverb.SampleInfo(key=tf.constant(0, tf.uint64),
                             probability=tf.constant(1.0, tf.float64),
                             table_size=tf.constant(0, tf.int64),
                             priority=tf.constant(1.0, tf.float64))
    return reverb.ReplaySample(info=info, data=(o, a, r, p))


def _build_sarsa_example(sequences):
    """Convert raw sequences into a Reverb n-step SARSA sample."""

    o_tm1 = tree.map_structure(lambda t: t[0], sequences['observation'])
    o_t = tree.map_structure(lambda t: t[1], sequences['observation'])
    a_tm1 = tree.map_structure(lambda t: t[0], sequences['action'])
    a_t = tree.map_structure(lambda t: t[1], sequences['action'])
    r_t = tree.map_structure(lambda t: t[0], sequences['reward'])
    p_t = tree.map_structure(
        lambda d, st: d[0] * tf.cast(st[1] != dm_env.StepType.LAST, d.dtype),
        sequences['discount'], sequences['step_type'])

    info = reverb.SampleInfo(key=tf.constant(0, tf.uint64),
                             probability=tf.constant(1.0, tf.float64),
                             table_size=tf.constant(0, tf.int64),
                             priority=tf.constant(1.0, tf.float64))
    return reverb.ReplaySample(info=info, data=(o_tm1, a_tm1, r_t, p_t, o_t, a_t))


def dataset_params(env):
    """Return shapes and dtypes parameters for bsuite offline dataset."""
    shapes = {
        'observation': env.observation_spec().shape,
        'action': env.action_spec().shape,
        'discount': env.discount_spec().shape,
        'reward': env.reward_spec().shape,
        'episodic_reward': env.reward_spec().shape,
        'step_type': (),
    }

    dtypes = {
        'observation': env.observation_spec().dtype,
        'action': env.action_spec().dtype,
        'discount': env.discount_spec().dtype,
        'reward': env.reward_spec().dtype,
        'episodic_reward': env.reward_spec().dtype,
        'step_type': np.int64,
    }

    return {'shapes': shapes, 'dtypes': dtypes}


def dataset(path: str,
            shapes: Dict[str, Tuple[int]],
            dtypes: Dict[str, type],  # pylint:disable=g-bare-generic
            num_threads: int,
            batch_size: int,
            num_shards: int,
            shuffle_buffer_size: int = 100000,
            shuffle: bool = True,
            sarsa: bool = True) -> tf.data.Dataset:
    """Create tf dataset for training."""

    filenames = [f'{path}-{i:05d}-of-{num_shards:05d}' for i in range(num_shards)]
    file_ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
      file_ds = file_ds.repeat().shuffle(num_shards)

    example_ds = file_ds.interleave(
        functools.partial(tf.data.TFRecordDataset, compression_type='GZIP'),
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=5)
    if shuffle:
      example_ds = example_ds.shuffle(shuffle_buffer_size)

    def map_func(example):
        example = _parse_seq_tf_example(example, shapes, dtypes)
        return example

    example_ds = example_ds.map(map_func, num_parallel_calls=num_threads)
    if shuffle:
      example_ds = example_ds.repeat().shuffle(batch_size * 10)

    if sarsa:
        example_ds = example_ds.map(
            _build_sarsa_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        example_ds.batch(batch_size)
    else:
        example_ds = example_ds.padded_batch(
            batch_size, padded_shapes=shapes, drop_remainder=True)

        example_ds = example_ds.map(
            _build_sequence_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    example_ds = example_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return example_ds
