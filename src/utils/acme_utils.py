import os
from typing import Tuple

from acme import specs
from acme import types
from acme.wrappers import base as wrapper_base
from acme.wrappers import single_precision
import bsuite
import dm_env
import numpy as np
import sonnet as snt
from src.utils import bsuite_offline_dataset
from src.utils import dm_control_suite
import tensorflow as tf
import tensorflow_probability as tfp
import tree

tfd = tfp.distributions


def load_offline_dm_control_dataset(
    task_name: str,
    noise_std: float,
    root_path: str,
    data_path: str,
    batch_size: int,
    valid_batch_size: int,
    num_shards: int = 1,
    num_valid_shards: int = 1,
    num_threads: int = 1,
    shuffle_buffer_size: int = 100000,
    shuffle: bool = True,
    repeat: bool = True) -> Tuple[tf.data.Dataset,
                                  tf.data.Dataset,
                                  dm_env.Environment]:
  """Reuse dm_control_suite to load offline datasets from a different path."""
  # Data file path format: {root_path}/{data_path}-?????-of-{num_shards:05d}

  # SinglePrecisionWrapper and NormilizeActionSpecWrapper are applied in
  # dm_control_suite.ControlSuite.
  task = dm_control_suite.ControlSuite(task_name=task_name)
  environment = ClippedGaussianNoisyActionWrapper(task.environment, noise_std)
  train_data_path = data_path + '_train'
  train_dataset = dm_control_suite.dataset(
      root_path=root_path,
      data_path=train_data_path,
      shapes=task.shapes,
      num_threads=num_threads,
      batch_size=batch_size,
      uint8_features=task.uint8_features,
      num_shards=num_shards,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle=shuffle,
      repeat=repeat)
  valid_data_path = data_path + '_valid'
  valid_dataset = dm_control_suite.dataset(
      root_path=root_path,
      data_path=valid_data_path,
      shapes=task.shapes,
      num_threads=num_threads,
      batch_size=valid_batch_size,
      uint8_features=task.uint8_features,
      num_shards=num_valid_shards,
      shuffle=False,
      repeat=False)
  return train_dataset, valid_dataset, environment


def load_offline_bsuite_dataset(
    bsuite_id: str,
    random_prob: float,
    path: str,
    batch_size: int,
    valid_batch_size: int,
    num_shards: int = 1,
    num_valid_shards: int = 1,
    num_threads: int = 1,
    single_precision_wrapper: bool = True,
    shuffle_buffer_size: int = 100000,
    shuffle: bool = True,
    repeat: bool = True) -> Tuple[tf.data.Dataset,
                                  tf.data.Dataset,
                                  dm_env.Environment]:
  """Load bsuite offline dataset."""
  # Data file path format: {path}-?????-of-{num_shards:05d}
  # The dataset is not deterministic and not repeated if shuffle = False.
  environment = bsuite.load_from_id(bsuite_id)
  if single_precision_wrapper:
    environment = single_precision.SinglePrecisionWrapper(environment)
  if random_prob > 0.:
    environment = RandomActionWrapper(environment, random_prob)
  params = bsuite_offline_dataset.dataset_params(environment)
  if os.path.basename(path):
    path += '_'
  train_path = path + 'train'
  train_dataset = bsuite_offline_dataset.dataset(
      path=train_path,
      num_threads=num_threads,
      batch_size=batch_size,
      num_shards=num_shards,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle=shuffle,
      repeat=repeat,
      **params)
  valid_path = path + 'valid'
  valid_dataset = bsuite_offline_dataset.dataset(
      path=valid_path,
      num_threads=num_threads,
      batch_size=valid_batch_size,
      num_shards=num_valid_shards,
      shuffle=False,
      repeat=False,
      **params)
  return train_dataset, valid_dataset, environment


class ClippedGaussianNoisyActionWrapper(wrapper_base.EnvironmentWrapper):
  """Environment wrapper to add Gaussian action noise and clip to spec."""

  def __init__(self, environment: dm_env.Environment, noise_std: float = 0.):
    super().__init__(environment)
    self._noise_std = noise_std
    self._action_spec = self._environment.action_spec()

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    def _add_noise(value, spec: specs.BoundedArray):
      """Add clipped Gaussian noise to one action array."""
      if value is not None:
        value = np.array(value, copy=False)
        value += np.random.normal(scale=self._noise_std,
                                  size=value.shape)
        value = np.clip(value, spec.minimum, spec.maximum)
      return value

    action = tree.map_structure(_add_noise, action, self._action_spec)
    return self._environment.step(action)


class RandomActionWrapper(wrapper_base.EnvironmentWrapper):
  """Environment wrapper to play a random discrete action with a probablity."""

  def __init__(self, environment: dm_env.Environment, random_prob: float = 0.):
    super().__init__(environment)
    self._random_prob = random_prob
    if not 0 <= random_prob <= 1:
      raise ValueError(f'random_prob ({random_prob}) must be within [0, 1]')

    self._action_spec = self._environment.action_spec()
    if not all(map(lambda spec: isinstance(spec, specs.DiscreteArray),
                   tree.flatten(self._action_spec))):
      raise ValueError('RandomActionWrapper requires the action_spec to be a '
                       'DiscreteArray or a nested DiscreteArray.')

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    if np.random.uniform() < self._random_prob:
      # Replace with a random action.
      action = tree.map_structure(
          lambda spec: np.random.randint(spec.num_values, dtype=spec.dtype),
          self._action_spec)
    return self._environment.step(action)


class GaussianNoise(snt.Module):
  """Sonnet module for adding Gaussian noise to each output."""

  def __init__(self, stddev: float, name: str = 'gaussian_noise'):
    super().__init__(name=name)
    self._noise = tfd.Normal(loc=0., scale=stddev)

  def __call__(self, inputs: types.NestedTensor) -> types.NestedTensor:
    def add_noise(tensor: tf.Tensor):
      output = tensor + self._noise.sample(tensor.shape)
      return output

    return tree.map_structure(add_noise, inputs)
