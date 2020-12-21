# pylint: disable=bad-indentation,missing-function-docstring
import functools
from acme.tf import networks
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import sonnet as snt


def get_bsuite_median(environment_spec, dataset):
    data = next(iter(dataset)).data
    obs, action = data[:2]

    action_network = functools.partial(
        tf.one_hot, depth=environment_spec.actions.num_values)
    net = networks.CriticMultiplexer(action_network=action_network)
    inputs = net(obs, action)

    arr = inputs.numpy()
    dists = cdist(arr, arr, "sqeuclidean")
    return 1.0 / np.median(dists)


def get_dm_control_median(dataset):
    data = next(iter(dataset)).data
    obs, action = data[:2]

    net = networks.CriticMultiplexer()
    inputs = net(obs, action)

    arr = inputs.numpy()
    dists = cdist(arr, arr, "sqeuclidean")
    return 1.0 / np.median(dists)


def get_median(task_id, environment_spec, dataset):
    if task_id.startswith("dm_control"):
        return get_dm_control_median(dataset)
    elif task_id.startswith("bsuite"):
        return get_bsuite_median(environment_spec, dataset)
    else:
        raise ValueError(f"task id {task_id} not known")
