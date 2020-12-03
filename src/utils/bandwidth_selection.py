import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import sonnet as snt

def get_bsuite_median(dataset, n_action):
    data = next(iter(dataset)).data
    obs, action = data[:2]
    action_aug = tf.one_hot(action, depth=n_action)
    flat = snt.Flatten()
    inputs = tf.concat([flat(obs), action_aug], axis=1)
    arr = inputs.numpy()
    dists = cdist(arr, arr)
    return 1.0 / np.median(dists)

