import tensorflow as tf


@tf.function
def fit_linear(target: tf.Tensor,
               feature: tf.Tensor,
               reg: float = 0.0):
    """
    Parameters
    ----------
    target: torch.Tensor[nBatch, dim1]
    feature: torch.Tensor[nBatch, feature_dim]
    reg: float
        value of l2 regularizer
    Returns
    -------
        weight: torch.Tensor[feature_dim, dim1, dim2, ...]
    """
    nData, nDim = feature.shape
    A = tf.matmul(feature, feature, transpose_a=True)
    A = A + reg * tf.eye(nDim)
    b = tf.matmul(feature, target, transpose_a=True)
    weight = tf.linalg.solve(A, b)
    return weight


@tf.function
def linear_reg_pred(feature: tf.Tensor, weight: tf.Tensor):
    return tf.matmul(feature, weight)


@tf.function
def linear_reg_loss(target: tf.Tensor,
                    feature: tf.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return tf.norm((target - pred)) ** 2 + reg * tf.norm(weight) ** 2


@tf.function
def outer_prod(mat1: tf.Tensor, mat2: tf.Tensor):
    """
    Parameters
    ----------
    mat1: torch.Tensor[nBatch, mat1_dim1, mat1_dim2, mat1_dim3, ...]
    mat2: torch.Tensor[nBatch, mat2_dim1, mat2_dim2, mat2_dim3, ...]

    Returns
    -------
    res : torch.Tensor[nBatch, mat1_dim1, ..., mat2_dim1, ...]
    """

    mat1_shape = tuple(mat1.shape)
    mat2_shape = tuple(mat2.shape)
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = tf.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = tf.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2


def add_const_col(mat: tf.Tensor):
    """

    Parameters
    ----------
    mat : torch.Tensor[n_data, n_col]

    Returns
    -------
    res : torch.Tensor[n_data, n_col+1]
        add one column only contains 1.

    """
    n_data = mat.shape[0]
    return tf.concat([mat, tf.ones((n_data, 1))], axis=1)
