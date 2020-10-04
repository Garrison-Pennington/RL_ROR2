from os import path
from functools import reduce

import tensorflow as tf
import numpy as np


def index_model_layers():
    with open(path.expanduser("~/model_summary.txt"), "r+") as f:
        lines = f.readlines()
    idx = 0
    for i in range(len(lines[4:])):
        if lines[i + 4][0] not in " _=":
            lines[i + 4] = f"{idx}: {lines[i+4]}"
            idx += 1
    with open(path.expanduser("~/model_summary.txt"), "w+") as f:
        f.writelines(lines)


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return -tf.math.log(tf.math.divide_no_nan(1., x - 1.))


def idx_tensor(shape):
    y = tf.tile(tf.range(shape[0], dtype=tf.float32)[:, tf.newaxis], [1, shape[1]])
    x = tf.tile(tf.range(shape[1], dtype=tf.float32)[tf.newaxis, :], [shape[0], 1])
    return tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))
