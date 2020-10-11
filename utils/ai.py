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


def yolo_to_bbox(y, target_shape, anchors, obj_threshold=0.5, class_threshold=0.5):
    """

    :param y: B, H, W, A * (5 + Cl)
    :param target_shape: H, W, Ch
    :param anchors: A, 2
    :param obj_threshold: threshold above which a prediction is considered an object
    :param class_threshold: threshold above which an object is considered a class member
    :return: N, (4 + C) array of objects in form (minx, miny, maxx, maxy, classes)
    """

    _, gh, gw, params = y.shape
    ih, iw, _ = target_shape
    y = y.reshape((1, gh, gw, len(anchors), -1))  # B, H, W, A, 5 + C
    xy, wh, o, c = np.split(y, (2, 4, 5), axis=-1)  # (B, H, W, A, _): (2, 2, 1, C)
    grid = np.indices((gh, gw))[::-1].transpose((1, 2, 0)).reshape((gh, gw, 1, 2))  # 2, H, W

    # t* -> b*
    xy = (sigmoid(xy) + grid) / np.array([gw, gh])  # B, H, W, A, 2
    wh = np.exp(wh) * anchors  # B, H, W, A, 2
    o = sigmoid(o)  # B, H, W, A, 1
    c = sigmoid(c)  # B, H, W, A, C

    # b* -> image bounds
    mins = (xy - wh / 2) * np.array([iw, ih])  # B, H, W, A, 2
    maxs = (xy + wh / 2) * np.array([iw, ih])  # B, H, W, A, 2

    # Only high probability objects
    B, H, W, A, _ = (o > obj_threshold).nonzero()
    mins = mins[B, H, W, A]  # N, 2
    maxs = maxs[B, H, W, A]  # N, 2
    c = c[B, H, W, A]  # N, C

    # Ensure bounds are within image
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, [iw-1, ih-1])

    # Discretize classes
    c[c > class_threshold] = 1
    c[c < class_threshold] = 0

    return np.concatenate((mins, maxs, c), axis=-1).astype(np.uint32)  # N, (2 + 2 + C)
