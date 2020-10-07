import tensorflow as tf

from utils.ai import idx_tensor


def mse(y_true, y_pred, mask=None):
    result = tf.square(tf.subtract(y_true, y_pred))
    if mask is not None: result *= tf.tile(mask[..., tf.newaxis], [1, 1, 1, 1, 2])
    return tf.reduce_sum(tf.reduce_mean(result, axis=0))


def bce():
    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


def yolo_loss(anchors, out_grids, num_boxes=3, ignore_threshold=.5, localization_weight=5, no_obj_weight=0.5):
    """
    Yolo Loss function constructor

    :param anchors:
    :param out_grids:
    :param num_boxes:
    :param ignore_threshold:
    :param localization_weight:
    :param no_obj_weight:
    :return:
    """
    def inner(y_actual, y_pred):
        """
        YoloV3 Loss function

        :param y_actual: batch x H x W x (5 + num_classes) * num_boxes
        :type y_actual: tf.Tensor
        :param y_pred: batch x H x W x (5 + num_classes) * num_boxes
        :type y_pred: tf.Tensor
        :return: scalar value
        """
        batch_size, grid_h, grid_w, params = y_pred.shape
        for i, (h, w) in enumerate(out_grids):
            if grid_h == h and grid_w == w:
                scale_anchors = tf.constant(anchors[-i - 1])
        y_actual = tf.reshape(y_actual, (-1, grid_h, grid_w, num_boxes, params//num_boxes))  # B, H, W, A, 5 + num_classes
        y_pred = tf.reshape(y_pred, (-1, grid_h, grid_w, num_boxes, params//num_boxes))  # B, H, W, A, 5 + num_classes
        true_boxes, true_obj, true_classes = y_actual[..., :4], y_actual[..., 4:5], y_actual[..., 5:]  # (B, H, W, A, _): xywh, o, c
        pred_boxes, pred_obj, pred_classes = y_pred[..., :4], y_pred[..., 4:5], y_pred[..., 5:]  # (B, H, W, A, _): xywh, o, c
        true_anchors, true_sizes = true_boxes[..., :2], true_boxes[..., 2:]  # (B, H, W, A, _): xy, wh
        pred_anchors, pred_sizes = pred_boxes[..., :2], pred_boxes[..., 2:]  # (B, H, W, A, _): xy, wh
        obj_mask = true_obj[..., 0]  # B, H, W, A
        iou = bbox_iou(t_to_b(true_boxes, scale_anchors), t_to_b(pred_boxes, scale_anchors))  # B, H, W, A
        ignore_mask = tf.where(iou < ignore_threshold, tf.ones(tf.shape(iou)), tf.zeros(tf.shape(iou)))  # B, H, W, A
        _, h, w, *_ = pred_boxes.shape
        xy_loss = localization_weight * mse(true_anchors, pred_anchors, obj_mask)
        shape_loss = localization_weight * mse(true_sizes, pred_sizes, obj_mask)
        obj_loss = tf.reduce_sum(tf.reduce_mean(bce()(true_obj, tf.sigmoid(pred_obj)) * obj_mask, axis=0))
        noobj_loss = no_obj_weight * tf.reduce_sum(
            tf.reduce_mean(bce()(true_obj, tf.sigmoid(pred_obj)) * (1 - obj_mask) * ignore_mask, axis=0)
        )
        class_loss = tf.reduce_sum(tf.reduce_mean(bce()(true_classes, tf.sigmoid(pred_classes)) * obj_mask, axis=0))
        return tf.reduce_sum([xy_loss, shape_loss, obj_loss, noobj_loss, class_loss])

    return inner


def t_to_b(y, anchors):
    """

    :param y: batch, H, W, A, 4
    :param anchors: A, (w, h)
    :return: B, H, W, A, 4
    """

    num_boxes = anchors.shape[0]
    conv_shape = tf.shape(y)
    batch_size = conv_shape[0]
    out_h, out_w = conv_shape[1], conv_shape[2]

    xy, wh = y[..., :2], y[..., 2:4]  # (B, H, W, A, _): xy, wh

    grid = tf.tile(idx_tensor((out_h, out_w))[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, num_boxes, 1])

    xy = tf.sigmoid(xy) + grid  # B, H, W, A, xy
    wh = tf.exp(wh) * anchors  # B, H, W, A, wh

    return tf.concat([xy, wh], axis=-1)  # B, H, W, A, 4


def yolo_bbox(arr, shape, anchors):
    """
    Convert YoloV3 predictions to bounding boxes

    :param arr: N x H x W x num_boxes x 4 (x, y, w, h)
    :type arr: tf.Tensor
    :param shape: tuple (H, w)
    :param anchors: num_boxes x 2
    :return: N x H x W x 4 (minx, miny, maxx, maxy in range [0, 1))
    """
    im_height, im_width = shape
    x, y, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    w *= im_width
    h *= im_height
    c_idx = tf.expand_dims(tf.repeat(tf.expand_dims(tf.range(im_width, dtype=tf.float32), axis=0), [im_height], axis=0), axis=-1)
    r_idx = tf.expand_dims(tf.transpose(
        tf.repeat(tf.expand_dims(tf.range(im_height, dtype=tf.float32), axis=0), [im_width], axis=0),
        perm=[1, 0]), axis=-1)
    minx = tf.subtract(tf.add(c_idx, x), w / 2)
    miny = tf.subtract(tf.add(r_idx, y), h / 2)
    maxx, maxy = tf.add(minx, w), tf.add(miny, h)
    res = tf.stack((minx, miny, maxx, maxy), axis=3)
    return res


def bbox_iou(b1, b2):
    """

    :param b1: B, H, W, A, xywh
    :param b2: B, H, W, A, xywh
    :return: B, H, W, A
    """
    xy1, wh1 = b1[..., :2], b1[..., 2:4]
    xy2, wh2 = b2[..., :2], b2[..., 2:4]
    xy1 -= wh1/2
    xy2 -= wh2/2
    mins = tf.stack((xy1, xy2), axis=0)  # 2, B, H, W, A, 2
    maxs = tf.stack((xy1 + wh1, xy2 + wh2), axis=0)  # 2, B, H, W, A, 2
    in_mins = tf.reduce_max(mins, axis=0)  # B, H, W, A, 2
    in_maxs = tf.reduce_min(maxs, axis=0)  # B, H, W, A, 2
    in_dims = in_maxs - in_mins  # B, H, W, A, 2
    in_area = tf.reduce_prod(in_dims, axis=-1)  # B, H, W, A
    un_dims = tf.stack((wh1, wh2), axis=0)  # 2, B, H, W, A, 2
    un_area = tf.reduce_prod(un_dims, axis=-1)  # 2, B, H, W, A
    un_area = tf.reduce_sum(un_area, axis=0)  # B, H, W, A
    un_area = un_area - in_area  # B, H, W, A
    return tf.math.divide_no_nan(in_area, un_area)  # B, H, W, A
