import tensorflow as tf


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return -tf.math.log(1. / x - 1.)


def mse(y_true, y_pred):
    return tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=0))


def bce():
    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


def yolo_loss(num_boxes=2, ignore_threshold=.5, localization_weight=5, no_obj_weight=0.5):
    """
    Yolo Loss function constructor

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
        y_actual = tf.concat(tf.split(y_actual, num_boxes, axis=3), axis=0)  # 2 * batch x H x W x 5 + num_classes
        y_pred = tf.concat(tf.split(y_pred, num_boxes, axis=3), axis=0)  # 2 * batch x H x W x 5 + num_classes
        true_boxes, true_obj, true_classes = y_actual[..., :4], y_actual[..., 4:5], y_actual[..., 5:]
        pred_boxes, pred_obj, pred_classes = y_pred[..., :4], y_pred[..., 4:5], y_pred[..., 5:]
        true_anchors, true_sizes = true_boxes[..., :2], true_boxes[..., 2:]
        pred_anchors, pred_sizes = pred_boxes[..., :2], pred_boxes[..., 2:]
        obj_mask = true_obj[..., 0]
        _, h, w, *_ = pred_boxes.shape
        true_boxes = yolo_bbox(true_boxes, (h, w))
        pred_boxes = yolo_bbox(pred_boxes, (h, w))
        iou = bbox_iou(true_boxes, pred_boxes)
        ignore_mask = tf.subtract(tf.constant(1, shape=(h, w), dtype=tf.float32),
                                  tf.floor(tf.math.divide_no_nan(iou, tf.constant(ignore_threshold, shape=(h, w),
                                                                                  dtype=tf.float32))))
        anchor_loss = localization_weight * mse(true_anchors, pred_anchors)
        # TODO: Compute true anchor pos with respect to prior box  -->  ln(true_pos / prior_pos) = ground_truth
        shape_loss = localization_weight * mse(true_sizes, pred_sizes)
        object_loss = tf.reduce_sum(tf.multiply(bce()(true_obj, pred_obj), obj_mask))
        no_object_loss = no_obj_weight * tf.reduce_sum(
            tf.multiply(
                tf.multiply(
                    bce()(true_obj, pred_obj),
                    tf.subtract(tf.constant(1, shape=(h, w), dtype=tf.float32), obj_mask)),
                ignore_mask
            )
        )
        class_loss = tf.reduce_sum(tf.multiply(bce()(true_classes, pred_classes), obj_mask))
        # tf.print(anchor_loss, shape_loss, object_loss, no_object_loss, class_loss, output_stream=sys.stdout)
        return tf.reduce_sum([anchor_loss, shape_loss, object_loss, no_object_loss, class_loss])

    return inner


def features_to_bboxes(y, anchors, num_classes):
    """

    :param y: batch x H x W x (5 + num_classes) * num_boxes
    :param anchors: num_boxes x (w, h)
    :param num_classes:
    :return:
    """

    num_boxes = len(anchors)
    conv_shape = tf.shape(y)
    batch_size = conv_shape[0]
    out_h, out_w = conv_shape[1:3]

    y = tf.reshape(y, (batch_size, out_h, out_w, num_boxes, 5 + num_classes))

    xy, wh, o, c = tf.split(y, [2, 2, 1, num_classes], axis=-1)  # batch x H x W x box x (2, 2, 1, num_classes)

    grid = tf.tile(idx_tensor((out_h, out_w))[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, num_boxes, 1])

    xy = tf.sigmoid(xy) + grid
    wh = tf.exp(wh) * anchors

    o = tf.sigmoid(o)
    c = tf.sigmoid(c)

    return tf.concat([xy, wh, o, c], axis=-1)  # batch x H x W x box x (5 + num_classes)


def idx_tensor(shape):
    y = tf.tile(tf.range(shape[0], dtype=tf.int32)[:, tf.newaxis], [1, shape[1]])
    x = tf.tile(tf.range(shape[1], dtype=tf.int32)[tf.newaxis, :], [shape[0], 1])
    return tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)


def yolo_bbox(arr, shape):
    """
    Convert YoloV3 predictions to bounding boxes

    :param arr: N x H x W x 4 (x, y, w, h)
    :type arr: tf.Tensor
    :param shape: tuple (H, w)
    :return: N x H x W x 4 (minx, miny, maxx, maxy in range [0, 1))
    """
    im_height, im_width = shape
    x, y, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    w *= im_width
    h *= im_height
    c_idx = tf.repeat(tf.expand_dims(tf.range(im_width, dtype=tf.float32), axis=0), [im_height], axis=0)
    r_idx = tf.transpose(
        tf.repeat(tf.expand_dims(tf.range(im_height, dtype=tf.float32), axis=0), [im_width], axis=0),
        perm=[1, 0])
    minx = tf.subtract(tf.add(c_idx, x), w / 2)
    miny = tf.subtract(tf.add(r_idx, y), h / 2)
    maxx, maxy = tf.add(minx, w), tf.add(miny, h)
    res = tf.stack((minx, miny, maxx, maxy), axis=3)
    return res


def bbox_iou(b1, b2):
    # N x ___
    b_bound = tf.stack([b1, b2], axis=1)  # 2 x h x w x 4
    in_mins = tf.reduce_max(b_bound[..., :2], axis=1)  # h x w x 2
    in_maxs = tf.reduce_min(b_bound[..., 2:4], axis=1)  # h x w x 2
    in_dims = tf.clip_by_value(tf.subtract(in_maxs, in_mins), 0, 1)  # h x w x 2
    in_area = tf.multiply(in_dims[..., 0], in_dims[..., 1])  # h x w
    un_dims = tf.clip_by_value(tf.subtract(b_bound[..., 2:], b_bound[..., :2]), 0, 1)  # 2 x h x w x 2
    un_area = tf.reduce_sum(tf.multiply(un_dims[..., 0], un_dims[..., 1]), axis=1)  # h x w
    # un_area = (XOR area + 2 * AND area), OR area = un_area - AND area
    un_area = tf.subtract(un_area, in_area)  # h x w
    return tf.math.divide_no_nan(in_area, un_area)  # h x w
