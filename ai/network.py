from functools import wraps

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers

from utils.ai import compose, index_model_layers


@wraps(layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': regularizers.l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return layers.Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = layers.Add()([x, y])
    return x


def darknet_body(x, num_resblocks, out_filters):
    """

    :param x: input of shape batch x H x W x ch
    :return:
    """
    x = DarknetConv2D_BN_Leaky(out_filters//(2**num_resblocks), (3, 3))(x)
    num_blocks = [1, 2] + [8 for _ in range(num_resblocks - 3)] + [4]
    for i in range(1, num_resblocks + 1):
        num_filters = out_filters//(2**(num_resblocks-i))
        x = resblock_body(x, num_filters, num_blocks[i-1])
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def output_block(x, otcc, num_filters, out_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        layers.UpSampling2D(2))(x)
    top_pad = 0 if otcc.shape[1] == x.shape[1] else 1
    left_pad = 0 if otcc.shape[2] == x.shape[2] else 1
    x = layers.ZeroPadding2D(padding=((top_pad, 0), (left_pad, 0)))(x)
    x = layers.Concatenate()([x, otcc])
    return make_last_layers(x, num_filters, out_filters)


class YOLOv3:
    """
    You Only Look Once object detection network


    """

    def __init__(self, input_shape, num_classes, num_boxes, num_scales=4, extra_ds=0):
        """

        :param self:
        :param input_shape: H x W x ch
        :return: S x S x (num_boxes * (num_classes + 4 offsets + 1 confidence))
        """
        inputs = tf.keras.Input(input_shape)
        model = models.Model(inputs=inputs, outputs=darknet_body(inputs, num_scales + 2, 1024))
        x, y1 = make_last_layers(model.output, 1024, num_boxes*(num_classes + 5))
        outputs = [y1]
        for s in range(num_scales - 1)[::-1]:
            x, y = output_block(x, model.layers[60 * s + 92].output, 64 * (2 ** s), num_boxes*(num_classes + 5))
            outputs.append(y)
        if extra_ds:
            outputs = outputs[:-extra_ds]
        model = models.Model(inputs, outputs)
        self.model = model
        self.save_model_summary()

    def save_model_summary(self):
        from os import path
        with open(path.expanduser("~/model_summary.txt"), "w+") as f:
            f.write("")
        with open(path.expanduser("~/model_summary.txt"), "a+") as f:
            self.model.summary(print_fn=(lambda s: f.write(f"{s}\n")))
        index_model_layers()

    def compile_model(self, **kwargs):
        self.model.compile(**kwargs)
        return self.model
