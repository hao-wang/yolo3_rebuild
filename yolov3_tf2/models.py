from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Concatenate,
    Input,
    Lambda,
    LeakyReLU,
    ZeroPadding2D,
    BatchNormalization,
    UpSampling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image.')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def darknet_conv(x, filters, size, strides=1, batch_norm=True):
    """
    Padding (if necessary) & Conv, with BN (if not turned off)
    --- what about biases when using BN?

    :param x:
    :param filters:
    :param size:
    :param strides:
    :param batch_norm:
    :return: (batch_size, new_h, new_w, filters)
    """
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def darknet_residual(x, filters):
    """
    A residual block composed of darknet_conv.

    :param x:
    :param filters:
    :return:
    """
    prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    x = Add()([prev, x])
    return x


def darknet_block(x, filters, blocks):
    """
    A block composed of darknet_conv & $blocks of darknet_residual

    :param x:
    :param filters:
    :param blocks:
    :return:
    """
    x = darknet_conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknet_residual(x, filters)
    return x


def darknet(name=None):
    """
    classification network, output 2-D images with $filters filter maps.
    :return:
    """
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32, 3)
    x = darknet_block(x, 64, 1)
    x = darknet_block(x, 128, 2)
    x = x_36 = darknet_block(x, 256, 8)  # x is the output of the 36th layer
    x = x_61 = darknet_block(x, 512, 8)
    x = darknet_block(x, 1024, 4)
    return Model(inputs, (x_36, x_61, x), name=name)


def yolo_output(filters, anchors, classes, name=None):
    """
    Turn 2-D features maps into object detection vectors (anchors * (classes+5)).

    :param filters: number of filters
    :param anchors: number of anchor boxes, e.g., 3
    :param classes: number of classes, e.g., 80
    :param name:
    :return: (batch_size, w, h, 3, 85) -- only one class is possible for the same position & same anchor
    """

    def yl_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x, filters * 2, 3)
        # Generate anchors*(classes + 4 + 1) filter maps
        x = darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
        # then reshape in order to compare with ground truths
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return Model(inputs, x, name=name)(x_in)

    return yl_output


def yolo_conv(filters, name=None):
    """
    Further convolutions over darknet outputs.

    :param filters:
    :param name:
    :return:
    """

    def yl_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yl_conv


def yolo_boxes(pred, anchors, classes):
    """
    Normalize predictions according to formulas (Equations on page 1 of the YOLOV3 paper.)

    :param pred: shape = (batch_size, grid, grid, anchors, 5+classes)
    :param anchors:
    :param classes:
    :return:
        bbox - shape (batch_size, grid, grid, anchors, 4);
        objectness - (batch_size, )
        class_probs - (batch_size, classes, )
        pred_box - (batch_size, grid, grid, anchors, 4)
    """
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    # Calculate for each grid point. Build a meshgrid, stack the arrays to form coordinates,
    #  and then on each coordinate do the arithmetic.
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (grid, grid, 1, 2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2

    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    """

    :param outputs:
    :param anchors:
    :param masks:
    :param classes:
    :return:
    """
    b, c, t = [], [], []
    for op in outputs:
        # each output's a tuple (bbox, objectness, class_probs, pred_box)
        #  op[0]->bbox, of shape (batch_size, grid, grid, anchors, 4)
        # tf.shape(op[0])[0] == batch_size, tf.shape(op[0])[-1] == 4
        b.append(tf.reshape(op[0], (tf.shape(op[0])[0], -1, tf.shape(op[0])[-1])))
        c.append(tf.reshape(op[1], (tf.shape(op[1])[0], -1, tf.shape(op[1])[-1])))
        t.append(tf.reshape(op[2], (tf.shape(op[2])[0], -1, tf.shape(op[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    # combined_non_max_suppression([batch_size, num_boxes, 1, 4], [batch_size, num_boxes, num_classes])
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )
    return boxes, scores, classes, valid_detections


def yolo_v3(size=None, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, 3], name='input')

    x_36, x_61, x = darknet(name='yolo_darknet')(x)
    print("x's shape after darknet: ", x.shape)

    x = yolo_conv(512, name='yolo_conv_0')(x)
    output0 = yolo_output(512, len(masks[0]), classes)(x)

    x = yolo_conv(256, name='yolo_conv_1')((x, x_61))
    output1 = yolo_output(256, len(masks[1]), classes)(x)

    x = yolo_conv(128, name='yolo_conv_2')((x, x_36))
    output2 = yolo_output(128, len(masks[2]), classes)(x)

    if training:
        return Model(inputs, (output0, output1, output2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes))(
        (boxes_0, boxes_1, boxes_2))

    return Model(inputs, outputs, name='yolov3')


def yolo_loss(anchors, classes=80, ignore_thresh=0.5):
    """
    Transform all y_true and y_pred & calculate losses.

    :param anchors:
    :param classes:
    :param ignore_thresh:
    :return:
    """

    def yl_loss(y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = (true_box[..., 2:4] - true_box[..., 0:2])

        # Emphasize small scale objects.
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        # deviation of the true box from the grid point
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        obj_mask = tf.squeeze(true_obj, -1)  # remove the trailing, dim-1 dimension
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))),
                axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
                   (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yl_loss
