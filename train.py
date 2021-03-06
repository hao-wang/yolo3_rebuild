from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard)
from yolov3_tf2.models import (
    yolo_loss, yolo_v3, yolo_anchors, yolo_anchor_masks)
from yolov3_tf2 import dataset
from yolov3_tf2.utils import freeze_all

flags.DEFINE_integer('size', 416, 'image_size')
flags.DEFINE_enum('start_from',
                    'flowchart',
                    ['flowchart', 'yolo'],
                    'continue from last training, or yolo')
flags.DEFINE_string('root_dir', './data', 'root data dir')
flags.DEFINE_string('spec_dir', 'FC_offline', 'specific data')
flags.DEFINE_integer('num_classes', 7, 'number of classes')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('epochs', 20, 'epochs')
flags.DEFINE_integer('weights_num_classes',
                     80,
                     'number of weights classes, specify num class for `weights` file '
                     'if different, useful in transfer learning with different number of classes')
flags.DEFINE_enum('mode',
                  'fit',
                  ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer',
                  'darknet',
                  ['none', 'darknet', 'fine_tune', 'frozen'],
                  'transfer learning mode, '
                  'none: training from scratch, '
                  'darknet: transfer darknet, '  # inherit & freeze darknet weights, other layers from scratch
                  'no_output: transfer all but output, '  # inherit & freeze all but output
                  'frozen: transfer and freeze all, '  # nothing changed
                  'fine_tune: transfer all and freeze darknet only.'  # inherit all weights, freeze only darknet
                  )
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_string('logdir', './logs', 'log directory')


def main(_argv):
    model = yolo_v3(FLAGS.size, training=True, classes=FLAGS.num_classes)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    train_data = os.path.join(FLAGS.root_dir, FLAGS.spec_dir, 'flowchart_train.tfrecord')
    val_data = os.path.join(FLAGS.root_dir, FLAGS.spec_dir, 'flowchart_val.tfrecord')
    classes = os.path.join(FLAGS.root_dir, 'flowchart.names')
    checkpoint_dir = os.path.join(FLAGS.root_dir, "checkpoints")

    weights = os.path.join(checkpoint_dir, 'yolov3.tf')

    if not os.path.exists(checkpoint_dir):
        print(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    else:
        print("writing to %s" % checkpoint_dir)

    if train_data:
        train_dataset = dataset.load_tfrecord_dataset(train_data, classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if val_data:
        val_dataset = dataset.load_tfrecord_dataset(
            val_data, classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()

    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    if FLAGS.start_from == "flowchart":
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("loading weights from %s" % latest)
        model.load_weights(latest)
        freeze_all(model.get_layer("yolo_darknet"))
    else:
        if FLAGS.transfer in ['darknet', 'no_output']:
            model_pretrained = yolo_v3(FLAGS.size, training=True,
                                       classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            model_pretrained.load_weights(weights)

            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights()
            )
            freeze_all(model.get_layer('yolo_darknet'))
        elif FLAGS.transfer in ['fine_tune', 'frozen']:
            model.load_weights(weights)
            if FLAGS.transfer == "fine_tune":
                freeze_all(model.get_layer("yolo_darknet"))
            else:
                freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [yolo_loss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    callback_list = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(os.path.join(checkpoint_dir, 'flowchart_{epoch}.tf'),
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir=FLAGS.logdir)
    ]

    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callback_list,
                        validation_data=val_dataset)


if __name__ == "__main__":
    app.run(main)
