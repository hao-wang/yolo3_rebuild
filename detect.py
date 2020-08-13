from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import yolo_v3
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('root_dir', './data', 'root data dir.')
flags.DEFINE_string('spec_dir', 'FC_offline', 'specific data set dir.')
flags.DEFINE_string('checkpoint', 'yolov3_train_20.tf', 'specific data set dir.')
flags.DEFINE_string('image', 'writer000_fc_001.png', 'specific image')
flags.DEFINE_string('output', 'output.jpg', 'output image path')
flags.DEFINE_integer('num_classes', 7, 'number of classes in the model')
flags.DEFINE_integer('size', 416, "image size as network's input")


def main(_argv):
    """
    CPU only.

    :return:
    """
    classes = os.path.join(FLAGS.root_dir, 'flowchart.names')
    checkpoint_path = os.path.join(FLAGS.root_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    weights = os.path.join(checkpoint_path, FLAGS.checkpoint)
    image = os.path.join(FLAGS.root_dir, FLAGS.spec_dir, 'JPEGImages', 'writer000_fc_001.png')

    yolo = yolo_v3(classes=FLAGS.num_classes)
    yolo.load_weights(weights).expect_partial()

    class_names = [c.strip() for c in open(classes).readlines()]

    img_raw = tf.image.decode_image(
        open(image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    boxes, scores, classes, nums = yolo(img)

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    cv2.imwrite(FLAGS.output, img)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
