from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import yolo_v3
from yolov3_tf2.utils import draw_outputs

data_dir = './data'
data_type = "FC_aug"
class_path = os.path.join(data_dir, 'flowchart.names')
weights_path = os.path.join('checkpoints', 'yolov3_train_10.tf')
image_path = os.path.join(data_dir, data_type, 'JPEGImages/writer000_fc_001.png')

flags.DEFINE_string('classes', class_path, 'path to class name file')
flags.DEFINE_string('weights', weights_path, 'path to weights file')
flags.DEFINE_string('image', image_path, 'path to weights file')
flags.DEFINE_string('output', 'output.jpg', 'output image path')
flags.DEFINE_integer('num_classes', 7, 'number of classes in the model')
flags.DEFINE_integer('size', 416, "image size as network's input")


def main(_argv):
    """
    CPU only.

    :return:
    """
    yolo = yolo_v3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights).expect_partial()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]

    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)

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
