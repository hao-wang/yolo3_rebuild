from absl import app, flags, logging
from absl.flags import FLAGS
import easyocr
import cv2
import numpy as np
import os
import random
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import yolo_v3
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('root_dir', './data', 'root data dir.')
flags.DEFINE_string('spec_dir', 'FC_offline', 'specific data set dir.')
flags.DEFINE_string('checkpoint', None, 'specific data set dir.')
flags.DEFINE_string('image', 'writer023_fc_016.png', 'specific image')
flags.DEFINE_string('extracted', 'extracted.jpg', 'image content extracted')
flags.DEFINE_string('output', 'output.jpg', 'output image path')
flags.DEFINE_integer('num_classes', 7, 'number of classes in the model')
flags.DEFINE_integer('size', 416, "image size as network's input")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_ocr_result(img, x1y1, x2y2):
    reader = easyocr.Reader(['en'])
    img_tmp = "image_%d.jpg" % random.randint(1, 10)
    cv2.imwrite(img_tmp, (img.numpy()*255).astype(int))
    result = reader.readtext(img_tmp, detail=0)

    return ' '.join(result)


def get_image_content(img, boxes, classes, nums, class_names):
    """

    Args:
        boxes: shape = (n_box, 4)
        classes:
        nums:

    Returns:
    """
    num_detect = nums.numpy()
    image_shape = np.flip(img.shape[0:2])
    img_ext = np.ones((image_shape[1], image_shape[0], 3)) * 255
    print("image extraction shape: ", img_ext.shape)
    content = []
    for i in range(num_detect):
        x1y1 = tuple((np.array(boxes[i][0:2]) * image_shape).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * image_shape).astype(np.int32))
        center = ((x1y1[0]+x2y2[0])//2, (x1y1[1]+x2y2[1])//2)
        class_name = class_names[int(classes[i])]
        if class_name not in ['arrow', 'connection', 'text']:
            text_content = get_ocr_result(img, x1y1, x2y2)
        elif class_name != 'text':
            text_content = class_name
        else:
            text_content = ""

        if not text_content:
            text_content = class_name

        content.append((text_content, x1y1, x2y2))
        # img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img_ext = cv2.putText(img_ext,
                              text_content,
                              center,
                              cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              1, (0, 0, 255), 2)

    return img_ext


def main(_argv):
    """
    :return:
    """
    classes = os.path.join(FLAGS.root_dir, 'flowchart.names')
    checkpoint_path = os.path.join(FLAGS.root_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if FLAGS.checkpoint:
        weights = os.path.join(checkpoint_path, FLAGS.checkpoint)
    else:
        weights = tf.train.latest_checkpoint(checkpoint_path)
    image = os.path.join(FLAGS.root_dir, FLAGS.spec_dir, 'JPEGImages', FLAGS.image)

    yolo = yolo_v3(classes=FLAGS.num_classes)
    yolo.load_weights(weights).expect_partial()

    class_names = [c.strip() for c in open(classes).readlines()]

    img_raw = tf.image.decode_image(
        open(image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    boxes, scores, classes, nums = yolo(img)
    img_ext = get_image_content(img[0], boxes[0], classes[0], nums[0], class_names)
    cv2.imwrite(FLAGS.extracted, img_ext)

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    print("output image")
    cv2.imwrite(FLAGS.output, img)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
