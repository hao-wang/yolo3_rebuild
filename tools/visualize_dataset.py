from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np

import os
import sys

sys.path.append('../yolo3_rebuild')
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.models import yolo_v3  # not used; only to import FLAGS.yolo_max_boxes
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('root_dir', './data', 'root data dir.')
flags.DEFINE_string('spec_dir', 'FC_offline', 'specific data set dir.')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('output', './output.jpg', 'path to output image')


def main(_argv):
    classes = os.path.join(FLAGS.root_dir, 'flowchart.names')
    dataset = os.path.join(FLAGS.root_dir, FLAGS.spec_dir, 'flowchart_train.tfrecord')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    dataset = load_tfrecord_dataset(dataset, classes, FLAGS.size)
    dataset = dataset.shuffle(128)

    for image, labels in dataset.take(1):
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]

        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    app.run(main)
