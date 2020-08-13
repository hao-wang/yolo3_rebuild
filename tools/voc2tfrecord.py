import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('root_dir', "./data", 'path to raw dataset')
flags.DEFINE_string('spec_dir', "FC_offline", 'path to specific data')
flags.DEFINE_enum('split', 'train', ['train', 'val'], 'specify train or val spit')


def build_example(annotation, class_map):
    """
    Read in image (via 'filename' key of annotation).
    Read all objects' bounding box coordinates & classifications.

    :param annotation:
    :param class_map:
    :return: tf.train.Example (a key->feature mapping)
    """
    img_path = os.path.join(
        FLAGS.root_dir, FLAGS.spec_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(float(annotation['size']['width']))
    height = int(float(annotation['size']['height']))
    # print(width, height)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    if 'object' in annotation:
        for obj in annotation['object']:
            x1 = float(obj['bndbox']['xmin']) / width
            y1 = float(obj['bndbox']['ymin']) / height
            x2 = float(obj['bndbox']['xmax']) / width
            y2 = float(obj['bndbox']['ymax']) / height
            xmin.append(x1)
            ymin.append(y1)
            xmax.append(x2)
            ymax.append(y2)
            if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                print(img_path, x1, y1, x2, y2)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    classes = os.path.join(FLAGS.root_dir, "flowchart.names")
    data_dir = os.path.join(FLAGS.root_dir, FLAGS.spec_dir)

    class_map = {name: idx for idx, name in enumerate(
        open(classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    image_list = open(os.path.join(
        data_dir, 'ImageSets', 'Main', 'flowchart_%s.txt' % FLAGS.split)).read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))

    output_file = os.path.join(data_dir, 'flowchart_%s.tfrecord' % FLAGS.split)

    writer = tf.io.TFRecordWriter(output_file)
    for image in tqdm.tqdm(image_list):
        name, _ = image.split()
        annotation_xml = os.path.join(
            FLAGS.root_dir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
