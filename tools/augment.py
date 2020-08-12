"""
Based on https://github.com/mukopikmin/bounding-box-augmentation
"""
from absl import flags, app
from absl.flags import FLAGS
import cv2
import glob
import imgaug as ia
import os
from pascal_voc_writer import Writer
import sequence
import xml.etree.ElementTree as ET


flags.DEFINE_string("image_source_dir",
                    "/Users/hao/Projects/dataset/FC_1.0_offline_extension/DB/images",
                    "image source directory")
flags.DEFINE_string("annot_source_dir",
                    "./data/FC_offline/Annotations",
                    "annotation source directory")

flags.DEFINE_string("image_output_dir",
                    "./data/FC_aug/JPEGImages",
                    "augmented images")
flags.DEFINE_string("annot_output_dir",
                    "./data/FC_aug/Annotations",
                    "augmented images' annotations.")
flags.DEFINE_integer("augment_size",
                     10,
                     "multiplies of original data")


def augment(augment_size=None):
    if not os.path.exists(FLAGS.annot_output_dir):
        os.mkdir(FLAGS.annot_output_dir)
    if not os.path.exists(FLAGS.image_output_dir):
        os.mkdir(FLAGS.image_output_dir)

    aug_size = augment_size or FLAGS.augment_size

    seq = sequence.get()
    for fn in glob.glob(FLAGS.annot_source_dir+'/*'):
        print(fn)
        stree = ET.parse(open(fn, 'r'))
        source_root = stree.getroot()
        img_name = source_root.find('./filename').text

        for i in range(aug_size):
            sp = img_name.split('.')
            img_outfile = '%s/%s-%02d.%s' % (FLAGS.image_output_dir, sp[0], i, sp[-1])
            xml_outfile = '%s/%s-%02d.xml' % (FLAGS.annot_output_dir, sp[0], i)

            seq_det = seq.to_deterministic()

            image = cv2.imread('%s/%s' % (FLAGS.image_source_dir, img_name))
            _bbs = []
            for obj in source_root.findall('./object'):
                bbox = obj.find('bndbox')
                bb = ia.BoundingBox(x1=int(float(bbox.find('xmin').text)),
                                    y1=int(float(bbox.find('ymin').text)),
                                    x2=int(float(bbox.find('xmax').text)),
                                    y2=int(float(bbox.find('ymax').text)),
                                    label=obj.find('name').text)
                _bbs.append(bb)

            bbs = ia.BoundingBoxesOnImage(_bbs, shape=image.shape)

            image_aug = seq_det.augment_images([image])[0]
            bbs_aug = seq_det.augment_bounding_boxes(
                [bbs])[0].remove_out_of_image().clip_out_of_image()

            writer = Writer(img_outfile,
                            int(float(source_root.find('size').find('width').text)),
                            int(float(source_root.find('size').find('height').text)))
            for bb in bbs_aug.bounding_boxes:
                writer.addObject(bb.label,
                                 int(bb.x1),
                                 int(bb.y1),
                                 int(bb.x2),
                                 int(bb.y2))

            cv2.imwrite(img_outfile, image_aug)
            writer.save(xml_outfile)


def main(_argv):
    augment(10)


if __name__ == "__main__":
    app.run(main)
