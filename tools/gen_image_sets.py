from absl import flags, app
from absl.flags import FLAGS
import glob
import os
import random

data_dir = "FC_aug"
flags.DEFINE_string("image_dir", "./data/%s/JPEGImages" % data_dir, "directory containing all images")
flags.DEFINE_string("fileset_dir", "./data/%s/ImageSets/Main" % data_dir, "directory containing all file lists")
flags.DEFINE_string("train_file", "./data/%s/ImageSets/Main/flowchart_train.txt" % data_dir, "file containing train set")
flags.DEFINE_string("val_file", "./data/%s/ImageSets/Main/flowchart_val.txt" % data_dir, "file containing validation set")
flags.DEFINE_string("test_file", "./data/%s/ImageSets/Main/flowchart_test.txt" % data_dir, "file containing test set")


def write_imagesets(imageset, filepath):
    with open(filepath, 'a') as fw:
        for img, is_pos in imageset:
            fw.write("%s %d\n" % (img, is_pos))


def generate_imagesets(image_dir, is_pos=1):
    test_set = []
    train_set = []
    val_set = []

    random.seed = 42
    image_pattern = image_dir + "/*"
    for fn in glob.glob(image_pattern):
        img = fn.rsplit('.', 1)[0].rsplit('/', 1)[-1]
        rnd = random.random()
        if rnd < 0.1:
            test_set.append((img, is_pos))
        elif rnd > 0.8:
            val_set.append((img, is_pos))
        else:
            train_set.append((img, is_pos))

    return train_set, val_set, test_set


def main(_argv):
    if not os.path.exists(FLAGS.fileset_dir):
        os.makedirs(FLAGS.fileset_dir)

    train_set, val_set, test_set = generate_imagesets(FLAGS.image_dir, 1)
    write_imagesets(train_set, FLAGS.train_file)
    write_imagesets(val_set, FLAGS.val_file)
    write_imagesets(test_set, FLAGS.test_file)


if __name__ == "__main__":
    app.run(main)
