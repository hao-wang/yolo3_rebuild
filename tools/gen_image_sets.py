from absl import flags, app
from absl.flags import FLAGS
import glob
import os
import random

flags.DEFINE_string("root_dir", "./data", "directory containing all images")
flags.DEFINE_string("spec_dir", "FC_offline", "specific data used.")


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
    output_dir = os.path.join(FLAGS.data_dir, 'ImageSets/Main')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(FLAGS.data_dir, 'JPEGImages')
    train_file = os.path.join(FLAGS.data_dir, 'ImageSets/Main/flowchart_train.txt')
    val_file = os.path.join(FLAGS.data_dir, 'ImageSets/Main/flowchart_val.txt')

    train_set, val_set, test_set = generate_imagesets(image_dir, 1)
    write_imagesets(train_set, train_file)
    write_imagesets(val_set, val_file)
    # write_imagesets(test_set, FLAGS.test_file)


if __name__ == "__main__":
    app.run(main)
