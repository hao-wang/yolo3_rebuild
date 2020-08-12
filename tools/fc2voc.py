from absl import app, flags
from absl.flags import FLAGS
import cv2
import glob
from lxml import etree
import os


rawdata_dir = "/Users/hao/Projects/dataset"
data_dir = "./data"
flags.DEFINE_string("annot_source_dir",
                    os.path.join(rawdata_dir, "FC_1.0_offline_extension/DB/annotation"),
                    "Original dataset annotations directory.")
flags.DEFINE_string("image_source_dir",
                    os.path.join(rawdata_dir, "FC_1.0_offline_extension/DB/images"),
                    "Original images.")
flags.DEFINE_string("annot_output_dir",
                    os.path.join(data_dir, "FC_offline/Annotations"),
                    "PASCAL style annotations directory.")
flags.DEFINE_string("image_output_dir",
                    os.path.join(data_dir, "FC_offline/JPEGImages"),
                    "Images processed.")


def mask_frame_box(img, annot, output_image_path):
    """Return (x_min, y_min, x_max, y_max).
    """
    x_list = []
    y_list = []
    for ele in annot.xpath("registration/markers/point"):
        # print(ele.items())
        x_list.append(float(ele.items()[0][1]))
        y_list.append(float(ele.items()[1][1]))

    x_list = sorted([int(x) for x in x_list])
    y_list = sorted([int(y) for y in y_list])

    img[y_list[0]-3: y_list[1]+3] = 255
    img[y_list[2]-3: y_list[3]+3] = 255
    img[:, x_list[0]-3: x_list[1]+3] = 255
    img[:, x_list[2]-3: x_list[3]+3] = 255
    cv2.imwrite(output_image_path, img)


def get_frame_box(annot):
    """Return (x_min, y_min, x_max, y_max).
    """
    x_list = []
    y_list = []
    for ele in annot.xpath("registration/markers/point"):
        # print(ele.items())
        x_list.append(float(ele.items()[0][1]))
        y_list.append(float(ele.items()[1][1]))

    x_min = sorted(x_list)[1]
    y_min = sorted(y_list)[1]
    x_max = sorted(x_list)[-2]
    y_max = sorted(y_list)[-2]
    return x_min, y_min, x_max, y_max


def get_object_boxes(annot, frame_origin=[]):
    if frame_origin:
        x_ori, y_ori = frame_origin[0], frame_origin[1]

    coord_list = []
    for ele in annot.xpath('symbols/symbol'):
        object_name = ele.items()[1][1]
        x, y, w, h = [float(el[1]) for el in ele[0].items()]
        coords = [x, y, x + w, y + h, object_name]
        coord_list.append(coords)

    return coord_list


def convert_xml(file_path):
    """
    Convert annotations to Pascal-style, at the same time do some image masking/cropping too.
    Args:
        file_path:

    Returns:

    """
    print(file_path)
    source_tree = etree.parse(open(file_path, 'r'))
    source_annot = source_tree.getroot()

    file_name = file_path.rsplit('/', 1)[1]
    image_name = file_name.replace('.xml', '.png')
    input_image_path = os.path.join(FLAGS.image_source_dir, image_name)
    output_image_path = os.path.join(FLAGS.image_output_dir, image_name)

    img = cv2.imread(input_image_path)
    height, width, depth = img.shape
    mask_frame_box(img, source_annot, output_image_path)

    target_root = etree.Element("annotation")
    target_file = etree.SubElement(target_root, "filename")
    target_file.text = image_name
    target_size = etree.SubElement(target_root, "size")
    img_width = etree.SubElement(target_size, "width")
    img_height = etree.SubElement(target_size, "height")
    img_depth = etree.SubElement(target_size, "depth")
    img_width.text = str(width)  # str(frame_box[2] - frame_box[0])
    img_height.text = str(height)  # str(frame_box[3] - frame_box[1])
    img_depth.text = str(depth)

    for bbox in get_object_boxes(source_annot):
        obj = etree.SubElement(target_root, "object")

        name = etree.SubElement(obj, "name")
        name.text = bbox[-1]

        bndbox = etree.SubElement(obj, "bndbox")

        xmin = etree.SubElement(bndbox, "xmin")
        ymin = etree.SubElement(bndbox, "ymin")
        xmax = etree.SubElement(bndbox, "xmax")
        ymax = etree.SubElement(bndbox, "ymax")

        xmin.text = str(bbox[0])
        ymin.text = str(bbox[1])
        xmax.text = str(bbox[2])
        ymax.text = str(bbox[3])

    tree = etree.ElementTree(target_root)

    output_path = os.path.join(FLAGS.annot_output_dir, file_name)
    print(output_path)
    tree.write(output_path, pretty_print=True)


def main(argv):
    if not os.path.exists(FLAGS.image_output_dir):
        os.makedirs(FLAGS.image_output_dir)
    if not os.path.exists(FLAGS.annot_output_dir):
        os.makedirs(FLAGS.annot_output_dir)

    for fp in glob.glob(FLAGS.annot_source_dir + "/*"):
        convert_xml(fp)


if __name__ == "__main__":
    app.run(main)
