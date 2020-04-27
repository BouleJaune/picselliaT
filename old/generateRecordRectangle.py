import tensorflow as tf
import json
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import io
from PIL import Image
import os

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS



# data["images"][k]["external_pictures_url"] => file_name d'image

def create_tf_example(image_ids, data, label_map):
    internal_picture_id = image_ids["internal_picture_id"]
    external_picture_url = image_ids["external_picture_url"]
    with tf.gfile.GFile("Images/"+external_picture_url, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = image_ids["external_picture_url"].encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for annotations_images in data["annotations"]:
        if internal_picture_id==annotations_images["internal_picture_id"]:
            for annotation in annotations_images["annotations"]:
                xmin = annotation["rectangle"]["top"]
                xmax = xmin + annotation["rectangle"]["width"]
                ymin = annotation["rectangle"]["left"]
                ymax = ymin + annotation["rectangle"]["height"]
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                classes_text.append(annotation["label"].encode("utf8"))
                label_id = label_map[annotation["label"]]
                classes.append(label_id)

   
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    json_file_path = "annotationBBOX.json"

    with open(json_file_path) as json_file:
        data = json.load(json_file)
    label_map = label_map_util.load_labelmap("label_map.pbtxt")
    label_map = label_map_util.get_label_map_dict(label_map)
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for image_ids in data["images"]:
        tf_example = create_tf_example(image_ids, data, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()


# python generateRecord.py --output_path==output_path/file.record