import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vu
from object_detection.protos import string_int_label_map_pb2 as pb
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder as TfDecoder
from google.protobuf import text_format

def main(tfrecords_filename, label_map=None):
    if label_map is not None:
        label_map_proto = pb.StringIntLabelMap()
        with tf.gfile.GFile(label_map,'r') as f:
            text_format.Merge(f.read(), label_map_proto)
            class_dict = {}
            for entry in label_map_proto.item:
                class_dict[entry.id] = {'name':entry.display_name}
    sess = tf.Session()
    decoder = TfDecoder(label_map_proto_file=label_map, use_display_name=False)
    sess.run(tf.tables_initializer())
    list_examples = []
    for record in tf.python_io.tf_record_iterator(tfrecords_filename):
        example = decoder.decode(record)
        list_examples.append(sess.run(example))
    return list_examples

list_examples = main(tfrecords_filename="train.record", label_map="label_map.pbtxt")

plt.imshow(list_examples[1]['image'])
plt.savefig("euh.jpg")

print(list_examples[0].keys())

print(list_examples[0]["groundtruth_area"])

print(list_examples[0]["groundtruth_classes"])