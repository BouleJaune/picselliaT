import json
import tensorflow as tf



flags = tf.app.flags
# flags.DEFINE_string('output_path', '', 'Path to output labelmap.pbtxt')
flags.DEFINE_string('json_path', '', 'Path to annotation.json')

FLAGS = flags.FLAGS

def main(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
        ## create labelmap.pbtxt
        categories = data["categories"]
        with open("label_map.pbtxt", "w+") as labelmap_file:
            k=0
            for category in categories:
                k+=1
                name = category["name"]
                labelmap_file.write("item {\n\tname: \""+name+"\""+"\n\tid: "+str(k)+"\n}\n")
            labelmap_file.close()
        print("Label_map.pbtxt crée")

## python create_label_map.py --json_path=<path_to.json> 

if __name__ == "__main__":
    json_file_path = FLAGS.json_path
    # output_path = FLAGS.output_dir
    main(json_file_path)
    