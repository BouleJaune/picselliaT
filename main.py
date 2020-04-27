import cv2
import json
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import io
from PIL import Image, ImageDraw
import os
import numpy as np 
import cv2

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('model_dir', None, 'Path to model')
flags.DEFINE_integer("batch_size", None, 'Batch size')
flags.DEFINE_float('learning_rate', None, 'Learning rate')
FLAGS = flags.FLAGS
json_file_path = "annotations/annotationPoly.json"



def create_tf_example(image_ids, data, label_map):

    '''
        Génère un objet tf.Example à partir d'une image.
        Les images doivent se situer dans un dossier /Images
        Ne gère que des fichiers d'annotations entièrement avec geometry = polygon et sans 'vide'!!
        
        Args:
            image_ids: les internal_picture_id et external_picture_url de l'image dans un dict
            data: Le fichier d'annotations.json loaded
            label_map: La label_map correspondante sous forme de dict.

        Returns:
            tf_example: Un message protobuf tf.Example
    '''
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
    masks = []
    for annotations_images in data["annotations"]:
        if internal_picture_id==annotations_images["internal_picture_id"]:
            annot = annotations_images["annotations"]
            for a in annot:
                geo = a["polygon"]["geometry"]
                poly = []
                for coord in geo:           
                    poly.append([[coord["x"],coord["y"]]])    
                    
                poly = np.array(poly, dtype=np.float32)
                mask = np.zeros((height, width), dtype=np.uint8)
                mask = Image.fromarray(mask)
                ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
                maskByteArr = io.BytesIO()
                mask.save(maskByteArr, format="PNG")
                maskByteArr = maskByteArr.getvalue()
                masks.append(maskByteArr)
                
                x,y,w,h = cv2.boundingRect(poly)
                xmins.append(x/width)
                xmaxs.append((x+w)/width)
                ymins.append(y/height)
                ymaxs.append((y+h)/height)
                classes_text.append(a["label"].encode("utf8"))
                label_id = label_map[a["label"]]
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
        'image/object/mask': dataset_util.bytes_list_feature(masks)
    }))
    return tf_example



def create_record_file(json_file_path, label_map, output_path): 
    '''
        Génère un fichier .record à partir des tf.Examples produits par create_tf_example et l'enregistre 
        dans output_path. 
        Suppose que les images sont dans /Images. (cf create_tf_example)
        à voir pour split en train/val

        Args:
            json_file_path: Doit contenir le nom du fichier  (path/to/json/annotations.json)
            label_map: label_map loaded en protobuf et non dict (cf create_label_map)
            output_path: Doit contenir le nom du fichier (path/to/record/file.record)

    '''
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    label_map = label_map_util.get_label_map_dict(label_map)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for image_ids in data["images"]:
        tf_example = create_tf_example(image_ids, data, label_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


def create_label_map(json_file_path):
    '''
        Génère un fichier label_map.pbtxt en protobuf text à partir du fichier d'annotations.json
        La label_map associe à chaque label un id.
        label_map_util.load_labelmap("label_map.pbtxt") pour load une label map en format protobuf
        label_map = label_map_util.get_label_map_dict(label_map) pour l'avoir sous forme de dictionnaire
        
        Args:
            json_file_path: Doit contenir le nom du fichier  (path/to/json/annotations.json)

    '''
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    categories = data["categories"]
    with open("label_map.pbtxt", "w+") as labelmap_file:
        k=0
        for category in categories:
            k+=1
            name = category["name"]
            labelmap_file.write("item {\n\tname: \""+name+"\""+"\n\tid: "+str(k)+"\n}\n")
        labelmap_file.close()
    print("label_map.pbtxt crée")
        
label_map = label_map_util.load_labelmap("label_map.pbtxt")


def update_num_classes(model_config, label_map):
    ''' Mets à jour num_classes dans la config protobuf par rapport au nombre de label de la label_map

        Args :
         model_config:  model_pb2.DetectionModel. (config["model"])
         label_map: label_map sous forme protobuf

        Raises:
            ValueError si le modèle n'est pas reconnu. (backbone)
    '''
    n_classes = len(label_map.item)
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")
    




def update_different_paths(config_dict, ckpt_path, label_map_path, train_record_path, eval_record_path):
    '''
        Set les bons paths dans le fichier de configuration protobuf.

    '''
    config_dict["train_config"].fine_tune_checkpoint = ckpt_path
    config_util._update_label_map_path(config_dict, label_map_path)
    config_util._update_tf_record_input_path(config_dict["train_input_config"], train_record_path)
    config_util._update_tf_record_input_path(config_dict["eval_input_config"], eval_record_path)



def edit_config(model_path=None, batch_size=None, learning_rate=None):

    '''
        Modifie le fichier de config protobuf de base et enregistre la nouvelle config.
        Folder structure :
            model_path
            ├──model
            |   ├──checkpoint (du zoo)
            |   ├──model.ckpt
            |   ├── ... autre files .ckpt
            |   ├──frozen_inference_graph.pb
            |   ├──pipeline.config (de base du zoo)
            |   └──save_model
            |       └──saved_model.pb => pour l'inférence
            ├──Images
            |   └── .... images
            ├──annotations
            |   └── annotations.json
            ├──cp
            |   └── le dossier où mettre nos checkpoints
            ├──pipeline.config => la version éditée
            ├──label_map.pbtxt
            └──record.record
        à revoir

        Suppose que la label_map et les .record sont générés
        Potentiellement mettre en argument la label_map plutôt que la reload
        Pour le moment edit : paths, num_classes, batch_size, learning_rate
        Args: 
            model_path: path du dossier du modèle
            batch_size: batch_size, si non précisé, pas de modifications et garde la valeur de base
            learning_rate: learning_rate, si non précisé, pas de modifications et garde la valeur de base

        Raises:
            Exception si le model_path n'est pas fourni
        
    '''
    if model_path is None:
        raise Exception("Please give model_path")

    configs = config_util.get_configs_from_pipeline_file(model_path+'/model/pipeline.config')
    label_map = label_map_util.load_labelmap(model_path+"/label_map.pbtxt")

    if learning_rate is not None:
            ''' Update learning rate
            Change le LR initial puis les schedules proportionnellement, exemple :
            LR_initial = 10
            LR_schedule = 1
            new_LR = 100
            new_LR_schedule = 10
            fonctionnenement de schedule:
            schedule : {step: n_steps 
                            learning_rate: lr}
            Mets à jour le LR à n_steps par lr, plusieurs schedule, plusieurs màj'''
        config_util._update_initial_learning_rate(configs, learning_rate)

    if batch_size is not None:
        config_util._update_batch_size(configs, batch_size)

    update_num_classes(configs["model"], label_map)
    update_different_paths(configs, ckpt_path=model_path+"/model/model.ckpt", label_map_path=model_path+"/label_map.pbtxt", 
                                 train_record_path=model_path+"/train.record", eval_record_path=model_path+"/eval.record")

    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=model_path)
    print("successfully saved in "+model_path)


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
        name = 'training_data'
        input_fn = eval_on_train_input_fn
        else:
        name = 'validation_data'
        # The first eval input will be evaluated.
        input_fn = eval_input_fns[0]
        if FLAGS.run_once:
        estimator.evaluate(input_fn,
                            steps=None,
                            checkpoint_path=tf.train.latest_checkpoint(
                                FLAGS.checkpoint_dir))
        else:
        model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                    train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
