import cv2
import json
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection import model_hparams
from object_detection import model_lib
import io
from PIL import Image, ImageDraw
import os
import numpy as np 
import cv2
from picsellia import Client

# flags = tf.app.flags
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('model_dir', None, 'Path to model')
# flags.DEFINE_integer("batch_size", None, 'Batch size')
# flags.DEFINE_float('learning_rate', None, 'Learning rate')
# FLAGS = flags.FLAGS
# json_file_path = "annotations/annotationPoly.json"

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
        
#ok
def create_record_file(client, output_path, label_map):
    '''
        Ne gère que des fichiers d'annotations entièrement avec geometry = polygon et sans 'vide'!!
        
        Génère un fichier .record à partir et l'enregistre dans output_path. 
        à voir pour split en train/val

        Args:
            output_path: Doit contenir le nom du fichier (path/to/record/file.record)
            label_map: Sous format protobuf
    '''
    writer = tf.python_io.TFRecordWriter(output_path)
    label_map = label_map_util.get_label_map_dict(label_map) 

    for variables in client.tf_vars_generator(label_map):
        (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                encoded_jpg, image_format, classes_text, classes, masks) = variables

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
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

#ok
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

#ok
def update_different_paths(config_dict, ckpt_path, label_map_path, train_record_path, eval_record_path):
    '''
        Set les bons paths dans le fichier de configuration protobuf.

    '''
    config_dict["train_config"].fine_tune_checkpoint = ckpt_path
    config_util._update_label_map_path(config_dict, label_map_path)
    config_util._update_tf_record_input_path(config_dict["train_input_config"], train_record_path)
    config_util._update_tf_record_input_path(config_dict["eval_input_config"], eval_record_path)


def edit_masks(configs, mask_type="PNG_MASKS"):
    """Ajoute la prise en compte des masks dans la configuration.
        Args:
            configs: Dictionary of configuration objects.
            mask_type: String name to identify mask type, either "PNG_MASKS" or "NUMERICAL_MASKS"
    """
    configs["train_input_config"].load_instance_masks = True
    configs["eval_input_config"].load_instance_masks = True
    if mask_type=="PNG_MASKS":
        configs["train_input_config"].mask_type = 2
        configs["eval_input_config"].mask_type = 2
    elif mask_type=="NUMERICAL_MASKS":
        configs["train_input_config"].mask_type = 2
        configs["eval_input_config"].mask_type = 2
    else:
        raise ValueError("Wrong Mask type provided")
#ok
def edit_config(model_selected, model_output, label_map_path, masks=None, batch_size=None, learning_rate=None):

    '''
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


    configs = config_util.get_configs_from_pipeline_file(model_selected+'/pipeline.config')
    label_map = label_map_util.load_labelmap(label_map_path)

    # configs["eval_config"].metrics_set="coco_detection_metrics"


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

    if masks is not None:
        edit_masks(configs, mask_type=masks)


    update_num_classes(configs["model"], label_map)
    update_different_paths(configs, ckpt_path=model_selected+"/model.ckpt", label_map_path=label_map_path, 
                                train_record_path=model_output+"train.record", eval_record_path=model_output+"train.record")

    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=model_output)
    print("Configuration successfully edited and saved in "+model_output)



def train(model_dir=None, pipeline_config_path=None, num_train_steps=None, eval_training_data=False, 
            sample_1_of_n_eval_examples=1, sample_1_of_n_eval_on_train_examples=5, hparams_overrides=None,
            checkpoint_dir=None, run_once=False):

    if model_dir==None or pipeline_config_path==None:
        raise Exception("Please model_dir and pipeline config path")
    config = tf.estimator.RunConfig(model_dir=model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams= model_hparams.create_hparams(hparams_overrides),
        pipeline_config_path=pipeline_config_path,
        train_steps=num_train_steps,
        sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if checkpoint_dir is not None:
        if eval_training_data is not None:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if run_once:
            estimator.evaluate(input_fn,
                            steps=None,
                            checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, checkpoint_dir, input_fn,
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

        sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples)
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    

    if checkpoint_dir is not None:
        if eval_training_data is not None:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if run_once:
            estimator.evaluate(input_fn,
                            steps=None,
                            checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, checkpoint_dir, input_fn,
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
