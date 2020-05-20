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

import functools
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.legacy import evaluator

from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from tensorflow.python.util import deprecation



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
        



def create_record_files(label_path, record_dir, tfExample_generator, annotation_type):
    '''
        Ne gère que des fichiers d'annotations entièrement avec geometry = polygon et sans 'vide'!!
        
        Génère un fichier .record à partir et l'enregistre dans output_path. 
        à voir pour split en train/val

        Args:
            output_path: Doit contenir le nom du fichier (path/to/record/file.record)
            label_map: Sous format protobuf
    '''
    label_map = label_map_util.load_labelmap(label_path)
    label_map = label_map_util.get_label_map_dict(label_map) 
    datasets = ["train", "eval"]
    
    for dataset in datasets:
        output_path = os.path.join(record_dir, dataset+".record")
        writer = tf.python_io.TFRecordWriter(output_path)
        for variables in tfExample_generator(label_map, ensemble=dataset, annotation_type=annotation_type):
            if annotation_type=="polygon":
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
                
            elif annotation_type=="rectangle":
                (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                        encoded_jpg, image_format, classes_text, classes) = variables

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
                    'image/object/class/label': dataset_util.int64_list_feature(classes)
                    }))

            writer.write(tf_example.SerializeToString())    
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))

def update_num_classes(config_dict, label_map):
    ''' Mets à jour num_classes dans la config protobuf par rapport au nombre de label de la label_map

        Args :
        config_dict:  
        label_map: label_map sous forme protobuf

        Raises:
            ValueError si le modèle n'est pas reconnu. (backbone)
    '''
    model_config = config_dict["model"]
    n_classes = len(label_map.item)
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def set_image_resizer(config_dict, shape):

    model_config = config_dict["model"]
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        image_resizer = model_config.faster_rcnn.image_resizer
    elif meta_architecture == "ssd":
        image_resizer = model_config.ssd.image_resizer
    else:
        raise ValueError("Unknown model type: {}".format(meta_architecture))
    
    if image_resizer.HasField("keep_aspect_ratio_resizer"):
        image_resizer.keep_aspect_ratio_resizer.max_dimension = shape[1]
        image_resizer.keep_aspect_ratio_resizer.min_dimension = shape[0]

    elif image_resizer.HasField("fixed_shape_resizer"):
        image_resizer.fixed_shape_resizer.height = shape[1]
        image_resizer.fixed_shape_resizer.width = shape[0]

def edit_eval_config(config_dict, annotation_type, eval_number):
    eval_config = config_dict["eval_config"]
    eval_config.num_visualizations = 0
    if annotation_type=="rectangle":
        eval_config.metrics_set[0] = "coco_detection_metrics"
    elif annotation_type=="polygon":
        eval_config.metrics_set[0] = "coco_mask_metrics"
    else:
        raise ValueError("Wrong annotation type provided")
    if isinstance(eval_number, int):
        eval_config.num_examples = eval_number
    else: 
        print("eval_number has type ", type(eval_number), " and not int")


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
        configs["train_input_config"].mask_type = 1
        configs["eval_input_config"].mask_type = 1
    else:
        raise ValueError("Wrong Mask type provided")

def edit_config(model_selected, config_output_dir, num_steps, label_map_path, record_dir, eval_number, resizer_size=None,
        annotation_type="polygon", batch_size=None, learning_rate=None):
    '''
        Suppose que la label_map et les .record sont générés
        Potentiellement mettre en argument la label_map plutôt que la reload
        Pour le moment edit : paths, num_classes, batch_size, learning_rate
        Args: 
            model_path: path du dossier du modèle
            batch_size: batch_size, si non précisé, pas de modifications et garde la valeur de base
            learning_rate: learning_rate, si non précisé, pas de modifications et garde la valeur de base

    '''

    
    file_list = os.listdir(model_selected)
    ckpt_ids = []
    for p in file_list:
        if "index" in p:
            if "-" in p:
                ckpt_ids.append(int(p.split('-')[1].split('.')[0]))
    if len(ckpt_ids)>0:
        ckpt_path = os.path.join(model_selected,"model.ckpt-{}".format(str(max(ckpt_ids))))
    
    else:
        ckpt_path = os.path.join(model_selected, "model.ckpt")

    configs = config_util.get_configs_from_pipeline_file(os.path.join(model_selected,'pipeline.config'))
    label_map = label_map_util.load_labelmap(label_map_path)

    config_util._update_train_steps(configs, num_steps)
    update_different_paths(configs, ckpt_path=ckpt_path, 
                            label_map_path=label_map_path, 
                            train_record_path=os.path.join(record_dir, "train.record"), 
                            eval_record_path=os.path.join(record_dir,"eval.record"))

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

    if annotation_type=="polygon":
        edit_masks(configs, mask_type="PNG_MASKS")
   
    if resizer_size is not None:
        set_image_resizer(configs, resizer_size)

    edit_eval_config(configs, annotation_type, eval_number)
    update_num_classes(configs, label_map)
    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=config_output_dir)


def train(master='', save_summaries_secs=30, task=0, num_clones=1, clone_on_cpu=False, worker_replicas=1, ps_tasks=0, 
                    ckpt_dir='', conf_dir='', train_config_path='', input_config_path='', model_config_path=''):   
   
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    deprecation.silence()
    pipeline_config_path = os.path.join(conf_dir,"pipeline.config")
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    assert ckpt_dir, '`ckpt_dir` is missing.'
    if task == 0: tf.gfile.MakeDirs(ckpt_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if task == 0:
            tf.gfile.Copy(pipeline_config_path,
                          os.path.join(ckpt_dir, 'pipeline.config'),
                          overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
                        model_config_path=model_config_path,
                        train_config_path=train_config_path,
                        train_input_config_path=input_config_path)
        if task == 0:
            for name, config in [('model.config', model_config_path),
                                ('train.config', train_config_path),
                                ('input.config', input_config_path)]:
                tf.gfile.Copy(config, os.path.join(ckpt_dir, name),
                            overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                job_name=task_info.type,
                                task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        num_clones,
        worker_replicas,
        clone_on_cpu,
        ps_tasks,
        worker_job_name,
        is_chief,
        ckpt_dir,
        graph_hook_fn=graph_rewriter_fn,
        save_summaries_secs=save_summaries_secs)

def evaluate(eval_dir, config_dir, checkpoint_dir, eval_training_data=False, run_once=True):

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.reset_default_graph()
    tf.gfile.MakeDirs(eval_dir)
    configs = config_util.get_configs_from_pipeline_file(os.path.join(config_dir, "pipeline.config"))
    model_config = configs['model']
    eval_config = configs['eval_config']
    input_config = configs['eval_input_config']
    if eval_training_data:
        input_config = configs['train_input_config']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    categories = label_map_util.create_categories_from_labelmap(
      input_config.label_map_path)

    if run_once:
        eval_config.max_evals = 1

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=False)

    metrics = evaluator.evaluate(
          create_input_dict_fn,
          model_fn,
          eval_config,
          categories,
          checkpoint_dir,
          eval_dir,
          graph_hook_fn=graph_rewriter_fn)
    return {k:str(round(v, 3)) for k,v in metrics.items()}



def tfevents_to_dict(path):
    event = [filename for filename in os.listdir(path) if filename.startswith("events.out")][0]
    event_acc = EventAccumulator(os.path.join(path,event)).Reload()
    logs = dict()
    for scalar_key in event_acc.scalars.Keys():
        scalar_dict = {"wall_time": [], "step": [], "value": []}
        for scalars in event_acc.Scalars(scalar_key):
            scalar_dict["wall_time"].append(scalars.wall_time)
            scalar_dict["step"].append(scalars.step)
            scalar_dict["value"].append(scalars.value)
        logs[scalar_key] = scalar_dict
    return logs


def export_infer_graph(ckpt_dir, exported_model_dir, pipeline_config_path,
                        write_inference_graph=False, input_type="image_tensor", input_shape=None):
    
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.reset_default_graph()
    pipeline_config_path = os.path.join(pipeline_config_path,"pipeline.config")
    config_dict = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    ckpt_number = str(config_dict["train_config"].num_steps)
    pipeline_config = config_util.create_pipeline_proto_from_configs(config_dict)

    trained_checkpoint_prefix = os.path.join(ckpt_dir,'model.ckpt-'+ckpt_number)
    exporter.export_inference_graph(
        input_type, pipeline_config, trained_checkpoint_prefix,
        exported_model_dir, input_shape=input_shape,
        write_inference_graph=write_inference_graph)

