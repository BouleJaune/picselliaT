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

import functools
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer

# flags = tf.app.flags
# DEFINE_string('output_path', '', 'Path to output TFRecord')
# DEFINE_string('model_dir', None, 'Path to model')
# DEFINE_integer("batch_size", None, 'Batch size')
# DEFINE_float('learning_rate', None, 'Learning rate')
# FLAGS = FLAGS
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
        
'''client en arg, si on veut pas client en arg il faudrait mettre un generateur en arg, 
faire un générateur ici, le mettre en arg par défaut et quand nous on l'appelle on le change
'''
def create_record_files(label_path, record_dir, tfExample_generator):
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
    ensembles = ["train", "eval"]
    
    for ensemble in ensembles:
        output_path = record_dir+ensemble+".record"
        writer = tf.python_io.TFRecordWriter(output_path)
        for variables in tfExample_generator(label_map, ensemble=ensemble):
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

#ok 5Rif9Sh2
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
        configs["train_input_config"].mask_type = 1
        configs["eval_input_config"].mask_type = 1
    else:
        raise ValueError("Wrong Mask type provided")
#ok
def edit_config(model_selected, config_output_dir, num_steps, label_map_path, record_dir, masks=None, batch_size=None, learning_rate=None):

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
    configs = config_util.get_configs_from_pipeline_file(model_selected+'pipeline.config')
    label_map = label_map_util.load_labelmap(label_map_path)

    # configs["eval_config"].metrics_set="coco_detection_metrics"

    config_util._update_train_steps(configs, num_steps)

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
    update_different_paths(configs, ckpt_path=model_selected+"model.ckpt", label_map_path=label_map_path, 
                                train_record_path=record_dir+"train.record", eval_record_path=record_dir+"eval.record")

    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=config_output_dir)
    print("Configuration successfully edited and saved in "+config_output_dir)

def edit_config_resume_from_ckpt(ckpt_path, previous_config_dir, num_steps):
    configs = config_util.get_configs_from_pipeline_file(previous_config_dir+"pipeline.config")
    ckpt_steps = configs["train_config"].num_steps
    configs["train_config"].fine_tune_checkpoint = ckpt_path+str(ckpt_steps)+"/"+"model.ckpt-"+str(ckpt_steps)
    added_num_steps =  ckpt_steps + num_steps
    config_util._update_train_steps(configs, added_num_steps)
    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=previous_config_dir)
    print("Configuration successfully edited to resume from checkpoint")


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




def legacy_train(master='', task=0, num_clones=1, clone_on_cpu=False, worker_replicas=1, ps_tasks=0, 
                    train_dir='', pipeline_config_path='', train_config_path='', input_config_path='', model_config_path=''):
    
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    train_dir = train_dir+str(configs["train_config"].num_steps)

    tf.logging.set_verbosity(tf.logging.INFO)
    assert train_dir, '`train_dir` is missing.'
    if task == 0: tf.gfile.MakeDirs(train_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if task == 0:
            tf.gfile.Copy(pipeline_config_path,
                          os.path.join(train_dir, 'pipeline.config'),
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
                tf.gfile.Copy(config, os.path.join(train_dir, name),
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
        train_dir,
        graph_hook_fn=graph_rewriter_fn)




def tfevents_to_json(path, log_dir):
    events = [filename for filename in os.listdir(path) if filename.startswith("events.out")]
    for k,event in enumerate(events):
        event_acc = EventAccumulator(path+event).Reload()
        logs = dict()
        for scalar_key in event_acc.scalars.Keys():
            scalar_dict = {"wall_time": [], "step": [], "value": []}
            for scalars in event_acc.Scalars(scalar_key):
                scalar_dict["wall_time"].append(scalars.wall_time)
                scalar_dict["step"].append(scalars.step)
                scalar_dict["value"].append(scalars.value)
            logs[scalar_key] = scalar_dict
        with open(log_dir+"logs"+str(k+1)+".json", "w") as f:
            json.dump(logs, f)

        tags = event_acc.Tags().keys()
        print(tags)


