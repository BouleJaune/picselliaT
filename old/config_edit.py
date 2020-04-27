import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util



#config keys = ['model', 'train_config', 'train_input_config', 
# 'eval_config', 'eval_input_configs', 'eval_input_config']

flags = tf.app.flags
flags.DEFINE_string('model_dir', None, 'Path to model')
flags.DEFINE_integer("batch_size", None, 'Batch size')
flags.DEFINE_float('learning_rate', None, 'Learning rate')
FLAGS = flags.FLAGS




## update num_classes
def update_num_classes(model_config, label_map):
    n_classes = len(label_map.item)
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")
    


## paths indispensables : 

def update_different_paths(config_dict, ckpt_path, label_map_path, train_record_path, eval_record_path):
    config_dict["train_config"].fine_tune_checkpoint = ckpt_path
    config_util._update_label_map_path(config_dict, label_map_path)
    config_util._update_tf_record_input_path(config_dict["train_input_config"], train_record_path)
    config_util._update_tf_record_input_path(config_dict["eval_input_config"], eval_record_path)



def main():
    model_path = FLAGS.model_dir
    if model_path is None:
        raise Exception("Please give model_path")

    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

    configs = config_util.get_configs_from_pipeline_file(model_path+'/model/pipeline.config')
    label_map = label_map_util.load_labelmap(model_path+"/label_map.pbtxt")

    ## update learning rate
    ## Change le LR initial puis les schedules proportionnellement, exemple :
    ## LR_initial = 10
    ## LR_schedule = 1
    ## new_LR = 100
    ## new_LR_schedule = 10
    ## fonctionnenement de schedule:
    ## schedule : {step: n_steps 
    #                   learning_rate: lr}
    # mets à jour le LR à n_steps par lr, plusieurs schedule, plusieurs màj


    if learning_rate is not None:
        config_util._update_initial_learning_rate(configs, learning_rate)

    ## update batch_size
    if batch_size is not None:
        config_util._update_batch_size(configs, batch_size)

    ##update num classes et paths
    update_num_classes(configs["model"], label_map)
    update_different_paths(configs, ckpt_path=model_path+"/model/model.ckpt", label_map_path=model_path+"/label_map.pbtxt", 
                                 train_record_path=model_path+"/train.record", eval_record_path=model_path+"/eval.record")

    ## save config en protobuff txt
    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=model_path)
    print("successfully saved in "+model_path)


if __name__=='__main__':
    main()



### différentes fonctions de eval_util
# ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 
# '_check_and_convert_legacy_input_config_key', '_get_classification_loss', '_is_generic_key', 
# '_maybe_update_config_with_key_value', '_update_all_eval_input_configs', '_update_batch_size', 
# '_update_classification_localization_weight_ratio', '_update_focal_loss_alpha', '_update_focal_loss_gamma', 
# '_update_generic', '_update_initial_learning_rate', '_update_label_map_path', '_update_mask_type', 
# '_update_momentum_optimizer_value', '_update_retain_original_image_additional_channels', 
# '_update_retain_original_images', '_update_tf_record_input_path', '_update_train_steps', 
# '_update_use_bfloat16', '_update_use_moving_averages', '_validate_message_has_field', 
# 'absolute_import', 'check_and_parse_input_config_key', 'create_configs_from_pipeline_proto', 
# 'create_pipeline_proto_from_configs', 'division', 'eval_pb2', 'file_io', 'get_configs_from_multiple_files', 
# 'get_configs_from_pipeline_file', 'get_graph_rewriter_config_from_file', 'get_image_resizer_config', 
# 'get_learning_rate_type', 'get_number_of_classes', 'get_optimizer_type', 'get_spatial_image_size', 
# 'graph_rewriter_pb2', 'input_reader_pb2', 'merge_external_params_with_configs', 'model_pb2', 'os', 
# 'pipeline_pb2', 'print_function', 'remove_unecessary_ema', 'save_pipeline_config', 'text_format', 
# 'tf', 'train_pb2', 'update_input_reader_config']