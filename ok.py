from picsellia import Client
import main
from main import create_record_files, edit_config, train, legacy_train, tfevents_to_json

# train(model_dir=ckpt_output, pipeline_config_path=conf_dir+"pipeline.config", num_train_steps=200)

 
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
# export TF_FORCE_GPU_ALLOW_GROWTH=true 


clt = Client(token="463389a8-52bd-4fd3-bc0a-9198d43fe76b")


ckpt_output = clt.checkpoint_dir
conf_dir = clt.config_dir
model_selected = "models/mask_rcnn/"


def config_initiale(clt):
    clt.dl_annotations()
    clt.generate_labelmap()
    clt.local_pic_save(prop=0.5)
    create_record_files(label_path=clt.label_path, record_dir=clt.record_dir, tfExample_generator=clt.tf_vars_generator)
    edit_config(model_selected=model_selected, config_output_dir=conf_dir, record_dir=clt.record_dir, 
                label_map_path=clt.label_path, masks="PNG_MASKS", num_steps=20)

# config_initiale(clt)
main.edit_config_resume_from_ckpt(ckpt_path=ckpt_output, previous_config_dir=conf_dir, num_steps=10)

legacy_train(train_dir=ckpt_output, pipeline_config_path=conf_dir+"pipeline.config")




# tfevents_to_json(path=ckpt_output, log_dir=clt.base_dir)