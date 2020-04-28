from object_detection.utils import label_map_util
from picsellia import Client
from main import create_record_file, edit_config, train

### train et eval split ? à revoir dans les paths de edit_config plus tard, pour le moment seulement "train.record"

clt = Client(token="463389a8-52bd-4fd3-bc0a-9198d43fe76b")


model_output = clt.checkpoint_dir

model_selected = "models/mask_rcnn"

clt.dl_annotations()
clt.generate_labelmap()
clt.local_png_save()
label_map = label_map_util.load_labelmap(clt.label_path)

create_record_file(client=clt, output_path=model_output+"train.record", label_map=label_map)

## model_selected et model_output
edit_config(model_selected=model_selected, model_output=model_output, label_map_path=clt.label_path, masks="PNG_MASKS")

train(model_dir=model_output, pipeline_config_path=model_output+"pipeline.config", num_train_steps=200)
