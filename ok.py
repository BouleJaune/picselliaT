from picsellia import Client
import main
from util.infer import infer
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
# export TF_FORCE_GPU_ALLOW_GROWTH=true 

path_models = "models/"
 
def train(token, model_name, model_selected, nb_steps,
         batch_size, learning_rate, mask_type=None):

    clt = Client(token=token)
    clt.init_model(model_name)
    model_selected = path_models + model_selected + "/"

    ## Prepare inputs
    clt.dl_annotations()
    clt.generate_labelmap()
    clt.local_pic_save(prop=0.5)

    main.create_record_files(label_path=clt.label_path, record_dir=clt.record_dir, 
                             tfExample_generator=clt.tf_vars_generator)

    # Edit .config file
    if clt.training_id==0:
        main.edit_config(model_selected=model_selected, config_output_dir=clt.config_dir,
                        record_dir=clt.record_dir, 
                        label_map_path=clt.label_path, 
                        masks=mask_type, 
                        num_steps=nb_steps,
                        batch_size=batch_size, 
                        learning_rate=learning_rate,
                        training_id=clt.training_id)
    else:
        previous_path = clt.base_dir.split("/")[:-1]
        previous_path[-1] = clt.training_id - 1
        model_selected = "{}/{}/{}/".format(*previous_path)+"checkpoint/"
        main.edit_config(model_selected=model_selected, config_output_dir=clt.config_dir,
                record_dir=clt.record_dir, 
                label_map_path=clt.label_path, 
                masks=mask_type, 
                num_steps=nb_steps,
                batch_size=batch_size, 
                learning_rate=learning_rate,
                training_id=clt.training_id)
    # Train
    main.legacy_train(ckpt_dir=clt.checkpoint_dir, 
                      conf_dir=clt.config_dir)

    # # Send logs to server
    dict_log = main.tfevents_to_dict(path=clt.checkpoint_dir)
    clt.send_logs(dict_log)

    # # # # Export inference graph
    main.export_infer_graph(ckpt_dir=clt.checkpoint_dir, 
                        exported_model_dir=clt.exported_model, 
                        pipeline_config_path=clt.config_dir,
                        write_inference_graph=False, input_type="image_tensor", input_shape=None)


    # Infer
    infer(clt.eval_list, exported_model_dir=clt.exported_model, 
          label_map_path=clt.label_path, results_dir=clt.results_dir, min_score_thresh=0.2)
    clt.send_examples()



'''list of possible models : 
'mask_rcnn' 
'faster_rcnn'
'ssd_inception' : train, log, export ok

mask_type = None or "PNG_MASKS"
'''

model_name = "ssd-test"

model_selected = "ssd_inception"
train(token="be6ea05d-52e1-4248-84f4-037739dd32cf", model_name=model_name, model_selected=model_selected,
    nb_steps=30, batch_size=5, learning_rate=None)
























def trainSsdInception(token, nb_steps, batch_size, learning_rate):
    clt = Client(token=token)
    clt.init_model("NouveauModel")
    ckpt_dir = clt.checkpoint_dir
    conf_dir = clt.config_dir
    model_selected = "models/ssd_inception/"
    ## Prepare inputs
    clt.dl_annotations()
    clt.generate_labelmap()
    clt.local_pic_save(prop=0.5)

    main.create_record_files(label_path=clt.label_path, record_dir=clt.record_dir, 
                             tfExample_generator=clt.tf_vars_generator)

    # Edit .config file
    if clt.training_id==0:
        main.edit_config(model_selected=model_selected, config_output_dir=conf_dir,
                        record_dir=clt.record_dir, 
                        label_map_path=clt.label_path, 
                        masks=None, 
                        num_steps=nb_steps,
                        batch_size=batch_size, 
                        learning_rate=learning_rate,
                        training_id=clt.training_id)
    else:
        previous_path = clt.base_dir.split("/")[:-1]
        previous_path[-1] = clt.training_id - 1
        model_selected = "{}/{}/{}/".format(*previous_path)+"checkpoint/"
        main.edit_config(model_selected=model_selected, config_output_dir=conf_dir,
                record_dir=clt.record_dir, 
                label_map_path=clt.label_path, 
                masks=None, 
                num_steps=nb_steps,
                batch_size=batch_size, 
                learning_rate=learning_rate,
                training_id=clt.training_id)
    # Train
    main.legacy_train(ckpt_dir=ckpt_dir, 
                      conf_dir=conf_dir)

    # Send logs to server
    dict_log = main.tfevents_to_dict(path=ckpt_dir)
    # clt.send_logs(dict_log)

    # Export inference graph
    main.export_infer_graph(ckpt_dir=ckpt_dir, 
                        exported_model_dir=clt.exported_model, 
                        pipeline_config_path=conf_dir,
                        write_inference_graph=False, input_type="image_tensor", input_shape=None)


    # # Infer
    # infer(clt.eval_list, exported_model_dir=clt.exported_model, 
    #       label_map_path=clt.label_path, results_dir=clt.results_dir, min_score_thresh=0.2)
    # clt.send_examples()
