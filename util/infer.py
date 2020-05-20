import tensorflow as tf
import cv2
import numpy as np
import os
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
from IPython.display import display
from object_detection.utils import label_map_util
import pandas as pd
import random




def infer(path_list, exported_model_dir, label_map_path, results_dir, num_infer=5, min_score_thresh=0.7):
    '''saved_model must be saved with input_type = "image_tensor"
    '''
    saved_model_path = exported_model_dir+"saved_model/"
    predict_fn = tf.contrib.predictor.from_saved_model(saved_model_path)
    random.shuffle(path_list)
    path_list = path_list[:num_infer]
    with tf.Session() as sess:
        category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
        for img_path in path_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = np.expand_dims(img, 0)
            output_dict = predict_fn({"inputs": img_tensor})
            
           

            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections] for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

            if 'detection_masks' in output_dict:
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                                            output_dict['detection_masks'], 
                                                            output_dict['detection_boxes'],
                                                            img.shape[0], 
                                                            img.shape[1])      
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)          
                mask_refr = sess.run(detection_masks_reframed)
                output_dict['detection_masks_reframed'] = mask_refr
            masks = output_dict.get('detection_masks_reframed', None)
            boxes = output_dict["detection_boxes"]
            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            

            b = []
            c = []
            s = []
            m = []
            k = 0
            # print(category_index)
            for classe in classes:
                # if classe in category_index.keys():
                b.append(boxes[k])
                c.append(classe)
                s.append(scores[k])
                if masks is not None:
                    m.append(masks[k])
                k+=1
            boxes = np.array(b)
            classes = np.array(c)
            # print(scores)
            scores = np.array(s)
            if masks is not None:
                masks = np.array(m)


            vis_util.visualize_boxes_and_labels_on_image_array(img, 
                                                boxes,
                                                classes,
                                                scores,
                                                category_index,
                                                instance_masks=masks,
                                                use_normalized_coordinates=True,
                                                line_thickness=7,
                                                min_score_thresh=min_score_thresh)

            img_name = img_path.split("/")[-1]
            Image.fromarray(img).save(results_dir+img_name)
            display(Image.fromarray(img))




