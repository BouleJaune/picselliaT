import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
from IPython.display import display
from object_detection.utils import label_map_util
import pandas as pd



PATH_TO_LABELS = 'MaskRCNN-inception/label_map.pbtxt'
img_path = "test.jpg"
model_path = "MaskRCNN-inception/model/saved_model"
dim = (426, 640)
def infer(list_files, model_path):
    predict_fn = tf.contrib.predictor.from_saved_model(model_path)
    with tf.Session() as sess:
        for file_name in list_files:
            img_path = "MaskRCNN-inception/Images/"+file_name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = np.expand_dims(img, 0)
            output_dict = predict_fn({"inputs": img_tensor})
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



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
            
            boxes = output_dict["detection_boxes"]
            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            masks = output_dict.get('detection_masks_reframed', None)

            b = []
            c = []
            s = []
            m = []
            k = 0
            for classe in classes:
                if classe==1 or classe==2:
                    b.append(boxes[k])
                    c.append(classe)
                    s.append(scores[k])
                    m.append(masks[k])
                k+=1

            boxes = np.array(b)
            classes = np.array(c)
            scores = np.array(s)
            masks = np.array(m)

            vis_util.visualize_boxes_and_labels_on_image_array(img, 
                                                boxes,
                                                classes,
                                                scores,
                                                category_index,
                                                instance_masks=masks,
                                                use_normalized_coordinates=True,
                                                line_thickness=7)

            Image.fromarray(img).show()



list_files = [
    "architectural-design-architecture-ceiling-chairs-380768.jpg",
    "architecture-book-shelves-bookcase-chairs-245240.jpg",
    "chairs-daylight-designer-empty-416320.jpg",
    "group-of-people-watching-on-laptop-1595385.jpg",
    "orange-excavator-on-brown-hill-1116035.jpg",
    "radiography.jpg",
    "three-woman-sitting-on-white-chair-in-front-of-table-2041627.jpg",
    "two_bears.jpg",
    "voitures.jpg",
    "yellow-tractor-in-asphalt-road-771146.jpg"
]

infer(list_files, model_path)