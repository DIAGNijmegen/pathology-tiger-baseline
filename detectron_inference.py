import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import random
import os
import json
import copy
import torch

from wholeslidedata.image.wholeslideimage import WholeSlideImage

import xml.etree.cElementTree as ET
from xml.dom import minidom
from write_annotations import write_point_set 
from utils import px_to_mm
from nms import to_wsd

SIZE = 128
AUG = T.FixedSizeCrop((SIZE, SIZE),  pad_value=0)
inv_label_map = {
    0: "til",
}

def transform(image):
    image = AUG.get_transform(image).apply_image(image)
    return image
    
class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""
    
    
    def __call__(self, images):
        
        input_images = []
        for image in images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            new_image = transform(image)
            new_image = torch.as_tensor(new_image.astype("float32").transpose(2, 0, 1))
    
            input_images.append({"image": new_image, "height": height, "width": width})
        
        with torch.no_grad():
            preds = self.model(input_images)
        return preds
    
    
class Detectron2DetectionPredictor:
    def __init__(self, weights_path, output_dir, threshold, nms_threshold):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        )

        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            512  
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        cfg.OUTPUT_DIR = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f'loaded model {os.path.join(cfg.OUTPUT_DIR, "model_final.pth")}')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        cfg.MODEL.RPN.NMS_THRESH = nms_threshold

        self._predictor = BatchPredictor(cfg)
        
    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(
            x_batch
        )  
        predictions = []
        for output in outputs:
            predictions.append([])
            pred_boxes = output["instances"].get("pred_boxes")

            scores = output["instances"].get("scores")
            classes = output["instances"].get("pred_classes")
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                confidence = scores[idx].cpu().detach().numpy()
                label = inv_label_map[int(classes[idx].cpu().detach())]
                prediction_record = {'x': int(x), 'y': int(y), 'label': str(label), 'confidence': float(confidence)}
                predictions[-1].append(prediction_record)
        return predictions

def inference(iterator, weights_path, model_dir:str, slide_folder: str, prediction_folder:str,  output:str, threshold=0.0, nms_threshold=0.01):

    print('creating predictor...')
    predictor = Detectron2DetectionPredictor(weights_path=weights_path, output_dir=model_dir, threshold=threshold, nms_threshold=nms_threshold)

    print('predicting...')
    output_dict = {"type": 'Multiple points',
                   "version": {"major": 1,
                               "minor": 0},
                   'points': []
                  }
    
    current_image = None
    annotations = []
    boxes = []
    
    for x_batch, y_batch, info in iterator:
        
        filekey = info['sample_references'][0]['reference'].file_key
        if current_image != filekey:
                        
            current_image = filekey
            slide = slide_folder + filekey + '.tif'
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(0.5)]
                spacing = wsi.get_real_spacing(0.5)
        
        predictions = predictor.predict_on_batch(x_batch)
        for idx, prediction in enumerate(predictions):
            point = info["sample_references"][idx]["point"]
            c, r = point.x, point.y
            
            for detections in prediction:
                x, y, label, confidence = detections.values()
                
                if y_batch[idx][y][x] == 0:
                    continue
                x += c
                y += r
                prediction_record = {'point': [px_to_mm(x, spacing), px_to_mm(y, spacing), 0.5009999871253967], 'probability': confidence}
                output_dict['points'].append(prediction_record)
                annotations.append((x,y))
                
    print(f'Predicted {len(annotations)} points')
    print('saving predictions...')
    
    annotations = to_wsd(annotations)

    write_point_set(annotations, 
            f'{prediction_folder}{current_image}'+'.xml',
            label_name='lymphocytes',
            label_color='blue')

    output_path = f'/output/detected-lymphocytes.json'
    with open(output_path, 'w') as outfile:
         json.dump(output_dict, outfile, indent=4)

    print('finished!')