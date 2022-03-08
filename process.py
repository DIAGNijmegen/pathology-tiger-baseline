from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.image.wholeslideimagewriter import WholeSlideMaskWriter
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.annotation.generate_polygons import cv2_polygonize
from wholeslidedata.annotation.structures import Polygon
from wholeslidedata.labels import Labels, Label
from wholeslidedata.source.files import File

from hooknet.configuration.config import create_hooknet

import sys 
import glob
import os
import torch
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from pathlib import Path
from shapely.ops import unary_union
import shutil, json

from write_annotations import write_point_set
from detectron_inference import inference
from nms import slide_nms, to_wsd, dist_to_px
from utils import folders_to_yml, slide_to_yml, get_centerpoints, match_by_name
from concave_hull import concave_hull
from write_mask import write_mask

def get_mask_area(slide, spacing=16):
    """Get the size of a mask in pixels where the mask is 1."""
    mask = WholeSlideImage(slide, backend='asap')
    patch = mask.get_slide(spacing)
    counts = np.unique(patch, return_counts=True)
    down = mask.get_downsampling_from_spacing(spacing)
    area = counts[1][1] * down**2
    return area

def build_hooknet(configpath):
    model = create_hooknet(configpath)
    return model

def hooknet_predict(x, model):
    """Transform and predict a batch with Hooknet"""
    x = list(x.transpose(1,0,2,3,4))
    prediction = model.predict_on_batch(x)
    return prediction  

def write_json(data, path):
    path = Path(path)
    with path.open('wt') as handle:
        json.dump(data, handle, indent=4, sort_keys=False)
        
def seg_inference(iterator, model, image_folder, inp_slide=''):
    """Loop trough the tiles in the batch iterator, predict them with Hooknet and write them to a mask"""
    current_image = None
    spacing = 0.5
    tile_size = 1024 

    for x_batch, y_batch, info in iterator:

        filekey = info['sample_references'][0]['reference'].file_key

        if current_image != filekey:

            if current_image:
                wsm_writer.save()

            current_image = filekey
            slide = image_folder + inp_slide
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
                spacing = wsi.get_real_spacing(spacing)

            wsm_writer = WholeSlideMaskWriter()
            wsm_writer.write(path=f'tempoutput/segoutput/{inp_slide}', spacing=spacing, dimensions=shape, tile_shape=(tile_size,tile_size))

        points = [x['point'] for x in info['sample_references']]    
        predictions = hooknet_predict(x_batch, model)

        if len(points) != len(predictions):
            points = points[:len(predictions)]

        for i, point in enumerate(points):
            mask = y_batch[i][0]
            c, r = get_centerpoints(point, 4.0, 1030)
            pred = predictions[i]
            wsm_writer.write_tile(tile=pred, coordinates=(int(c), int(r)), mask=mask) 
    wsm_writer.save()
    
def bulk_inference(bulk_iterator, image_folder):
    """Write stromal tissue within the tumor bulk to a new tissue mask"""
    spacing = 0.5
    tile_size = 512 # always **2 
    current_image = None
    
    for x_batch, y_batch, info in bulk_iterator:
    
        filekey = info['sample_references'][0]['reference'].file_key
        if current_image != filekey:
            if current_image:
                bulk_wsm_writer.save()

            current_image = filekey
            slide = image_folder + filekey + '.tif'
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
                spacing = wsi.get_real_spacing(spacing)

            bulk_wsm_writer = WholeSlideMaskWriter()
            bulk_wsm_writer.write(path=f'tempoutput/detoutput/{filekey}.tif', 
                                  spacing=spacing, 
                                  dimensions=shape, 
                                  tile_shape=(tile_size,tile_size))
            
        points = [x['point'] for x in info['sample_references']]  

        if len(points) != len(x_batch):
            points = points[:len(x_batch)]

        for i, point in enumerate(points):
            c, r = point.x, point.y 
            x_batch[i][x_batch[i] == 1] = 0
            x_batch[i][x_batch[i] == 3] = 0
            x_batch[i][x_batch[i] == 5] = 0
            x_batch[i][x_batch[i] == 4] = 0
            x_batch[i][x_batch[i] == 7] = 0
            new_mask = x_batch[i].reshape(tile_size, tile_size) * y_batch[i] 
            new_mask[new_mask > 0] = 1 
            bulk_wsm_writer.write_tile(tile=new_mask, coordinates=(int(c), int(r)), mask=y_batch[i])

    bulk_wsm_writer.save()
    
def set_tf_gpu_config():
    """Hard-coded GPU limit to balance between tensorflow and Pytorch"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            
def tf_be_silent():
    """Surpress exessive TF warnings"""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as ex:
        print('failed to silence tf warnings:', str(ex))

class TIGERSegDet(object):
    
    def __init__(self, input_folder='/input/',
                       mask_folder='/input/images/',
                       output_folder='/output/', 
                       config_folder='/configs/', 
                       model_folder='/models/'):
        
        self.input_folder = input_folder + '/'
        self.input_folder_masks = mask_folder + '/'
        self.output_folder = output_folder + '/'
        self.seg_config = 'seg-inference-config'
        self.det_config = 'det-inference-config'
        self.bulk_config = 'bulk-inference-config'

        self.model_paramaters_path = self.seg_config+ '/hooknet_params.yml'
        self.det_model_path = 'detmodel/ckp'

    def stroma_to_mask(self, segpath='tempoutput/segoutput/*.tif', bulkpath='tempoutput/segoutput/*.tif'):
        """Create a mask out of the stroma segmentations within the tumor bulk"""
        folders_to_yml(segpath, bulkpath, self.bulk_config)
        bulk_iterator = create_batch_iterator(mode='training',
                                  user_config=f'{self.bulk_config}/slidingwindowconfig.yml',
                                  presets=('slidingwindow',),
                                  cpus=1, 
                                  number_of_batches=-1, 
                                  return_info=True)
    
        bulk_inference(bulk_iterator, "tempoutput/segoutput/")
        bulk_iterator.stop()

    def detection_in_mask(self, slide_file):
        
        slide_to_yml(f'{self.input_folder}', 'tempoutput/detoutput/', slide_file, slide_file, self.det_config)
        iterator = create_batch_iterator(
                mode="training",
                user_config=f'./{self.det_config}/slidingwindowconfig.yml',
                presets=("slidingwindow",),
                cpus=1,
                number_of_batches=-1,
                return_info=True,)

        inference(iterator, 
          'model_final.pth', 
          'detmodel/output_fixed_resize/', 
          f'{self.input_folder}',
          'tempoutput/detoutput/',
          f'{self.output_folder}',
          0.1, 
          0.3)

        iterator.stop()

    def process(self):

        """INIT"""
        set_tf_gpu_config()
        tf_be_silent()
        model = build_hooknet(self.model_paramaters_path)        
        torch.cuda.set_per_process_memory_fraction(0.55, 0)

        """Segmentation inference"""
        slide_file = [x for x in os.listdir(self.input_folder) if x.endswith('.tif')][0]
        tissue_mask_slide_file = [x for x in os.listdir(self.input_folder_masks) if x.endswith('.tif')][0]
        iterator = create_batch_iterator(mode='training',
                                         user_config=f'./{self.seg_config}/slidingwindowconfig.yml',
                                         presets=('slidingwindow', 'folders'),
                                         cpus=1, 
                                         number_of_batches=-1, 
                                         return_info=True)

        seg_inference(iterator, model, self.input_folder, slide_file)
        iterator.stop()
        # K.clear_session()
        # del model 

        try:
            shutil.copyfile(f'tempoutput/segoutput/{slide_file}', f'{self.output_folder}/images/breast-cancer-segmentation-for-tils/{slide_file}')
        except OSError:
            print(f'FileNotFoundError: Could not copy segmentation {slide_file}')
            write_json(0.0, f'{self.output_folder}/til-score.json')
            det_result = dict(type='Multiple points', points=[], version={ "major": 1, "minor": 0 })
            write_json(det_result, f'{self.output_folder}/detected-lymphocytes.json')

        print('Finished segmentation inference')

        """Get Tumor Bulk"""
        concave_hull(input_file=glob.glob('tempoutput/segoutput/*.tif')[0], 
              output_dir='tempoutput/bulkoutput/',
              input_level=6,
              output_level=0,
              level_offset=0, 
              alpha=0.07,
              min_size=1.5,
              bulk_class=1
            )

        # Copy bulk to output folder, if the bulk is not found then we predict in the whole mask to do detection for LB1
        slide_file_xml = f'{slide_file[:-4]}.xml'
        try:
            shutil.copyfile(f'tempoutput/bulkoutput/{slide_file_xml}', f'{self.output_folder}/bulks/{slide_file_xml}')
        except OSError:
            print(f'FileNotFoundError: Could not copy bulk {slide_file_xml}')
            self.stroma_to_mask()
            self.detection_in_mask(slide_file)
            write_json(0.0, f'{self.output_folder}/til-score.json')
            sys.exit(0)
        
        """Write tumor bulk as mask"""
        wsi = WholeSlideImage(f'tempoutput/segoutput/{slide_file}', backend='asap')
        wsa = WholeSlideAnnotation(f'tempoutput/bulkoutput/{slide_file[:-4]}.xml')
        if wsa.annotations:
            write_mask(wsi, wsa, spacing=0.5, suffix='_bulk.tif')
        print('Wrote bulk to mask')

        """Write stroma within bulk to mask"""
        self.stroma_to_mask(segpath='tempoutput/segoutput/*.tif', bulkpath='tempoutput/bulkoutput/*.tif')
        print('Wrote stroma within bulk to mask')

        """Detection inference"""
        self.detection_in_mask(slide_file)

        """slide_level_nms"""
        points_path = f'tempoutput/detoutput/{slide_file[:-4]}.xml'
        slide_path = f'tempoutput/segoutput/{slide_file}'
        points = slide_nms(slide_path, points_path, 256)
        wsd_points = to_wsd(points)

        """Compute TIL score and write to output"""
        til_area = dist_to_px(8, 0.5) ** 2
        tils_area = len(wsd_points) * til_area
        stroma_area = get_mask_area(f'tempoutput/detoutput/{slide_file}')
        tilscore = (100/int(stroma_area)) * int(tils_area) 
        print(f'{slide_file} has a tilscore of {tilscore}')
        write_json(tilscore, f'{self.output_folder}/til-score.json')

if __name__ == '__main__':
    output_folder = '/output/'
    
    Path("tempoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/segoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/detoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/bulkoutput").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/images/breast-cancer-segmentation-for-tils").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/bulks").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection/asap").mkdir(parents=True, exist_ok=True)
    TIGERSegDet().process()



    

    
