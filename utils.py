import os
import numpy as np
import yaml
import json 
import glob
from pathlib import Path
from typing import List, Dict


"""General utility"""
def match_by_name(imagenames:List[str], masknames: List[str], x_suffix='.tif', y_suffix='_tissue.tif'):
    """Match files by their name ingoring suffixes."""
    new_imagenames, new_masknames = [], []
    for x in imagenames:
        for y in masknames:
            if x == y or x[:-len(x_suffix)] == y[:-len(y_suffix)]:
                new_imagenames.append(x)
                new_masknames.append(y)
    return new_imagenames, new_masknames

def slide_to_yml(imagefolder: str, 
                   annotationfolder: str,
                   slide: str,
                   annotation: str,
                   folder='config', 
                   name='slidingwindowdata'):

    """Convert an individual slide to yml file"""
                 
    imagefiles = [imagefolder+slide]
    annotationfiles = [annotationfolder+annotation]

    with open(f"{folder}/{name}.yml", "w") as f:
        print('training:', file=f)
        for x, y in zip(imagefiles, annotationfiles):
                space = ' ' 
                print(f'{space*4}- wsi: \n {space*8}path: "{x}"', file=f)
                print(f'{space*4}  wsa: \n {space*8}path: "{y}"', file=f)

def folders_to_yml(imagefolder: str, 
                   annotationfolder: str,
                   folder='config', 
                   name='slidingwindowdata'):

    """
    Generate a yaml file to be used as WSD dataconfig from a folder of slides and a folder of annotation or mask files.
    Assumes files use the same name for both the slides and masks.
    """
    
    imagefiles = glob.glob(imagefolder)
    annotationfiles = glob.glob(annotationfolder)

    if len(imagefiles) != len(annotationfiles):
        imagefoldername, annotationfoldername = imagefolder[:-5], annotationfolder[:-5]
        imagefiles = [x.replace(f'{imagefoldername}', '') for x in imagefiles]
        annotationfiles = [x.replace(f'{annotationfoldername}', '') for x in annotationfiles]
        imagefiles, annotationfiles = match_by_name(imagefiles, annotationfiles)
        imagefiles = [imagefoldername+x for x in imagefiles]
        annotationfiles = [annotationfoldername+x for x in annotationfiles]

    with open(f"{folder}/{name}.yml", "w") as f:
        print('training:', file=f)
        for x, y in zip(imagefiles, annotationfiles):
                space = ' ' 
                print(f'{space*4}- wsi: \n {space*8}path: "{x}"', file=f)
                print(f'{space*4}  wsa: \n {space*8}path: "{y}"', file=f)

"""conversion"""

def dist_to_px(dist, spacing):
    """ distance in um (or rather same unit as the spacing) """
    dist_px = int(round(dist / spacing))
    return dist_px

def mm2_to_px(mm2, spacing):
    return (mm2*1e6) / spacing**2
    
def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def px_to_um2(px, spacing):
    area_um2 = px*(spacing**2)
    return area_um2

"""Segmentation utility"""
def one_hot_decoding_batch(y_batch):
    return np.argmax(y_batch, axis=3) + 1

def get_centerpoints(point, scalar=4.0, output_size=1030):
    c, r = point.x-output_size//scalar, point.y-output_size//scalar
    return c, r 
