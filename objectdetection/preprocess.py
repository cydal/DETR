#from torchvision.ops import masks_to_boxes
#from torchvision.io import read_image
import torch
from PIL import Image
import cv2
import os

import click
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


import glob2
import pandas as pd
import numpy as np
import json
import pandas as pd
from create_coco import make_coco 

import shutil
from sklearn.model_selection import train_test_split

def get_label(name):
  lab = ""
  for l in labs:
    if l in name:
      lab = l

  return(lab)

def get_contour(thresh, h, w, b, s, c, i, f, n, a, invert=False):
    if invert:
        thresh = np.invert(thresh)
        c.append('None')
    else:
        c.append(labels[i])
    
    f.append(fileids[i])
    n.append(fnames[i])
    a.append(anno_files[i])

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    height, width = thresh.shape
    h.append(height)
    w.append(width)

    boxes = []
    segmentation = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) # [[23, 23, 33, 44]]
        boxes.append([x, y, w, h])

        coords = []
        for point in cnt: # [x, y, x, y, x, y]
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        segmentation.append(coords)


    b.append(boxes)
    s.append(segmentation)

image_files = glob2.glob('dataset/images/*.jpg')
mask_files = glob2.glob('dataset/masks/*.jpg')

labs = ['rebar', 'spall', 'crack']

fileids, fnames, anno_files, labels = [], [], [], []
bboxes, segmentations = [], []
classes, heights, widths = [], [], []
fileids_, fnames_, anno_files_ = [], [], []


def preprocess():

    for f in mask_files:
        anno_file = f.split('/')[-1]
        id_label = anno_file.split('.')[0]

        label = get_label(anno_file)
        file_id = id_label.replace(label, "")

        fileids.append(file_id)
        fnames.append(file_id + ".jpg")
        anno_files.append(anno_file)
        labels.append(label)

    j = 0
    for i in range(len(fileids)):
        image = "dataset/masks/" + anno_files[i]

        
        src = cv2.imread(image, 0)
        ret, thresh = cv2.threshold(src, 127, 255, 0)

        get_contour(thresh, heights, widths, bboxes, segmentations, classes, i, 
                    fileids_, fnames_, anno_files_)
        get_contour(thresh, heights, widths, bboxes, segmentations, classes, i, 
                    fileids_, fnames_, anno_files_, invert=True)


    capstone_df = pd.DataFrame({
        "fileid": fileids_,
        "filename": fnames_,
        "anno_files": anno_files_,
        "height": heights,
        "width": widths,
        "bboxes": bboxes,
        "segmentations": segmentations,
        "class": classes
    })

    print(capstone_df.head())

    melt_df_boxes = capstone_df['bboxes'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 
                                                                                                'value']].set_index('index')
    melt_df_segment = capstone_df['segmentations'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 
                                                                                                'value']].set_index('index')

    merged_df = pd.merge(capstone_df, melt_df_boxes, left_index=True, right_index=True)
    merged_df = pd.merge(merged_df, melt_df_segment, left_index=True, right_index=True)

    print(merged_df.head())

    merged_df[['xmin', 'ymin', 'xmax','ymax']] = pd.DataFrame(merged_df.value_x.tolist(), index= merged_df.index)
    merged_df = merged_df.drop(['bboxes', 'fileid', 'value_x',
                                'segmentations', 'anno_files'], axis=1)
    merged_df.rename(columns = {'value_y':'segmentation'}, inplace = True)

    allfiles = merged_df['filename'].tolist()
    allfiles = list(set(allfiles))
    split = int(len(allfiles) * 0.8)
    train_files, val_files = allfiles[:split], allfiles[split:]

    for eachimage in train_files:
        shutil.copy("dataset/images/" + eachimage, 'train/')

    for eachimage in val_files:
        shutil.copy("dataset/images/" + eachimage, 'val/')


    train_df = merged_df[merged_df['filename'].isin(train_files)]
    val_df = merged_df[merged_df['filename'].isin(val_files)]

    make_coco('train_coco.json', train_df)
    make_coco('val_coco.json', val_df)

if __name__ == '__main__':
    preprocess()