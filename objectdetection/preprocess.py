import torch
from PIL import Image, ImageDraw, ImageFont
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


def build_groundval(val_ind, val_merged):

  val_library = {}

  for ind in val_ind:

    val_merged_subset = val_merged[val_merged.filename == ind]

    labels = val_merged_subset['class'].tolist()
    boxes = val_merged_subset.apply(lambda x: [x['xmin'], x['ymin'], 
                                              x['xmax'], x['ymax']], axis=1).tolist()

    if ind in val_library.keys():
      val_library[ind]["ground_truth"]["boxes"].extend(boxes)
      val_library[ind]["ground_truth"]["labels"].extend(labels)

    else:
      val_library[ind] = {"ground_truth": {"boxes": [], "labels": []}}

      val_library[ind]["ground_truth"]["boxes"].extend(boxes)
      val_library[ind]["ground_truth"]["labels"].extend(labels)

  return(val_library)


def build_ground_truth(val_ind, val_library):
    for ind in val_ind:

        fileinfo = val_library[ind]['ground_truth']


        path = f"val/{ind}"
        im = Image.open(path).convert('RGB')

        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()

        labels = ['None', 'rebar', 'spall', 'crack']
        colors = ['blue', 'red', 'black', 'white']

        idx2label = {v: k for v, k in enumerate(labels)}
        label2idx = {k: v for v, k in enumerate(labels)}

        for i, box in enumerate(fileinfo['boxes']):
            label = fileinfo['labels'][i]

            if label == "None":
                continue

            draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline=colors[label2idx[label]], width=3)
            draw.text((box[0] + 20, box[1] + 20), label, 
                        font=font, fill=colors[label2idx[label]])
            
            im.save(f"groundviz/{ind}")



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

    for i, cnt in enumerate(contours):

        if invert:
            if i != len(contours) - 1:
                continue

        x, y, w, h = cv2.boundingRect(cnt) 
        boxes.append([x, y, w, h])

        coords = []
        for point in cnt: 
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        segmentation.append(coords)


    b.append(boxes)
    s.append(segmentation)

image_files = glob2.glob('/root/Documents/capstone/dataset/images/*.jpg')
mask_files = glob2.glob('/root/Documents/capstone/dataset/masks/*.jpg')

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
        image = "/root/Documents/capstone/dataset/masks/" + anno_files[i]

        
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
        shutil.copy("/root/Documents/capstone/dataset/images/" + eachimage, 'train/')

    for eachimage in val_files:
        shutil.copy("/root/Documents/capstone/dataset/images/" + eachimage, 'val/')


    files_100 = glob2.glob('val/*.jpg')[:100]

    for eachimage in files_100:
        shutil.copy("/root/Documents/capstone/DETR/objectdetection/" + eachimage, '100/')


    train_df = merged_df[merged_df['filename'].isin(train_files)]
    val_df = merged_df[merged_df['filename'].isin(val_files)]

    make_coco('train_coco.json', train_df)
    make_coco('val_coco.json', val_df)
    

    val_files = [x.split('/')[-1] for x in glob2.glob('100/*.jpg')]
    val_merged = merged_df[merged_df.filename.isin(val_files)][['filename', 'class', "xmin",	"ymin",	"xmax",	"ymax"]]
    val_ind = val_merged['filename'].value_counts().index

    val_library = build_groundval(val_ind, val_merged)

    build_ground_truth(val_ind, val_library)


if __name__ == '__main__':
    preprocess()