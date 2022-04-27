import os
import cv2
import json
import click
import torch
import pylab
#import scipy
import glob2
import shutil
import numpy as np
import pandas as pd
from helper import *
#import skimage.io as io
from panopticapi import utils
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision.io import read_image
from pycocotools import mask as mask_util
from torchvision.ops import masks_to_boxes
from PIL import Image, ImageDraw, ImageFont
from pycocotools import _mask as _mask_util
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--mask_path', default='/root/Data/dataset/masks', help='Path to Masks')
@click.option('--data_path', default='/root/Data/dataset/images', help='Path to Images')
@click.option('--panoptic_path', default='.', help='Path to Panoptic')
@click.option('--detection_path', default='.', help='Path to Detection')

def preprocess(mask_path, data_path, panoptic_path, detection_path):
    """Generating detectiion & Panoptic info from images"""

    fileids, fnames = [], []
    output_dict = {k: [] for k in tags}
    image_files = glob2.glob(f'{data_path}/*.jpg')

    print("Length of Image Files -- ", len(image_files))
    for f in image_files:
        filename = f.split('/')[-1]
        fileid = filename.split('.')[0]

        fnames.append(filename)
        fileids.append(fileid)


    for i in range(len(fileids)):

        fileid = fileids[i]
        fname = fnames[i]

        #print('"""""""""""""""""""""""""""""""""""""""""""')
        #print(f"ID & Name are {fileid} & {fname}")

        image = cv2.imread(f'{data_path}/{fname}')
        height, width, _ = image.shape
        empty_mask = np.zeros((height, width))

        #print(f"Height & Width are {height} & {width}")

        # eg [spall, rebar, etc...]
        labels_check = check_exists(fileid, mask_path)

        #print(f"Output of Label Check. Length {len(labels_check)} & {labels_check}")

        # No mask found
        if len(labels_check) == 0:
            #print("No Mask Detected ")
            continue
        
        # idx, spall
        for lab_idx, each_label in enumerate(labels_check):

            #print(f"Lab Index & Label {lab_idx} & {each_label}")
            image_path = f"{mask_path}/{fileid}{each_label}.jpg"
            src = cv2.imread(image_path, 0)
            ret, thresh = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)

            num_labels, con_labels = cv2.connectedComponents(thresh)


            #print(f"Connected Comp {num_labels}")
            for num_label_idx in range(1, num_labels):

                label_mask = con_labels == num_label_idx
                label_mask = label_mask.astype(np.uint8)
                empty_mask += label_mask

                encoded = Image.fromarray(label_mask)
                encoded = np.asfortranarray(encoded)
                encoded = mask_util.encode(encoded)

                encoded['counts'] = encoded['counts'].decode('utf-8')

                bbox = mask_util.toBbox(encoded).tolist()

                output_dict['labels'].append(each_label)
                output_dict['bboxes'].append(bbox)
                output_dict['segmentations'].append(encoded)
                output_dict['widths'].append(width)
                output_dict['heights'].append(height)

                output_dict['fileids_'].append(fileid)
                output_dict['fnames_'].append(fname)

        # Repeated --- Fix this
        image_bg = np.invert(empty_mask.astype(np.uint8)) / 255
        encoded = np.asfortranarray(image_bg.astype(np.uint8))
        encoded = mask_util.encode(encoded)
        encoded['counts'] = encoded['counts'].decode('utf-8')
        bbox = [0, 0, width, height]

        output_dict['bboxes'].append(bbox)
        output_dict['segmentations'].append(encoded)
        output_dict['widths'].append(width)
        output_dict['heights'].append(height)

        output_dict['fileids_'].append(fileid)
        output_dict['fnames_'].append(fname)
        output_dict['labels'].append('None')

        assert(boundary_iou(empty_mask.astype('float32'), image_bg.astype('float32')) == 0, "Overlap Detected")
        #print('"""""""""""""""""""""""""""""""""""""""""""')

    for key in output_dict.keys():
        print(f'{key} -- {len(output_dict[key])}')

    capstone_df = pd.DataFrame({
        "fileid": output_dict['fileids_'],
        "filename": output_dict['fnames_'],
        "height": output_dict['heights'],
        "width": output_dict['widths'],
        "bboxes": output_dict['bboxes'],
        "segmentations": output_dict['segmentations'],
        "class": output_dict['labels']
    })

    print(capstone_df.head())

    capstone_df[['xmin', 'ymin', 'xmax','ymax']] = pd.DataFrame(capstone_df.bboxes.tolist(), index= capstone_df.index)
    capstone_df = capstone_df.drop(['bboxes', 'fileid'], axis=1)
    capstone_df.rename(columns = {'value_y':'segmentation'}, inplace = True)
    print(capstone_df)


    allfiles = capstone_df['filename'].tolist()
    allfiles = list(set(allfiles))

    split = int(len(allfiles) * 0.8)


    train_files, val_files = allfiles[:split], allfiles[split:]

    for eachimage in train_files:
        shutil.copy(f'{data_path}/{eachimage}', 'train/')

    for eachimage in val_files:
        shutil.copy(f'{data_path}/{eachimage}', 'val/')

    train_df = capstone_df[capstone_df['filename'].isin(train_files)]
    val_df = capstone_df[capstone_df['filename'].isin(val_files)]
    datasets = [train_df, val_df]


    for d, dataset in enumerate(['train', 'val']):

        save_json_path = f'{dataset}coco.json'
        data = datasets[d]

        images = []
        categories = []
        annotations = []

        category = {}
        category["supercategory"] = 'none'
        category["id"] = 0
        category["name"] = 'None'
        category["isthing"] = 0

        data['fileid'] = data['filename'].astype('category').cat.codes
        data['categoryid']= pd.Categorical(data['class'], ordered= True).codes
        data['categoryid'] = data['categoryid']+1
        data['annid'] = data.index

        for row in data.itertuples():
            annotations.append(annotation(row))

        imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        for row in imagedf.itertuples():
            images.append(image_coco(row))

        catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
        for row in catdf.itertuples():
            categories.append(category_coco(row))

        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations

        json.dump(data_coco, open(save_json_path, "w"), indent=4)



if __name__ == '__main__':
    preprocess()