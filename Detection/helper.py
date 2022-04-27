import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


labs = ['rebar', 'spall', 'crack']
tags = ['labels', 'heights', 'widths', 'bboxes', 
        'segmentations', 'fileids_', 'fnames_']

lab_colors = {
    'spall': [220, 20, 60], 
    'None': [119, 11, 32], 
    'crack': [73, 77, 174], 
    'rebar': [107, 255, 200]
}


def check_exists(file_id, mask_path):
  avail_labels = []

  for lab in labs:
    file_name = f"{file_id}{lab}.jpg"
    
    if os.path.exists(f"{mask_path}/{file_name}"):
      avail_labels.append(lab)

  return(avail_labels)



  ### segments intersection


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode



def image_coco(row):
    image = {}
    image["height"] = row.height
    image["width"] = row.width
    image["id"] = row.fileid
    image["file_name"] = row.filename
    return image

def category_coco(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid

    category["name"] = row[5]
    category["isthing"] = 1 if row[5] in labs else 0
    category["color"] = lab_colors[row[5]]
    return category

def annotation(row):
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)

    annotation["segmentation"] = row.segmentations
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    annotation["bbox"] = [row.xmin, row.ymin, row.xmax, row.ymax]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation



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