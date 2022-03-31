"""
Custom dataset.

Mostly copy-paste from coco.py
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} to custom dataset does not exist'
    PATHS = {
        "train": ("/home/cydal/Documents/others/tempfolder/train", "/home/cydal/Documents/others/tempfolder/train_coco.json"),
        "val": ("/home/cydal/Documents/others/tempfolder/val", "/home/cydal/Documents/others/tempfolder/val_coco.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
