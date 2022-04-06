# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import glob2
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import datasets.transforms as T
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from PIL import Image, ImageDraw, ImageFont

import albumentations as A
from albumentations.pytorch import ToTensor

import os
import pandas as pd
from torchvision.io import read_image

from torchvision.io import read_image
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as T

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


labels = ['spall', 'background', 'crack', 'rebar']
coco_mean = (0.485, 0.456, 0.406)
coco_std = (0.229, 0.224, 0.225)
colors = ['black', 'blue', 'white', 'red']

# MEAN, STD - Transform
def get_transform(MEAN, STD):

    test_transform = A.Compose([
                                A.Resize(800, 600),
                                A.Normalize(mean=MEAN, std=STD),
                                ToTensor()
    ])
    return(test_transform)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--evaluation', default="", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=210, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--num_classes', default=4, type=int,
                        help='#classes in your dataset, which can override the value hard-coded in file models/detr.py')
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        print("resume")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])



    ## Path to evaluation images
    images = glob2.glob(args.evaluation + '/*.jpg')


    model_without_ddp.eval()
    transform = get_transform(coco_mean, coco_std)

    for im in images:

      name = im.split('/')[-1]
      im = Image.open(im).convert('RGB')
      im_array = np.array(im)

      img = transform(image=im_array)['image'].unsqueeze(0)
      img = img.to(device)

      output = model_without_ddp(img)

      # Get max predictions / filter by confidence
      probas = output['pred_logits'].softmax(-1)[0, :, :-1]
      keep = probas.max(-1).values > 0.8

      # filter predictions by keep idx
      outputs_boxes = output['pred_boxes'][0, keep]
      outputs_pred = output['pred_logits'][0, keep]

      # Rescale bbox pred to original size
      bboxes_scaled = rescale_bboxes(outputs_boxes.cpu(), im.size)


      draw = ImageDraw.Draw(im)
      font = ImageFont.load_default()


      for i, box in enumerate(bboxes_scaled.tolist()):

        pred_idx = outputs_pred[i].argmax(-1).item()
        if pred_idx != 1:
          draw.rectangle(box, outline=colors[pred_idx], width=3)
          draw.text((box[0] + 20, box[1] + 20), labels[pred_idx], 
                    font=font, fill=colors[pred_idx])

      name = args.output_dir + "/" + name 

      im.show()
      im.save(name)
      
    images = glob2.glob('100/*.jpg')
    imgss = [x.split('/')[-1] for x in images]
    paths = ["100/", "groundviz/", "pred_output/"]

    viz_files = []
    for img in imgss:
        viz_files.append(f"{paths[0]}{img}")
        viz_files.append(f"{paths[1]}{img}")
        viz_files.append(f"{paths[2]}{img}")

    images_list = [T.Resize((300, 300))(read_image(x)) for x in viz_files]

    grid = make_grid(images_list, nrow=3)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(f"grid.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
