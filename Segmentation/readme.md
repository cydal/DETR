# Panoptic Segmentation

Following the Bounding Box detection, we add the mask head to obtain Panoptic segmentation.

> python /root/detr/main.py --masks --epochs 60 --lr_drop 15 --coco_path .  --coco_panoptic_path . --dataset_file coco_panoptic --frozen_weights /root/DETR/Detection/detection_output/checkpoint.pth --output_dir /root/DETR/Detection/segm_output


The output of detection are passed as object queries to the segmentation model. The result of this is a mask for the class of the object query. 

[![image.png](https://i.postimg.cc/nLJqjBxn/image.png)](https://postimg.cc/pmsm39yS)

The maps will be of size H/32 x W/32, which will be upsampled by the model. 

[![image.png](https://i.postimg.cc/rF5G9MJV/image.png)](https://postimg.cc/N5frshSS)

## Some Segmentations

