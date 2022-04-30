# Panoptic Segmentation

## Data Preparation

The panoptic API library https://github.com/cocodataset/panopticapi was used to convert from detection format to panoptic format. 

> python /root/panopticapi/converters/detection2panoptic_coco_format.py --input_json_file valcoco.json --output_json_file panoptic_valcoco.json

> python /root/panopticapi/converters/detection2panoptic_coco_format.py --input_json_file traincoco.json --output_json_file panoptic_traincoco.json

PNG files were generated per image to store the image segmentation. 

[![image.png](https://i.postimg.cc/vmtKBCTh/image.png)](https://postimg.cc/xkq5pxKN)

[![image.png](https://i.postimg.cc/zvc6NWv1/image.png)](https://postimg.cc/fSYBjVYH)

## DETR for Panoptic Segmentation
Following the Bounding Box detection, we add the mask head to obtain Panoptic segmentation.

> python /root/detr/main.py --masks --epochs 60 --lr_drop 15 --coco_path .  --coco_panoptic_path . --dataset_file coco_panoptic --frozen_weights /root/DETR/Detection/detection_output/checkpoint.pth --output_dir /root/DETR/Detection/segm_output


The output of detection are passed as object queries to the segmentation model. The result of this is a mask for the class of the object query. 

[![image.png](https://i.postimg.cc/nLJqjBxn/image.png)](https://postimg.cc/pmsm39yS)

The maps will be of size H/32 x W/32, which will be upsampled by the model. 

[![image.png](https://i.postimg.cc/rF5G9MJV/image.png)](https://postimg.cc/N5frshSS)

## Prediction

[![image.png](https://i.postimg.cc/Hk6VHhzD/image.png)](https://postimg.cc/4YhJQwjB)

[![image.png](https://i.postimg.cc/FzDFVf7d/image.png)](https://postimg.cc/QKKrXdgh)


[![image.png](https://i.postimg.cc/sXTbn54r/image.png)](https://postimg.cc/MMjsHMLP)

[![image.png](https://i.postimg.cc/bYFKkHv0/image.png)](https://postimg.cc/w3LfdNL3)


[![image.png](https://i.postimg.cc/nzW0j21C/image.png)](https://postimg.cc/tZhh8hwj)

[![image.png](https://i.postimg.cc/Xqyxw52x/image.png)](https://postimg.cc/Zv4rmWzy)

[![image.png](https://i.postimg.cc/hvr880LL/image.png)](https://postimg.cc/0KzJPDHj)

[![image.png](https://i.postimg.cc/PJhz87wf/image.png)](https://postimg.cc/G87sWqK6)

[![image.png](https://i.postimg.cc/XY8JVzVc/image.png)](https://postimg.cc/YvvkR8Lh)

[![image.png](https://i.postimg.cc/8kJjh4HN/image.png)](https://postimg.cc/xqQfQKh4)

[![image.png](https://i.postimg.cc/yYn4W4Zd/image.png)](https://postimg.cc/tY1mmcDQ)

[![image.png](https://i.postimg.cc/XqkMwt1s/image.png)](https://postimg.cc/H8nPmvgM)

[![image.png](https://i.postimg.cc/PqVm2MRh/image.png)](https://postimg.cc/DWsJz1hY)

[![image.png](https://i.postimg.cc/sXB58ZkW/image.png)](https://postimg.cc/ftN3SkTW)
