## DETR for Panoptic Segmentation



### Strategy

* Convert segmentation annotation to bounding boxes
* Train DETR for object detection on bbox dataset. Also include COCO validation set data (also converted to bbox) to account for stuff classes. 
* Freeze object detection model. 
* Take the original encoded image along with bbox predictions and perform steps detailed below.
* Add segmentation head and train for additional 25 epochs



[![image.png](https://i.postimg.cc/qvZhC8D1/image.png)](https://postimg.cc/nXBcyQJm)

The first step in DETR is to pass an image through a pretrained ResNet50 backbone. This produces the encoded image of dimension (d, H/32, W/32) which is then sent to the Multi-Head Attention layer. 


We take the encoded image (which is the output of the resnet50 backbone), and pass it to a multi-head attention layer. Also passed along are box embeddings that contain the prediction output from the object detection and this  produces attention maps of size (N x M x H/32 x W/32).


During the object detection training, 4 activation layers were set aside when the original image was passed through the resnet50 backbone. Res5 refers to the final activation layer. This block is concatenated with the feature maps from the previous step. 
