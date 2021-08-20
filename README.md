### DETR for Panoptic Segmentation


[![image.png](https://i.postimg.cc/qvZhC8D1/image.png)](https://postimg.cc/nXBcyQJm)

* The first step in DETR is to pass an image through pretrained ResNet50 backbone. This produces the encoded image of dimension (d, H/32, W/32) which is then sent to the Multi-Head Attention. 


* We take the encoded image (which is the output of the resnet50 backbone) and passed that into a multi-head attention layer along with the box embeddings that contain the prediction output from the object detection to produce the attention maps of size (N x M x H/32 x W/32).


* During the object detection training, 4 activation layers were set aside when the original image was passed through  the resnet50 backbone. Res5 refers to the final activation layer. This block is concatenated with the feature maps from the previous step. 
