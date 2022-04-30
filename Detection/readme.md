# Capstone Object Detection

## Concrete Defect

The dataset contains mask labeling of three major types of concrete surface defects: crack, spalling and exposed rebar. The dataset includes images containing binary masks of the target segmentation. Included in the file names is the target class of the segmentation. 

Example - 

00001.jpg
[![image.png](https://i.postimg.cc/NGx4R8Mw/image.png)](https://postimg.cc/YjhYHWYX)

00001rebar.jpg
[![image.png](https://i.postimg.cc/vTjD5Sxz/image.png)](https://postimg.cc/T5VR609L)


## Bounding Boxes
The dataset consists of jpg files containing binary images containing segmentations. The segmentations were converted to bounding boxes using contours. 

```python
ret, thresh = cv2.threshold(src, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)
for cnt in contours:
  x,y,w,h = cv2.boundingRect(cnt)
  img = cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 255, 255), 2)
```

Original Image
[![image.png](https://i.postimg.cc/j2GPRjct/image.png)](https://postimg.cc/zHFVj84c)

Contour
[![image.png](https://i.postimg.cc/Dw5QX7yd/image.png)](https://postimg.cc/YvGmcKFh)

Bbox
[![image.png](https://i.postimg.cc/rsLRHHww/image.png)](https://postimg.cc/KkJYkJT6)


## Stuffs
We need to also obtain bounding boxes around the background. To achieve this, we invert the mask before obtaining bounding boxes. 

```python
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

  height, width = src.shape
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
  ```
  
This is arranged in a Pandas DataFrame
[![image.png](https://i.postimg.cc/9fYH7Wxp/image.png)](https://postimg.cc/R3hYrzx6)


## Train/Val Split

```python
allfiles = merged_df['filename'].tolist()
allfiles = list(set(allfiles))

split = int(len(allfiles) * 0.8)


train_files, val_files = allfiles[:split], allfiles[split:]
```


The train and validation as well as segmentation and bounding box annotations may be viewed here - 


| Dataset | Link |
| ------ | ------ |
| Train | [https://app.activeloop.ai/sijpapi/coco_defect__train][PlDb] |
| Validation | [https://app.activeloop.ai/sijpapi/coco_defect__val][PlGh] |




## Pandas to COCO Annotation Format
```python
def image(row):
    image = {}
    image["height"] = row.height
    image["width"] = row.width
    image["id"] = row.fileid
    image["file_name"] = row.filename
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[4]
    category["isthing"] = 1 if row[4] in labs else 0
    return category

def annotation(row):
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
    annotation["segmentation"] = [row.segmentation]
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    #annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin]
    annotation["bbox"] = [row.xmin, row.ymin, row.xmax, row.ymax]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation

def make_coco(save_json_path, data):
  images = []
  categories = []
  annotations = []

  category = {}
  category["supercategory"] = 'none'
  category["id"] = 0
  category["name"] = 'None'
  category["isthing"] = 0
  #categories.append(category)

  data['fileid'] = data['filename'].astype('category').cat.codes
  data['categoryid']= pd.Categorical(data['class'], ordered= True).codes
  data['categoryid'] = data['categoryid']+1
  data['annid'] = data.index


  for row in data.itertuples():
      annotations.append(annotation(row))

  imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
  for row in imagedf.itertuples():
      images.append(image(row))

  catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
  for row in catdf.itertuples():
      categories.append(category(row))

  data_coco = {}
  data_coco["images"] = images
  data_coco["categories"] = categories
  data_coco["annotations"] = annotations
  json.dump(data_coco, open(save_json_path, "w"), indent=4)
  ```
Reference:
https://stackoverflow.com/questions/62545034/how-to-convert-a-csv-table-into-coco-format-in-python


## Validation Loss Visualization
[![image.png](https://i.postimg.cc/L4JHT9Ty/image.png)](https://postimg.cc/QVrGNrv5)
[![image.png](https://i.postimg.cc/1tYjXZVX/image.png)](https://postimg.cc/ftmKqpKQ)
[![image.png](https://i.postimg.cc/ZYXJsGsJ/image.png)](https://postimg.cc/JtcwG2WF)

## Original |      Ground Truth |            BBox Prediction
![Image Grid](https://github.com/cydal/DETR/blob/main/objectdetection/grid.png)
