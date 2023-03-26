import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the COCO 2017 validation dataset
coco_val_ds = tfds.load('coco/2017', split='validation', shuffle_files=False)

# Load the SSD MobileNet v2 object detection model from TensorFlow Hub
model_url = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
model = hub.load(model_url)

# Create a COCO object for the validation dataset annotations
coco_val = COCO()

# Initialize lists for holding detection results and ground truth annotations
results = []
annotations = []

# Loop over the validation dataset and perform inference with the model
id = 1
for example in coco_val_ds:
    print(f"LEGIA {id}")
    if id >= 10:
        break
    image = tf.expand_dims(example['image'], axis=0)
    image_id = int(example['image/id'].numpy()[0])
    pred = model(image)['detection_boxes'][0].numpy()
    scores = model(image)['detection_scores'][0].numpy()
    labels = model(image)['detection_classes'][0].numpy().astype(np.uint8) + 1
    num_detections = int(model(image)['num_detections'][0])
    pred = pred[:num_detections]
    scores = scores[:num_detections]
    labels = labels[:num_detections]
    for i in range(num_detections):
        results.append({
            'image_id': image_id,
            'category_id': int(labels[i]),
            'bbox': [float(pred[i][1]), float(pred[i][0]), float(pred[i][3] - pred[i][1]), float(pred[i][2] - pred[i][0])],
            'score': float(scores[i])
        })

    # Get the ground truth annotations for this image
    ann_ids = coco_val.getAnnIds(imgIds=image_id)
    anns = coco_val.loadAnns(ann_ids)
    for ann in anns:
        annotations.append({
            'image_id': image_id,
            'category_id': int(ann['category_id']),
            'bbox': ann['bbox'],
            'iscrowd': int(ann['iscrowd'])
        })
    id = id+1

# Compute the COCO metrics
coco_results = coco_val.loadRes(results)
coco_eval = COCOeval(coco_val, coco_results, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print the mAP scores
print('COCO mAP: {:.2f}'.format(coco_eval.stats[0]))