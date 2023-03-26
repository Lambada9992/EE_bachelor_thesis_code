import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_ranking as tfr

# Load the validation set
val_data, info = tfds.load("coco/2017", split="validation", with_info=True)

# Load the trained model
# module_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
# model = hub.KerasLayer(module_url, trainable=False)
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Define the evaluation metric
mAP = tfr.keras.metrics.MeanAveragePrecisionMetric()

# Iterate over the dataset and compute the mAP
for example in val_data.batch(1):
    image = example['image']
    groundtruth_labels = example['objects']['label']
    groundtruth_boxes = example['objects']['bbox']

    print("boxes")
    print(groundtruth_boxes)
    print("label")
    print(groundtruth_labels)
    # The model returns a dictionary of predictions

    predictions = model(image)
    print("predictions")
    print(predictions)
    # The `groundtruth_labels` and `groundtruth_boxes` are stored in the dataset examples
    # Check the shape of the predicted detection scores and boxes
    detection_scores_shape = predictions['detection_scores'].shape
    detection_boxes_shape = predictions['detection_boxes'].shape
    # Compute the metric
    # break
    mAP.update_state(groundtruth_boxes, predictions['detection_boxes'].reshape(detection_boxes_shape))

# Compute the final mAP score
final_mAP = mAP.result().numpy()

print("Mean Average Precision (mAP):", final_mAP)