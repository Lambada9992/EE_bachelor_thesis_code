import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
print("1")
# Load the SSD MobileNet object detection model from TensorFlow Hub
model = hub.load('https://tfhub.dev/tensorflow/efficientdet/d7/1')

print("2")
# Load the COCO 2017 validation dataset from TensorFlow Datasets
dataset, info = tfds.load('coco/2017', split='validation', with_info=True)

print("3")
input_size = model.signatures['serving_default'].inputs[0].shape[1:3]

# Create the input pipeline for the validation dataset
def preprocess_data(data):
    image = data['image']
    size = tf.constant(input_size, dtype=tf.int32)
    image = tf.image.resize(image, size)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image, data['objects']['bbox'], data['objects']['label']

dataset = dataset.map(preprocess_data)
dataset = dataset.batch(32)

# Define the evaluation metrics for the model
metrics = [
    tf.keras.metrics.Mean('AP', dtype=tf.float32),
    tf.keras.metrics.Mean('AP50', dtype=tf.float32),
    tf.keras.metrics.Mean('AP75', dtype=tf.float32),
    tf.keras.metrics.Mean('APs', dtype=tf.float32),
    tf.keras.metrics.Mean('APm', dtype=tf.float32),
    tf.keras.metrics.Mean('APl', dtype=tf.float32),
]

# Perform inference on the validation dataset and compute the mAP metrics
print("4")
for images, labels, _ in dataset:
    detections = model(images, training=False)
    bboxes = detections['detection_boxes'].numpy()
    scores = detections['detection_scores'].numpy()
    classes = detections['detection_classes'].numpy()
    num_detections = detections['num_detections'].numpy()
    for i, bbox in enumerate(bboxes):
        bbox = bbox[:num_detections[i]]
        scores_i = scores[i][:num_detections[i]]
        classes_i = classes[i][:num_detections[i]]
        indices = tf.image.non_max_suppression(
            bbox, scores_i, max_output_size=100, iou_threshold=0.5, score_threshold=0.05)
        detections = {
            'detection_boxes': tf.gather(bbox, indices),
            'detection_scores': tf.gather(scores_i, indices),
            'detection_classes': tf.gather(classes_i, indices),
            'num_detections': tf.shape(indices)[0]
        }
        for j, metric in enumerate(metrics):
            metric(detections['detection_boxes'], detections['detection_scores'], detections['detection_classes'])

print(5)

# Print the mAP metrics
mAP = metrics[0].result().numpy()
mAP50 = metrics[1].result().numpy()
mAP75 = metrics[2].result().numpy()
mAPs = metrics[3].result().numpy()
mAPm = metrics[4].result().numpy()
mAPl = metrics[5].result().numpy()

print(f'{mAP}, {mAP50}, {mAP75}, {mAPs}, {mAPm}, {mAPl}')
