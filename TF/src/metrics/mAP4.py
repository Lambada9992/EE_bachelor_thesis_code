import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load the SSD MobileNet object detection model from TensorFlow Hub
model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

# Load the COCO 2017 validation dataset from TensorFlow Datasets
dataset, info = tfds.load('coco/2017', split='validation', with_info=True)

# Create the input pipeline for the validation dataset
def preprocess_data(data):
    image = data['image']
    # image = tf.image.resize(image, (model.input_shape[1], model.input_shape[2]))
    image = tf.image.resize(image, (320, 320))
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image, data['objects']['bbox'], data['objects']['label']

# dataset = dataset.map(preprocess_data)
# dataset = dataset.batch(32)

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
for data in dataset.batch(1):
    predictions = model(data['image'])
    detections = tf.nest.map_structure(lambda x: x.numpy(), predictions)
    print(detections)
    for i, detection in enumerate(detections):
        # bboxes = detection['detection_boxes']
        bboxes = detection[1]
        # scores = detection['detection_scores']
        scores = detection[4]
        # labels = detection['detection_classes']
        labels = detection[2]
        # num_detections = detection['num_detections']
        num_detections = detection[5]
        indices, _, _ = tf.image.combined_non_max_suppression(
            bboxes, scores, max_output_size_per_class=100, max_total_size=100, iou_threshold=0.5, score_threshold=0.05)
        indices = tf.gather(indices, tf.where(indices[:, 3] > -1)[:, 0])
        detections = {
            'detection_boxes': tf.gather(bboxes, indices[:, 3]),
            'detection_scores': tf.gather(scores, indices[:, 3]),
            'detection_classes': tf.gather(labels, indices[:, 3]),
            'num_detections': tf.shape(indices)[0]
        }
        result = tf.image.combined_non_max_suppression(
            bboxes, scores, max_output_size_per_class=100, max_total_size=100, iou_threshold=0.5, score_threshold=0.05)
        for j, metric in enumerate(metrics):
            metric(result['detection_metrics'][j])

# Print the mAP metrics
mAP = metrics[0].result().numpy()
mAP50 = metrics[1].result().numpy()
mAP75 = metrics[2].result().numpy()
mAPs = metrics[3].result().numpy()
mAPm = metrics[4].result().numpy()
mAPl = metrics[5].result().numpy()
print('mAP: {:.2f}%, mAP50: {:.2f}%, mAP75: {:.2f}%, mAPs: {:.2f}%, mAPm: {:.2f}%, mAPl: {:.2f}%'.format(
    mAP * 100, mAP50 * 100, mAP75 * 100, mAPs * 100, mAPm * 100, mAPl * 100))