import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr

# Load the SSD MobileNet V2 model from the H5 file
model = tf.keras.models.load_model('../../models/ssd_mobilenet_v2_2/my_model.h5')

# Load the COCO 2017 validation dataset from TensorFlow Datasets
dataset, info = tfds.load('coco/2017', split='validation', with_info=True)
num_examples = info.splits['validation'].num_examples

# Define the ranking metric for evaluating mAP
# metric = tfr.metrics.MeanAveragePrecision(num_queries=num_examples)
metric = tfr.keras.metrics.MeanAveragePrecisionMetric()

# Define the input pipeline for preprocessing the dataset
batch_size = 32
dataset = dataset.batch(batch_size)
dataset = dataset.map(lambda x: (x['image'], x['objects']['bbox'], x['objects']['label_id']))
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Evaluate the model on the dataset using the ranking metric
results = model.evaluate(dataset, verbose=1, steps=num_examples // batch_size, metrics=[metric])

# Print the mAP score
print('mAP:', results[1])
