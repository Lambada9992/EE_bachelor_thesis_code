# import necessary packages and then download from tensorflow hub the MobileNetV2 model and load it into a Keras model and then prune it on a coco 2017 dataset and then save the model to h5 format

import tensorflow as tf
import tensorflow_model_optimization as tfmo
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# load the coco 2017 dataset
dataset_name = 'coco/2017'
dataset = tfds.load(name=dataset_name, split='train[:80%]')
# Split the dataset into train and validation sets
num_train_examples = int(0.8 * tf.data.experimental.cardinality(dataset).numpy())
train_dataset = dataset.take(num_train_examples)
val_dataset = dataset.skip(num_train_examples)

# load the MobileNetV2 model from tensorflow hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
                     trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(1000, activation='softmax')
])
model.build([None, 224, 224, 3])  # Batch input shape.

# prune the model
pruning_schedule = tfmo.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=2000,
    end_step=4000
)

model_for_pruning = tfmo.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule
)

model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model_for_pruning.fit(train_dataset, epochs=10, validation_data=val_dataset)

model_for_pruning.summary()

# save the model to h5 format
model_for_pruning.save('mobilenetv2_pruned.h5')
