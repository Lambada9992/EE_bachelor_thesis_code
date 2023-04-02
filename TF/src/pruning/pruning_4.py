import tensorflow as tf
import tempfile
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
import numpy as np

model = tf.keras.applications.MobileNetV2()

(train_ds, val_ds), info = tfds.load('imagenet2012', split=['train', 'validation'], with_info=True)

def preprocess(data):
    image = data['image']
    label = data['label']
    # Resize image to 224x224
    image = tf.image.resize(image, (224, 224))
    # Convert image to float32
    image = tf.cast(image, tf.float32)
    # Normalize image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# take 1% of the dataset
train_ds = train_ds.take(int(info.splits['train'].num_examples * 0.01))

# train_ds = train_ds.map(preprocess).shuffle(10000).batch(32)
train_ds = train_ds.map(preprocess).batch(32)
val_ds = val_ds.map(preprocess).batch(32)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

_, baseline_model_accuracy = model.evaluate(val_ds, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = train_ds.__len__().numpy() * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    jit_compile = True
)

# model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_ds, validation_data=val_ds,
                  batch_size=batch_size, epochs=epochs,
                  callbacks=callbacks)

_, pruned_model_accuracy = model.evaluate(val_ds, verbose=0)

print('Pruned test accuracy:', pruned_model_accuracy)
