import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_optimization as tfom
import tensorflow_datasets as tfds

# Load the CIFAR-10 dataset
dataset_name = 'cifar10'
dataset = tfds.load(name=dataset_name, split='train[:80%]', as_supervised=True)

# Split the dataset into train and validation sets
num_train_examples = int(0.8 * tf.data.experimental.cardinality(dataset).numpy())
train_dataset = dataset.take(num_train_examples)
val_dataset = dataset.skip(num_train_examples)


# Define preprocessing functions for the datasets
def preprocess_image(image, label):
    # Cast the image to float32 and normalize its values to the [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0

    # Resize the image to the input shape of the MobileNet V2 model
    image = tf.image.resize(image, (224, 224))

    return image, label


# Preprocess the train and validation datasets
train_dataset = train_dataset.map(preprocess_image).shuffle(1000).batch(32)
val_dataset = val_dataset.map(preprocess_image).batch(32)

# Load the MobileNet V2 model from TensorFlow Hub
model = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")

# Define a sparsity schedule for pruning
pruning_params = {
    "pruning_schedule": tfom.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30, final_sparsity=0.90, begin_step=0, end_step=1000
    )
}

# Create a pruned model
pruned_model = tfom.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Train the pruned model
pruned_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Strip pruning wrappers from the pruned model to reduce its size
stripped_model = tfom.sparsity.keras.strip_pruning(pruned_model)

# Save the pruned model to disk
tf.keras.models.save_model(stripped_model, "pruned_mobilenet_v2")
