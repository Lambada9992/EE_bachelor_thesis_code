import tensorflow as tf
import tensorflow_model_optimization as tfmo
import tensorflow_datasets as tfds


dataset_name = 'imagenette'
dataset = tfds.load(name=dataset_name, split='train[:80%]')

# Split the dataset into train and validation sets
num_train_examples = int(0.8 * tf.data.experimental.cardinality(dataset).numpy())
train_dataset = dataset.take(num_train_examples)
val_dataset = dataset.skip(num_train_examples)

model = tf.keras.applications.mobilenet.MobileNet()

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

