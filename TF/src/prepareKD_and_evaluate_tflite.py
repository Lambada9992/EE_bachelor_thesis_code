import tensorflow as tf
import tensorflow_datasets as tfds
from utils.kd_utils import Distiller
import os

tf.keras.utils.set_random_seed(42)
tf.random.set_seed(42)

# prepare kd process
preprocess_image = tf.keras.applications.densenet.preprocess_input
shape = (224, 224)

teacher = tf.keras.applications.DenseNet201()
student = tf.keras.applications.MobileNetV2(weights=None)

distiller = Distiller(teacher=teacher, student=student)
distiller.compile(
    optimizer=tf.keras.optimizers.SGD(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3,
)

# prepare dataset
(train_ds, val_ds), info = tfds.load('imagenet2012', split=['train', 'validation'], with_info=True)
train_size = int(info.splits['train'].num_examples)
train_ds = train_ds.shuffle(1000, seed=1234).take(train_size)
val_size = int(info.splits['validation'].num_examples)
val_ds = val_ds.shuffle(1000, seed=1234).take(val_size)


def preprocess(data):
    image = data['image']
    label = data['label']
    image = tf.image.resize(image, shape)
    image = tf.cast(image, tf.float32)
    image = preprocess_image(image)
    return image, label


batch_size = 16
train_ds = train_ds.map(preprocess).batch(batch_size)
val_ds = val_ds.map(preprocess).batch(batch_size)

# run kd process
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./kd/checkpoints/weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5",
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=False
    ),
    tf.keras.callbacks.BackupAndRestore(backup_dir="./kd/backup", delete_checkpoint=False),
    tf.keras.callbacks.TensorBoard(log_dir="./kd/logs")
]
distiller.fit(train_ds,
              validation_data=val_ds,
              batch_size=batch_size,
              epochs=60,
              callbacks=callbacks,
              )

result = distiller.evaluate(val_ds, verbose=0)
print(distiller.metrics_names)
print(result)
distiller.student.save("model.keras")

