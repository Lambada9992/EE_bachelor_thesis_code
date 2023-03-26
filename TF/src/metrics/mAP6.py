import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


MODEL_URI = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'

def process_image(data):
  data['image'] = tf.image.resize(data['image'], (320, 320))
  return data


# Representative dataset
def representative_dataset(dataset):
  def _data_gen():
    for data in dataset.batch(1):
      yield [data['image']]
  return _data_gen


def eval_tflite(tflite_model, dataset):
  """Evaluates tensorflow lite classification model with the given dataset."""
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_idx = interpreter.get_input_details()[0]['index']
  output_idx = interpreter.get_output_details()[0]['index']

  results = []

  for data in representative_dataset(dataset)():
    interpreter.set_tensor(input_idx, data[0])
    interpreter.invoke()
    results.append(interpreter.get_tensor(output_idx).flatten())

  results = np.array(results)
  gt_labels = np.array(list(dataset.map(lambda data: data['label'] + 1)))
  accuracy = (
      np.sum(np.argsort(results, axis=1)[:, -5:] == gt_labels.reshape(-1, 1)) /
      gt_labels.size)
  print(f'Top-5 accuracy (quantized): {accuracy * 100:.2f}%')


# model = tf.keras.Sequential([
#   tf.keras.layers.Input(shape=(320, 320, 3), batch_size=1, dtype=tf.uint8),
#   hub.KerasLayer(MODEL_URI)
# ])

model = keras.Model(
tf.keras.layers.Input(shape=(320, 320, 3), batch_size=1, dtype=tf.uint8),
hub.KerasLayer(MODEL_URI)
)

model.compile(
    loss='sparse_categorical_crossentropy',
    metrics='sparse_top_k_categorical_accuracy')
model.build([1, 320, 320, 3])

# Prepare dataset with 100 examples
ds = tfds.load('coco/2017', split='validation[:1%]',try_gcs=False)
ds = ds.map(process_image)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.representative_dataset = representative_dataset(ds)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# quantized_model = converter.convert()

test_ds = ds.map(lambda data: (data['image'], data['label'] + 1)).batch(16)
loss, acc = model.evaluate(test_ds)
print(f'Top-5 accuracy (float): {acc * 100:.2f}%')


# eval_tflite(quantized_model, ds)