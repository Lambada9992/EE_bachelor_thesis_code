import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_ranking as tfr

# Load the validation set
val_data, info = tfds.load("coco/2017", split="validation", with_info=True)

# Load the trained model
# model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# module_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
# loaded_obj = hub.load(module_url)
# hub_layer = hub.KerasLayer(loaded_obj, trainable=False)
# model = keras.Sequential([hub_layer])
# mAP = tfr.keras.metrics.MeanAveragePrecisionMetric()

module = hub.KerasLayer("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
model = tf.keras.Sequential([module])
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("LEGIA")
model.evaluate(tf.convert_to_tensor(val_data))


# model = keras.models.load_model("../models/ssd_mobilenet_v2_2")
# mAP = tfr.keras.metrics.MeanAveragePrecisionMetric()
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# results = model.evaluate(val_data)
#
# print('mAP: ', results[1])
