import tensorflow as tf
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
model_path = "../../models/ssd_mobilnet_v2_2"
tf.saved_model.save(model, model_path)