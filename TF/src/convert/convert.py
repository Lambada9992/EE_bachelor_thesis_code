import tensorflow as tf
import os

saved_model_path = "../../models/ssd_mobilenet_v2_2"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('ssd_mobilenet_v2_2_3.tflite', 'wb') as f:
    f.write(tflite_model)