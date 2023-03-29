import tensorflow as tf
import tensorflow_hub as hub

# Download the model from TensorFlow Hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.KerasLayer(module_url, trainable=True)

# Define the input to the module
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string)

# Pass the input to the module and get its output
output = embed(input_text)

# Create a Keras model using the input and output
model = tf.keras.Model(inputs=input_text, outputs=output)

# Save the model in the HDF5 format
save_path = "my_model.h5"
tf.keras.models.save_model(model, save_path, save_format="h5")