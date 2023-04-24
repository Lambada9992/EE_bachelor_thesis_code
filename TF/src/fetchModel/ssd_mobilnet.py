import tensorflow as tf
import tensorflow_hub as hub

model_path = "../../models/ssd_mobilenet_v2_2"
# model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
# tf.saved_model.save(model, model_path)

module_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
embed = hub.KerasLayer(module_url, trainable=True)

# Define the input to the module
input_text = tf.keras.layers.Input(type_spec=tf.TensorSpec(shape=(1,None,None,3), dtype=tf.uint8, name=None))

# Pass the input to the module and get its output
output = embed(input_text)

# Create a Keras model using the input and output
model = tf.keras.Model(inputs=input_text, outputs=output)

# Save the model in the HDF5 format
save_path = model_path +"/"+ "my_model.h5"
tf.keras.models.save_model(model, save_path, save_format="h5")