import tensorflow as tf

# Define the path to the TensorFlow checkpoint file
checkpoint_path = "path/to/checkpoint"

# Define the path where the Keras H5 file will be saved
h5_path = "path/to/h5/file"

# Load the TensorFlow checkpoint into a dictionary
vars_dict = tf.train.load_checkpoint(checkpoint_path).get_tensor_map()

# Create a Keras model
model = tf.keras.models.Sequential()

# Add layers to the Keras model using the weights from the checkpoint
for var_name, var_value in vars_dict.items():
    layer_name = var_name.split('/')[0]
    layer = model.get_layer(layer_name)
    if layer is None:
        layer = tf.keras.layers.Dense(var_value.shape[-1], name=layer_name)
        model.add(layer)
    layer.set_weights([var_value])

# Save the Keras model to an H5 file
tf.keras.models.save_model(model, h5_path)


