import tensorflow as tf

def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    sparse: bool = False,
    float16: bool = False,
    dynamic_range: bool = False,
    full_integer: bool = False,
    dataset=None,
):
    def tfds_to_representative_dataset():
        for batch in dataset.unbatch().take(100).batch(1):
            yield [batch[0].numpy()]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if sparse and (float16 or dynamic_range or full_integer):
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
    elif not sparse and (float16 or dynamic_range or full_integer):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if float16:
        converter.target_spec.supported_types = [tf.float16]
    if full_integer:
        converter.representative_dataset = tfds_to_representative_dataset

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)


