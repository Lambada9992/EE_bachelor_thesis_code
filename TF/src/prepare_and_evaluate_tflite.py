import tensorflow as tf
import tensorflow_datasets as tfds
import utils.convert_utils as convert_utils
import utils.eval_utils as eval_utils
import utils.model_size_utils as model_size_utils
import utils.prune_utils as prune_utils
import utils.save_stats_utils as save_stats_utils
import os

if not os.path.exists('output'):
    os.makedirs('output')

excel_path = f'output/result.xlsx'

prunable_models = [
    (
        'DenseNet121',
        tf.keras.applications.DenseNet121(),
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'DenseNet169',
        tf.keras.applications.DenseNet169(),
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'DenseNet201',
        tf.keras.applications.DenseNet201(),
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'InceptionResNetV2',
        tf.keras.applications.InceptionResNetV2(input_shape=(299, 299, 3)),
        tf.keras.applications.inception_resnet_v2.preprocess_input,
        tf.keras.optimizers.SGD(),
        (299, 299)
    ),
    (
        'InceptionV3',
        tf.keras.applications.InceptionV3(input_shape=(299, 299, 3)),
        tf.keras.applications.inception_v3.preprocess_input,
        tf.keras.optimizers.SGD(),
        (299, 299)
    ),
    (
        'MobileNet',
        tf.keras.applications.MobileNet(input_shape=(224, 224, 3)),
        tf.keras.applications.mobilenet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'MobileNetV2',
        tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3)),
        tf.keras.applications.mobilenet_v2.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'NASNetMobile',
        tf.keras.applications.NASNetMobile(input_shape=(224, 224, 3)),
        tf.keras.applications.nasnet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet101',
        tf.keras.applications.ResNet101(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet152',
        tf.keras.applications.ResNet152(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet50',
        tf.keras.applications.ResNet50(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet101V2',
        tf.keras.applications.ResNet101V2(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet152V2',
        tf.keras.applications.ResNet152V2(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
    (
        'ResNet50V2',
        tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3)),
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.optimizers.SGD(),
        (224, 224)
    ),
]

def process_model(
        model_name,
        postfix,
        model_instance,
        base_model_eval,
        base_model_size,
        pruned_model_eval,
        pruned_model_size,
        sparse=False,
        float16=False,
        dynamic_range=False,
        full_integer=False,
        dataset=None
):
    model_path = f'output/{model_name}{postfix}.tflite'
    if not os.path.exists(model_path):
        convert_utils.convert_to_tflite(model_instance, model_path,
                                        float16=float16, dynamic_range=dynamic_range,
                                        full_integer=full_integer, sparse=sparse, dataset=dataset)
        tflite_model_eval = eval_utils.eval_tflite_model(model_path, val_ds)
        print(f"tflite model{postfix} eval: {tflite_model_eval}")
        tflite_model_size = model_size_utils.get_gzipped_model_size(model_path)

        quantized = "None" if not float16 and not dynamic_range and not full_integer else "Float16" if float16 else "DynamicRange" if dynamic_range else "FullInteger"

        save_stats_utils.save_model_stats_to_file(excel_path, name,
                                                  pruned=sparse, quantized=quantized,
                                                  base_model_accuracy=base_model_eval, base_model_size=base_model_size,
                                                  base_pruned_model_accuracy=pruned_model_eval,
                                                  base_pruned_model_size=pruned_model_size,
                                                  accuracy=tflite_model_eval,
                                                  size=tflite_model_size
                                                  )

models_test = [(
        'MobileNetV2',
        tf.keras.applications.MobileNetV2(),
        tf.keras.applications.mobilenet_v2.preprocess_input,
        (224, 224)
    ),]

def check_if_proces_done(name):
    if not os.path.exists(f'output/{name}.tflite'): return False
    if not os.path.exists(f'output/{name}.h5'): return False
    if not os.path.exists(f'output/{name}_pruned.tflite'): return False
    if not os.path.exists(f'output/{name}_pruned.h5'): return False
    if not os.path.exists(f'output/{name}_fp16.tflite'): return False
    if not os.path.exists(f'output/{name}_fp16_pruned.tflite'): return False
    if not os.path.exists(f'output/{name}_dint8.tflite'): return False
    if not os.path.exists(f'output/{name}_dint8_pruned.tflite'): return False
    if not os.path.exists(f'output/{name}_fint8.tflite'): return False
    if not os.path.exists(f'output/{name}_fint8_pruned.tflite'): return False

    return True


for name, model, preprocess_image, optimizer, shape in prunable_models:
    if check_if_proces_done(name): continue
    try:
        # prepare data
        (train_ds, val_ds), info = tfds.load('imagenet2012', split=['train', 'validation'], with_info=True)

        train_size = int(info.splits['train'].num_examples * 0.075)
        train_ds = train_ds.shuffle(1000, seed=1234).take(train_size)
        val_size = int(info.splits['validation'].num_examples * 0.25)
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

        # eval base model
        print(f"WORKING ON MODEL: {name}")
        base_model_eval = eval_utils.eval_keras_model(model, val_ds)
        print(f"base model eval: {base_model_eval}")
        try:
            if not os.path.exists(f'output/{name}.h5'):
                model.save(f'output/{name}.h5')
            base_model_size = model_size_utils.get_gzipped_model_size(f'output/{name}.h5')
        except:
            base_model_size = 0

        # pruned model
        pruning_failed = False

        try:
            if not os.path.exists(f'output/{name}_pruned.h5'):
                pruned_model = prune_utils.prune_model(model, train_ds, val_ds, train_size,
                                                   final_sparsity=0.5, epochs=3, batch_size=batch_size,
                                                   optimizer=optimizer)
                pruned_model.save(f'output/{name}_pruned.h5')
            else:
                pruned_model = tf.keras.models.load_model(f'output/{name}_pruned.h5')
            pruned_model_eval = eval_utils.eval_keras_model(pruned_model, val_ds)
            print(f"pruned model eval: {pruned_model_eval}")
            pruned_model_size = model_size_utils.get_gzipped_model_size(f'output/{name}_pruned.h5')
        except Exception as e:
            print("Pruning failed")
            print(e)
            pruning_failed = True
            pruned_model_eval = 0
            pruned_model_size = 0

        # tflite base model
        process_model(model_name=name, model_instance=model, base_model_eval=base_model_eval,
                      base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                      pruned_model_size=pruned_model_size,
                      postfix="",
                      sparse=False,
                      float16=False,
                      dynamic_range=False,
                      full_integer=False,
                      dataset=None
                      )

        # tflite model pruned
        if not pruning_failed:
            process_model(model_name=name, model_instance=pruned_model, base_model_eval=base_model_eval,
                          base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                          pruned_model_size=pruned_model_size,
                          postfix="_pruned",
                          sparse=True,
                          float16=False,
                          dynamic_range=False,
                          full_integer=False,
                          dataset=None
                          )

        # tflite model float16
        process_model(model_name=name, model_instance=model, base_model_eval=base_model_eval,
                      base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                      pruned_model_size=pruned_model_size,
                      postfix="_fp16",
                      sparse=False,
                      float16=True,
                      dynamic_range=False,
                      full_integer=False,
                      dataset=None
                      )

        # tflite model float16 + pruning
        if not pruning_failed:
            process_model(model_name=name, model_instance=pruned_model, base_model_eval=base_model_eval,
                          base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                          pruned_model_size=pruned_model_size,
                          postfix="_fp16_pruned",
                          sparse=True,
                          float16=True,
                          dynamic_range=False,
                          full_integer=False,
                          dataset=None
                          )

        # tflite model dynamic int8
        process_model(model_name=name, model_instance=model, base_model_eval=base_model_eval,
                      base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                      pruned_model_size=pruned_model_size,
                      postfix="_dint8",
                      sparse=False,
                      float16=False,
                      dynamic_range=True,
                      full_integer=False,
                      dataset=None
                      )

        # tflite model dynamic int8 + pruning
        if not pruning_failed:
            process_model(model_name=name, model_instance=pruned_model, base_model_eval=base_model_eval,
                          base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                          pruned_model_size=pruned_model_size,
                          postfix="_dint8_pruned",
                          sparse=True,
                          float16=False,
                          dynamic_range=True,
                          full_integer=False,
                          dataset=None
                          )

        # tflite model full int 8
        process_model(model_name=name, model_instance=model, base_model_eval=base_model_eval,
                      base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                      pruned_model_size=pruned_model_size,
                      postfix="_fint8",
                      sparse=False,
                      float16=False,
                      dynamic_range=False,
                      full_integer=True,
                      dataset=val_ds
                      )

        # tflite model full int 8 + pruning
        if not pruning_failed:
            process_model(model_name=name, model_instance=pruned_model, base_model_eval=base_model_eval,
                          base_model_size=base_model_size, pruned_model_eval=pruned_model_eval,
                          pruned_model_size=pruned_model_size,
                          postfix="_fint8_pruned",
                          sparse=True,
                          float16=False,
                          dynamic_range=False,
                          full_integer=True,
                          dataset=val_ds
                          )

    except Exception as e:
        print("Failed to process model: " + name)
        print(f"ERROR: {e}")
        continue
