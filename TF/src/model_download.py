import tensorflow as tf
import os
import tarfile

# Tensorflow Object Detection Model ZOO
TF_OD_ZOO_URLS = [
"http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_kpts.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20200711/extremenet.tar.gz",
"http://download.tensorflow.org/models/object_detection/tf2/20210507/extremenet.tar.gz",
]

TF_CLASSIFICATION_ZOO_URLS = [
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b0.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b1.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b2.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b3.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b4.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b5.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b6.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/efficientnet_b7.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/resnet50_v1.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/resnet101_v1.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/resnet152_v1.tar.gz",
"http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/inception_resnet_v2.tar.gz",
# "http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilnet_v1.tar.gz",
# "http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilnet_v2.tar.gz",
]

def download_model(url, save_path):
    filename = os.path.basename(url)
    filepath = tf.keras.utils.get_file(fname=save_path + f"{filename}", origin=url)
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=save_path)

def download_tf_od_zoo(save_path: str):
    for url in TF_OD_ZOO_URLS:
        download_model(url, save_path)

def download_tf_classification_zoo(save_path: str):
    for url in TF_CLASSIFICATION_ZOO_URLS:
        download_model(url, save_path)

# create function that convert tensorflow saved model to keras h5
