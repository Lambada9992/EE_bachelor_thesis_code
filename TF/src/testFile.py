import numpy as np
import tensorflow as tf
from object_detection.utils import metrics


# define ground truth and detection bounding boxes
groundtruth_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
groundtruth_classes = np.array([1, 2])
detection_boxes = np.array([[150, 150, 250, 250], [350, 350, 450, 450]])
detection_scores = np.array([0.9, 0.8])
detection_classes = np.array([1, 2])

# calculate mAP
mAP, precisions, recalls, overlaps = metrics.compute_precision_recall(
    groundtruth_boxes, groundtruth_classes, detection_boxes, detection_scores, detection_classes, iou_threshold=0.5)

# print mAP and precision-recall values
print("mAP: ", mAP)
for i in range(len(precisions)):
    print("Class ", i+1, " precision: ", precisions[i])
    print("Class ", i+1, " recall: ", recalls[i])