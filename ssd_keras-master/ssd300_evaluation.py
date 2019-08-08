from keras import backend as K
from keras.models import load_model
# from keras.optimizers import Adam
from os import getenv
# from scipy.misc import imread
# import numpy as np
# from matplotlib import pyplot as plt

# from models.keras_ssd300 import ssd_300
from keras.optimizers import SGD

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set a few configuration parameters.
from models.keras_ssd300 import ssd_300

img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
mean_color = [123, 117,
              104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1,
                 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 5  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                 1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

model_mode = 'inference'

RESULTS_PATH = getenv("RESULTS_PATH", "results_boxes.json")

# TODO: Set the path to the `.h5` file of the model to be loaded.
# model_path = getenv("MODEL_PATH")
#
# # We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#
# K.clear_session() # Clear previous models from memory.
#
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'DecodeDetections': DecodeDetections,
#                                                'compute_loss': ssd_loss.compute_loss})

MODEL_WEIGHTS_PATH = "ssd300_bone-cell-dataset_epoch-22_loss-3.2010_val_loss-2.7213_weights-only.h5"

K.clear_session()
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: Load some weights into the model.
model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)



dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
# VOC_2007_TEST_images_dir = getenv('IMAGES_DIR_2007_TEST', 'datasets/VOCdevkit/VOC2007/JPEGImages/')
# VOC_2007_TEST_annotations_dir = getenv('ANNOT_DIR_2007_TEST', 'datasets/VOCdevkit/VOC2007/Annotations/')
# VOC_2007_test_image_set_filename = getenv('IMAGESET_TEST_2007', 'datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
IMAGES_DIR = getenv("IMAGES_DIR", "data_xml/images")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR", "data_xml/annotations")
IMAGESET_FILENAME = getenv("IMAGESET_FILENAME", "data_xml/test_new.txt")

# Pascal_VOC_dataset_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages/'
# Pascal_VOC_dataset_annotations_dir = '../../datasets/VOCdevkit/VOC2007/Annotations/'
# Pascal_VOC_dataset_image_set_filename = '../../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor']

classes = ['background',
           '0_1', '0_2', '0_3', 'p', 'g']

dataset.parse_xml(images_dirs=[IMAGES_DIR],
                  image_set_filenames=[IMAGESET_FILENAME],
                  annotations_dirs=[ANNOTATIONS_DIR],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)


evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
# try:
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)
# except Exception as e:
#     print(e)

results_by_image = evaluator.prediction_results_by_image
# print("RESULTSSSSSS")
# print(results_by_image)

try:
    with open(RESULTS_PATH, "w") as f:
        import json
        f.write(json.dumps(results_by_image))
except:
    print("Could not save results to file")

mean_average_precision, average_precisions, precisions, recalls = results


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))


# img_height=img_height
# img_width=img_width
# batch_size=1
# data_generator_mode='resize'
# round_confidences=False
# matching_iou_threshold=0.5
# border_pixels='include'
# sorting_algorithm='quicksort'
# average_precision_mode='sample'
# num_recall_points=11
# ignore_neutral_boxes=True
# return_precisions=True
# return_recalls=True
# return_average_precisions=True
# verbose=True
#
# decoding_confidence_thresh=0.01
# decoding_iou_threshold=0.45
# decoding_top_k=200
# decoding_pred_coords='centroids'
# decoding_normalize_coords=True
#
# evaluator.predict_on_dataset(img_height=img_height,
#                             img_width=img_width,
#                             batch_size=batch_size,
#                             data_generator_mode=data_generator_mode,
#                             decoding_confidence_thresh=decoding_confidence_thresh,
#                             decoding_iou_threshold=decoding_iou_threshold,
#                             decoding_top_k=decoding_top_k,
#                             decoding_pred_coords=decoding_pred_coords,
#                             decoding_normalize_coords=decoding_normalize_coords,
#                             decoding_border_pixels=border_pixels,
#                             round_confidences=round_confidences,
#                             verbose=verbose,
#                             ret=False)
#
# # evaluator.write_predictions_to_txt()
#
# results = evaluator.prediction_results_by_image
# for img_id, res in results.items():
#     real_res = [x for x in res if x[0] > 0]
#     num_res = len(real_res)
#     print(f"{img_id} :: {num_res}")
#     if real_res:
#         print(real_res)