from keras import backend as K
from keras.models import load_model
# from keras.optimizers import Adam
from os import getenv
# from scipy.misc import imread
# import numpy as np
# from matplotlib import pyplot as plt

# from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = int(getenv("NUM_CLASSES", "20"))
model_mode = 'inference'

RESULTS_PATH = getenv("RESULTS_PATH", "results_boxes.json")

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = getenv("MODEL_PATH")

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})


dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
VOC_2007_TEST_images_dir = getenv('IMAGES_DIR_2007_TEST', 'datasets/VOCdevkit/VOC2007/JPEGImages/')
VOC_2007_TEST_annotations_dir = getenv('ANNOT_DIR_2007_TEST', 'datasets/VOCdevkit/VOC2007/Annotations/')
VOC_2007_test_image_set_filename = getenv('IMAGESET_TEST_2007', 'datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt')

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
           '0_1', '0_2', '0_3', 'g',
           'p']

dataset.parse_xml(images_dirs=[VOC_2007_TEST_images_dir],
                  image_set_filenames=[VOC_2007_test_image_set_filename],
                  annotations_dirs=[VOC_2007_TEST_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)


evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
try:
    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=8,
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
except:
    print("FAIL")

results_by_image = evaluator.prediction_results_by_image
print("RESULTSSSSSS")
print(results_by_image)

# with open(RESULTS_PATH, "w") as f:
#     import json
#     f.write(json.dumps(results_by_image))

mean_average_precision, average_precisions, precisions, recalls = results


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))