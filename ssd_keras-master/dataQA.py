from os import getenv
import numpy as np
from matplotlib import pyplot as plt

from bounding_box_utils.bounding_box_utils import iou

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_patch_sampling_ops import *
from data_generator.object_detection_2d_geometric_ops import *
from data_generator.object_detection_2d_photometric_ops import *
from data_generator.object_detection_2d_image_boxes_validation_utils import *
from data_generator.data_augmentation_chain_original_ssd import *

dataset = DataGenerator(labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'))

# images_dir         = 'C:/Users/shahare/Desktop/TAU/workshop ML/photos/bone cell project'
# annotations_dir    = 'C:/Users/shahare/Desktop/TAU/workshop ML/photos/XML/raw'
# image_set_filename = 'C:/Users/shahare/Desktop/TAU/workshop ML/photos/imageset.txt'

# images_dir = getenv("IMAGES_DIR")
# annotations_dir = getenv("ANNOTATIONS_DIR")
# images_set_filename = getenv("DATASET_FILENAME")

images_dir = "data"
annotations_dir = "data"
images_set_filename = "data\\files.txt"

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           '0_1', '0_2', '0_3', 'g',
           'p']

dataset.parse_xml(images_dirs=[images_dir],
                  image_set_filenames=[image_set_filename],
                  annotations_dirs=[annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

translate = Translate(dy=-0.2,
                      dx=0.3,
                      clip_boxes=False,
                      box_filter=None,
                      background=(0,0,0))
batch_size = 1

data_generator = dataset.generate(batch_size=batch_size,
                                  shuffle=False,
                                  transformations=[translate],
                                  label_encoder=None,
                                  returns={'processed_images',
                                           'processed_labels',
                                           'filenames',
                                           'original_images',
                                           'original_labels'},
                                  keep_images_without_gt=False)
processed_images, processed_annotations, filenames, original_images, original_annotations = next(data_generator)

i = 0 # Which batch item to look at

print("Image:", filenames[i])
print()
print("Original ground truth boxes:\n")
print(np.array(original_annotations[i]))

colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()  # Set the colors for the bounding boxes

fig, cell = plt.subplots(1, 2, figsize=(20, 16))
cell[0].imshow(original_images[i])
cell[1].imshow(processed_images[i])

for box in original_annotations[i]:
    xmin = box[1]
ymin = box[2]
xmax = box[3]
ymax = box[4]
color = colors[int(box[0])]
label = '{}'.format(classes[int(box[0])])
cell[0].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
cell[0].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

for box in processed_annotations[i]:
    xmin = box[1]
ymin = box[2]
xmax = box[3]
ymax = box[4]
color = colors[int(box[0])]
label = '{}'.format(classes[int(box[0])])
cell[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
cell[1].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})




