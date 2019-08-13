from os import getenv, path

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize, RandomFlip, RandomTranslate
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels, ConvertDataType
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation, SSDExpand
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

SAVE_IN_MEMORY = bool(int(getenv("SAVE_IN_MEMORY", "0")))
MODEL_PATH = getenv("MODEL_PATH", "")

WEIGHTS_PATH = getenv("WEIGHTS_PATH", "")

## 1. Set the model configuration parameters

img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
mean_color = [123, 117,
              104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1,
                 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = int(getenv("NUM_CLASSES", 5))  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
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

enable_ssd_expand = bool(int(getenv("ENABLE_SSD_EXPAND", "0")))

## 2. Build or load the model

K.clear_session()  # Clear previous models from memory.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

if MODEL_PATH != "":
    model = load_model(MODEL_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'compute_loss': ssd_loss.compute_loss})
else:
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

    # TODO: Set the path to the weights you want to load.
    # weights_path = 'data/VGG_ILSVRC_16_layers_fc_reduced.h5'

    model.load_weights(WEIGHTS_PATH, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

## 3. Set up the data generators for the training

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=SAVE_IN_MEMORY, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=SAVE_IN_MEMORY, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

IMAGES_DIR = getenv("IMAGES_DIR")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR")

TRAIN_IMAGE_SET_FILENAME = getenv("TRAIN_IMAGESET_FILENAME")
VAL_IMAGE_SET_FILENAME = getenv("VAL_IMAGESET_FILENAME")
TEST_IMAGE_SET_FILENAME = getenv("TEST_IMAGESET_FILENAME")


# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           '0_1', '0_2', '0_3', 'p', 'g']


train_dataset.parse_xml(images_dirs=[IMAGES_DIR],
                        image_set_filenames=[TRAIN_IMAGE_SET_FILENAME],
                        annotations_dirs=[ANNOTATIONS_DIR],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[IMAGES_DIR],
                      image_set_filenames=[VAL_IMAGE_SET_FILENAME],
                      annotations_dirs=[ANNOTATIONS_DIR],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


# 3: Set the batch size.

batch_size = int(getenv("BATCH_SIZE", "8"))  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
# ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
#                                             img_width=img_width,
#                                             background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# Create Transformations
convert_to_3_channels = ConvertTo3Channels()
convert_to_uint8      = ConvertDataType(to='uint8')
resize                = Resize(height=img_height, width=img_width)
random_flip_hor           = RandomFlip(dim='horizontal', prob=0.5)
random_flip_ver           = RandomFlip(dim='vertical', prob=0.5)

ssd_expand = SSDExpand()

box_filter = BoxFilter(overlap_criterion='area',
                       overlap_bounds=(0.4, 1.0))
image_validator = ImageValidator(overlap_criterion='area',
                                 bounds=(0.3, 1.0),
                                 n_boxes_min=1)
random_translate = RandomTranslate(dy_minmax=(0.03,0.3),
                                   dx_minmax=(0.03,0.3),
                                   prob=0.5,
                                   clip_boxes=False,
                                   box_filter=None,
                                   image_validator=image_validator,
                                   n_trials_max=3)

augmentations = [convert_to_3_channels, convert_to_uint8, random_flip_hor, random_flip_ver, random_translate]

if enable_ssd_expand:
    augmentations.append(ssd_expand)

augmentations.append(resize)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         # transformations=[ssd_data_augmentation],
                                         transformations=augmentations,
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
MODEL_CHECKPOINT_PATH = getenv("MODEL_CHECKPOINT_PATH", ".")

model_checkpoint = ModelCheckpoint(
    filepath=path.join(MODEL_CHECKPOINT_PATH, 'ssd300_bone-cell-dataset_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=int(getenv("CHECKPOINT_PERIOD", "8")))

csv_logger = CSVLogger(filename='ssd300_bone-cell-dataset_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = int(getenv("INITIAL_EPOCH", 0))
final_epoch = int(getenv("FINAL_EPOCH", 120))
steps_per_epoch = int(getenv("STEPS_PER_EPOCH", 1000))

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)
