import os
from keras import backend as K
from keras.optimizers import SGD
import numpy as np

from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.models.keras_ssd300 import ssd_300
from ssd_keras.ssd_encoder_decoder.ssd_output_decoder import decode_detections


class Model:
    def __init__(self):
        self.img_height = 300  # Height of the model input images
        self.img_width = 300  # Width of the model input images
        self.img_channels = 3  # Number of color channels of the model input images
        self.mean_color = [123, 117,
                      104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
        self.swap_channels = [2, 1,
                         0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
        self.n_classes = 5  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
        self.scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                         1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
        self.scales = self.scales_pascal
        self.aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
        two_boxes_for_ar1 = True
        self.steps = [8, 16, 32, 64, 100,
                 300]  # The space between two adjacent anchor box center points for each predictor layer.
        self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
                   0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
        self.clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
        self.variances = [0.1, 0.1, 0.2,
                     0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
        self.normalize_coords = True

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        classes = ['background', '0_1', '0_2', '0_3', 'p', 'g']

        MODEL_WEIGHTS_PATH = os.path.join("models", "batch2_epoch197_loss-4.2264_val_loss-3.2413_weights-only.h5")

        K.clear_session()
        self.model = ssd_300(image_size=(self.img_height, self.img_width, self.img_channels),
                        n_classes=self.n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=self.scales,
                        aspect_ratios_per_layer=self.aspect_ratios,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=self.steps,
                        offsets=self.offsets,
                        clip_boxes=self.clip_boxes,
                        variances=self.variances,
                        normalize_coords=self.normalize_coords,
                        subtract_mean=self.mean_color,
                        swap_channels=self.swap_channels)

        # 2: Load some weights into the model.
        self.model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

        # 3: Instantiate an optimizer and the SSD loss function and compile the model.
        #    If you want to follow the original Caffe implementation, use the preset SGD
        #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

        self.model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)


    def predict(self, np_img):
        y_pred = self.model.predict(np.array([np_img]))

        # 4: Decode the raw predictions in `y_pred`.

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=self.normalize_coords,
                                           img_height=self.img_width,
                                           img_width=self.img_height)
        y_pred_decoded = y_pred_decoded[0]

        return y_pred_decoded