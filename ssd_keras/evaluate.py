import os
import numpy
from PIL import Image
import xml.etree.ElementTree as ET
from keras import backend as K
from keras.optimizers import SGD

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from eval_utils.average_precision_evaluator import Evaluator
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd300 import ssd_300
from ssd_encoder_decoder.ssd_output_decoder import decode_detections

classes = ['background', '0_1', '0_2', '0_3', 'p', 'g']

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


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text
        lbl = boxes.find('name').text
        cat_id = classes.index(lbl)

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [cat_id, xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


def get_model(weights_path):
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

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
    model.load_weights(weights_path, by_name=True)

    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

    return model


def evaluate_dataset(imageset_path, model):
    filenames = []
    with open(imageset_path, "r") as f:
        filenames = f.read()


    i = 0
    dataset = {}

    ground_truth = {}
    prediction_results = [[] for k in range(n_classes + 1)]

    filenames = filenames.split("\n")
    for filename in filenames:
        if not filename:
            continue

        img_id = filename
        i += 1
        # print(f"{i}/{len(filenames)}")

        img = numpy.asarray(Image.open(os.path.join(images_dir, f"{filename}.jpg")))
        gt_boxes = read_content(os.path.join(annotations_dir, f"{filename}.xml"))

        # resize = Resize(300, 300, labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4})
        # inverter = None
        if img.shape != (300, 300, 3):
            continue

        dataset[filename] = (img, gt_boxes)

        y_pred = model.predict(numpy.array([img]))

        # 4: Decode the raw predictions in `y_pred`.

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)
        y_pred_decoded = y_pred_decoded[0]

        ground_truth[img_id] = (numpy.asarray(gt_boxes), numpy.array([False] * len(gt_boxes)))

        for pred_box in y_pred_decoded:
            lbl, score, xmin, ymin, xmax, ymax = pred_box
            # pred_res = (img_id, score, int(xmin), int(ymin), int(xmax), int(ymax))
            pred_res = (img_id, score, xmin, ymin, xmax, ymax)
            prediction_results[int(lbl)].append(pred_res)



    evaluator = Evaluator(model=None, n_classes=n_classes, data_generator=None, bypass=True)

    num_gt_per_class = numpy.array([len(prediction_results[i]) for i in range(n_classes+1)])
    evaluator.num_gt_per_class = num_gt_per_class
    evaluator.prediction_results = prediction_results
    evaluator.ground_truth = ground_truth

    true_positives, false_positives, cumulative_true_positives, cumulative_false_positives = evaluator.match_predictions(ignore_neutral_boxes=True,
                               matching_iou_threshold=0.5,
                               border_pixels='include',
                               sorting_algorithm='quicksort',
                               verbose=True,
                               ret=True)

    cumulative_precisions, cumulative_recalls = evaluator.compute_precision_recall(verbose=True, ret=True)

    #############################################################################################
    # Compute the average precision for this class.
    #############################################################################################

    average_precisions = evaluator.compute_average_precisions(mode='sample',
                                    num_recall_points=11,
                                    verbose=True,
                                    ret=True)

    mean_average_precision = evaluator.compute_mean_average_precision(ret=True)

    ret = (mean_average_precision, average_precisions, cumulative_precisions, cumulative_recalls)
    return ret


if __name__ == "__main__":
    import os

    file = os.getenv("MODEL_WEIGHTS_PATH")
    images_dir = os.getenv("IMAGES_DIR", "../data_xml/images")
    annotations_dir = os.getenv("ANNOTATIONS_DIR", "../data_xml/annotations")
    image_set_filename = os.getenv("IMAGESET_FILENAME", "../data_xml/test.txt")

    print(f"Starting evaluation :: {file}")

    model = get_model(file)
    evaluation = evaluate_dataset(image_set_filename, model)
    mean_average_precision, average_precisions, cumulative_precisions, cumulative_recalls = evaluation

    print(f"{file}")
    print(f"mAP: {mean_average_precision}")
    print(f"average_precisions: {average_precisions}")
