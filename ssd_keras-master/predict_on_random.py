import os
from evaluate_new import display_image, classes, n_classes, get_model, read_content
import random
from PIL import Image
import numpy

from ssd_encoder_decoder.ssd_output_decoder import decode_detections

TARGET_IMAGES="../data_xml/images"
TARGET_ANNOTATIONS="../data_xml/annotations"
IMAGE_SET="../data_xml/test.txt"

with open(IMAGE_SET, "r") as f:
    imageset_content = f.read().split('\n')

title = random.choice(imageset_content)

print(title)
# title = "B4_02_02_22_hor"

img = Image.open(os.path.join(TARGET_IMAGES, f"{title}.jpg"))
np_img = numpy.array(img)

xml_path = os.path.join(TARGET_ANNOTATIONS, f"{title}.xml")
boxes = read_content(xml_path)



model = get_model()

y_pred = model.predict(numpy.array([np_img]))

# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=300,
                                   img_width=300)
y_pred_decoded = y_pred_decoded[0]

display_image(np_img, boxes)
display_image(np_img, y_pred_decoded)

print('test')