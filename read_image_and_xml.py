import os
import xml.etree.ElementTree as ET


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        # filename = root.find('filename').text
        lbl = boxes.find("name").text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        list_with_single_boxes = [lbl, xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


from matplotlib import pyplot as plt
import numpy as np
from create_dataset import Polygon, classes, n_classes


def display_image(img, polygons=None):
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
    plt.figure(figsize=(20, 12))
    plt.imshow(img)

    current_axis = plt.gca()

    if polygons:
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                color = colors[classes.index(polygon.label)]
                current_axis.add_patch(
                    plt.Polygon(xy=polygon.get_points(), closed=True, color=color, fill=False, linewidth=2))
            else:
                print(polygon)
                if len(polygon) == 5:
                    lbl, xmin, ymin, xmax, ymax = polygon
                else:
                    lbl, score, xmin, ymin, xmax, ymax = polygon
                color = colors[classes.index(polygon[0])]
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, lbl, size='x-large', color='white',
                                  bbox={'facecolor': 'green', 'alpha': 1.0})

    plt.show()

import skimage

TARGET_IMAGES = os.path.join("data_xml", "images")
TARGET_ANNOTAIONS = os.path.join("data_xml", "annotations")

filename = "B2_01_02_5"

boxes = read_content(os.path.join(TARGET_ANNOTAIONS, f"{filename}.xml"))

print(boxes)

img = skimage.data.imread(os.path.join(TARGET_IMAGES, f"{filename}.jpg"))
display_image(img, boxes)