from skimage.transform import rotate
from skimage.util import random_noise
import numpy as np
import matplotlib.image as mpimg


def augment_image(original_image, labeled_image):
    horizontal_flip = (original_image[:, ::-1], labeled_image[:, ::-1])
    vertical_flip = (original_image[::-1, :], labeled_image[::-1, :])
    rotation_90 = (rotate(original_image, 90), rotate(labeled_image, 90))
    rotation_180 = (rotate(original_image, 180), rotate(labeled_image, 180))
    rotation_270 = (rotate(original_image, 270), rotate(labeled_image, 270))
    new_images_set = [(original_image, labeled_image), horizontal_flip, vertical_flip, rotation_270, rotation_180,
                      rotation_90]
    all_images = new_images_set
    for set in new_images_set:
        all_images.append(random_noise(set[0]), random_noise(set[1]))


def concat_images_horizontal(imga, imgb):
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def concat_images_vertical(imga, imgb):
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_width = np.max([wa, wb])
    total_height = ha + hb
    new_img = np.zeros(shape=(total_height, max_width, 3))
    new_img[:ha, :wa] = imga
    new_img[ha:total_height, :wb] = imgb
    return new_img


def concat_n_images(image_path_list, type):
    output = None
    if type == 'horizontal':
        for i, img_path in enumerate(image_path_list):
            img = mpimg.imread(img_path)[:, :, :3]
            if i == 0:
                output = img
            else:
                output = concat_images_horizontal(output, img)
        return output
    else:
        for i, img_path in enumerate(image_path_list):
            img = mpimg.imread(img_path)[:, :, :3]
            if i == 0:
                output = img
            else:
                output = concat_images_vertical(output, img)
        return output
