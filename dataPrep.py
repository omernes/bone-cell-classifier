import numpy as np
import matplotlib.path as mplpath
from keras.preprocessing.image import array_to_img
from skimage.transform import rotate
from os import getenv
from PIL import Image
from pascal_voc_writer import Writer
import os
import json

# IMAGES_DIR = getenv("IMAGES_DIR")
# ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR")
#
# TARGET_IMAGES_DIR = getenv("TARGET_IMAGES_DIR")
# TARGET_ANNOTATIONS_DIR = getenv("TARGET_ANNOTATIONS_DIR")


ANNOTATIONS_DIR = "/home/omer/uni/ml-workshop/bone-cell-classifier/data/raw"
TARGET_ANNOTATIONS_DIR = "/home/omer/uni/ml-workshop/bone-cell-classifier/data/data_xml/annotations"

IMAGES_DIR = "/home/omer/uni/ml-workshop/bone-cell-classifier/data/raw"
TARGET_IMAGES_DIR = "/home/omer/uni/ml-workshop/bone-cell-classifier/data/data_xml/images"

# base_path = "C:/Users/shahare/Desktop/TAU/workshop ML/photos/bone cell project"
# new_root = "C:/Users/shahare/Desktop/TAU/workshop ML/photos/XML"
# raw_path = os.path.join(new_root, "raw")
# raw_lbls_path = os.path.join(new_root, "raw_lbl")
#
# for path in [new_root, raw_path, raw_lbls_path]:
#     if not os.path.exists(path):
#         os.mkdir(path)


def augment_image(img):
    #     print("flipping...")
    horizontal_flip = img[:, ::-1]
    vertical_flip = img[::-1, :]
    #     print("rotating...")
    rotation_90 = rotate(img, 90)
    rotation_180 = rotate(img, 180)
    rotation_270 = rotate(img, 270)

    aug_imgs = {
        "orig": img,
        "rot_90": rotation_90,
        "rot_180": rotation_180,
        "rot_270": rotation_270,
        "flip_hor": horizontal_flip,
        "flip_ver": vertical_flip
    }
    return aug_imgs




def create_xml(json_content, file_name, abs_xmin, abs_ymin, abs_xmax, abs_ymax, idx, aug_name):
    try:
        dict = json.loads(json_content)

    except:
        print(f"ERROR: Could not open labels - {file_name}")
        return False

    dims = (width, height) = (dict["imageHeight"], dict["imageWidth"])
    writer = Writer('', 300, 300)

    has_obj = False

    for shape in dict["shapes"]:
        lbl = shape["label"]
        points = np.array(shape["points"])

        left, right, bottom, top = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])

        (xmin, ymin, xmax, ymax), check = contains_polygon(abs_xmin, abs_ymin, abs_xmax, abs_ymax, left,bottom,right,top)

        left, bottom, right, top = xmin-abs_xmin, ymin-abs_ymin, xmax-abs_xmin, ymax-abs_ymin

        if check != 0:
            print(left, bottom, right, top)

            if aug_name == "orig":
                l, b, r, t = left, bottom, right, top

            if aug_name == "rot_90":
                l, b, r, t = bottom, left, top, right

            if aug_name == "rot_180":
                l, b, r, t = 300 - right, 300 - top, 300 - left, 300 - bottom

            if aug_name == "rot_270":
                l, b, r, t = 300 - top, 300 - right, 300 - bottom, 300 - left

            if aug_name == "flip_hor":
                l, b, r, t = left, 300 - top, right, 300 - bottom

            if aug_name == "flip_ver":
                l, b, r, t = 300 - right, bottom, 300 - left, top

            has_obj = True
            writer.addObject(lbl,l, b, r, t)

    if not has_obj:
        return False
    writer.save(os.path.join(TARGET_ANNOTATIONS_DIR, f'{file_name}_{idx}_{aug_name}.xml'))
    return True


def contains_polygon(abs_xmin, abs_ymin, abs_xmax, abs_ymax, xmin, ymin, xmax, ymax):
    if xmax < abs_xmin or ymax < abs_ymin or xmin > abs_xmax or ymin > abs_ymax:              #out of bounds
        return (0, 0, 0, 0), 0

    if xmax <= abs_xmax and ymax <= abs_ymax and xmin >= abs_xmin and ymin >= abs_ymin:      #all in
        return (xmin, ymin, xmax, ymax), 1

    if ymax <= abs_ymax and ymin >= abs_ymin:                                                 #in y bounds
        if xmax > abs_xmax:
            return (xmin,ymin,abs_xmax,ymax), 1
        elif xmin < abs_xmin:
            return (abs_xmin,ymin,xmax,ymax), 1

    if xmax <= abs_xmax and xmin >= abs_xmin:                                                    #in x bounds
        if ymax > abs_ymax:
            return (xmin,ymin,xmax,abs_ymax), 1
        elif ymin < abs_ymin:
            return (xmin,abs_ymin,xmax,ymax), 1

    if ymax < abs_ymax and xmin < abs_xmin:                                                       #in bottom left corner
        return (abs_xmin, abs_ymin, xmax, ymax), 1

    if ymax > abs_ymax and xmin < abs_xmin:                                                       #in top left corner
        return (abs_xmin, ymin, xmax, abs_ymax), 1

    if ymax < abs_ymax and xmax > abs_xmax:                                                     #in bottom right corner
        return (xmin, abs_ymin, abs_xmax, ymax), 1

    if ymax > abs_ymax and xmax > abs_xmax:                                                     #in top right corner
        return (xmin, ymin, abs_xmax, abs_ymax), 1

    return (0, 0, 0, 0), 0


def create_sliding_windows(mat, window_size=(300, 300), step_size=300):
    windows = {}
    cords = {}
    idx = 0
    for x in range(0, mat.shape[0] - window_size[0], step_size):
        for y in range(0, mat.shape[1] - window_size[1], step_size):
            windows[idx] = mat[x:x + window_size[0], y:y + window_size[1]]
            cords[idx] = (x, y, x + window_size[0], y + window_size[1])
            idx += 1
        windows[idx] = mat[x:x + window_size[0], mat.shape[1] - window_size[1]:]
        cords[idx] = (x, mat.shape[1] - window_size[1], x + window_size[0], mat.shape[1])
        idx += 1
    for y in range(0, mat.shape[1] - window_size[1], step_size):
        windows[idx] = mat[mat.shape[0] - window_size[0]:, y:y + window_size[1]]
        cords[idx] = (mat.shape[0] - window_size[0], y, mat.shape[0], y + window_size[1])
        idx += 1
    windows[idx] = mat[mat.shape[0] - window_size[0]:, mat.shape[1] - window_size[1]:]
    cords[idx] = (mat.shape[0] - window_size[0], mat.shape[1] - window_size[1], mat.shape[0], mat.shape[1])
    return windows, cords


def create_dataset(window_size, step_size):
    image_names = set()

    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if root != IMAGES_DIR:
                continue

            if file.endswith(".png"):
                name = file.split(".")[0]
                if os.path.exists(os.path.join(ANNOTATIONS_DIR, f"{name}.json")):
                    image_names.add(name)

    print(f"Identified {len(image_names)} images with labels")

    for name in image_names:
        png_path = os.path.join(IMAGES_DIR, f"{name}.png")
        json_path = os.path.join(ANNOTATIONS_DIR, f"{name}.json")

        try:
            im = Image.open(png_path)
            np_im = np.array(im)

        except:
            print(f"ERROR: Could not open file - {png_path}")
            continue

        try:
            with open(json_path, "r") as f:
                json_content = f.read()
        except:
            print(f"ERROR: Could not get labels - {json_path}")
            continue
        windows, cords = create_sliding_windows(np_im, window_size, step_size)
        print("cords")
        for idx, window_mat in windows.items():
            aug_imgs = augment_image(window_mat)
            cord = cords[idx]
            for aug_name, aug_img in aug_imgs.items():
                if not create_xml(json_content, name, cord[0], cord[1], cord[2], cord[3], idx, aug_name):
                    continue
                new_filename = f"{name}_{idx}_{aug_name}.jpg"
                path_to_save = os.path.join(TARGET_IMAGES_DIR, new_filename)
                img = array_to_img(aug_img)
                img.save(path_to_save, format="JPEG")
            # np.save(path_to_save, img)

        print(f"Finished {name}")


create_dataset((300, 300), 100)
