import numpy as np
import matplotlib.path as mplpath
# from skimage.transform import rotate
from os import getenv
from PIL import Image
from pascal_voc_writer import Writer
import os
import json

IMAGES_DIR = getenv("IMAGES_DIR")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR")

TARGET_IMAGES_DIR = getenv("TARGET_IMAGES_DIR")
TARGET_ANNOTATIONS_DIR = getenv("TARGET_ANNOTATIONS_DIR")

# base_path = "C:/Users/shahare/Desktop/TAU/workshop ML/photos/bone cell project"
# new_root = "C:/Users/shahare/Desktop/TAU/workshop ML/photos/XML"
# raw_path = os.path.join(new_root, "raw")
# raw_lbls_path = os.path.join(new_root, "raw_lbl")
#
# for path in [new_root, raw_path, raw_lbls_path]:
#     if not os.path.exists(path):
#         os.mkdir(path)


# def augment_image(img):
#     #     print("flipping...")
#     horizontal_flip = img[:, ::-1]
#     vertical_flip = img[::-1, :]
#     #     print("rotating...")
#     rotation_90 = rotate(img, 90)
#     rotation_180 = rotate(img, 180)
#     rotation_270 = rotate(img, 270)
#     new_images_set = [horizontal_flip, vertical_flip, rotation_270, rotation_180,
#                       rotation_90]
#     aug_imgs = {
#         "orig": img,
#         "rot_90": rotation_90,
#         "rot_180": rotation_180,
#         "rot_270": rotation_270,
#         "flip_hor": horizontal_flip,
#         "flip_ver": vertical_flip
#     }
#     return aug_imgs


# def convert_label_to_int(lbl_str):
#     if lbl_str == "g":
#         return 5
#     elif lbl_str == "p":
#         return 4
#     elif lbl_str == "0_1":
#         return 1
#     elif lbl_str == "0_2":
#         return 2
#     elif lbl_str == "0_3":
#         return 3
#     else:
#         return 0


def create_xml(json_content, file_name, abs_xmin, abs_ymin, abs_xmax, abs_ymax, idx):
    try:
        dict = json.loads(json_content)

    except:
        print(f"ERROR: Could not open labels - {file_name}")
        return False

    dims = (width, height) = (dict["imageHeight"], dict["imageWidth"])
    writer = Writer('', width, height)

    for shape in dict["shapes"]:
        lbl = shape["label"]
        points = np.array(shape["points"])

        left, right, bottom, top = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])

        ((xmin, ymin, xmax, ymax), check) = contains_polygon(abs_xmin, abs_ymin, abs_xmax, abs_ymax, left, bottom,
                                                             right, top)
        if check != 0:
            writer.addObject(lbl, xmin, ymin, xmax, ymax)

    writer.save(os.path.join(TARGET_ANNOTATIONS_DIR, f'{file_name}_{idx}.xml'))
    return True


def contains_polygon(abs_xmin, abs_ymin, abs_xmax, abs_ymax, xmin, ymin, xmax, ymax):
    area = (xmax - xmin) * (ymax - ymin) / 2
    if xmax < abs_xmin or ymax < abs_ymin or xmin > abs_xmax or ymin > abs_ymax:
        return ((0, 0, 0, 0), 0)
    if xmax <= abs_xmax and ymax <= abs_ymax and xmin >= abs_xmin and ymin >= abs_ymin:
        return ((xmin, ymin, xmax, ymax), 1)
    if ymax <= abs_ymax and ymin >= abs_ymin:
        if xmax > abs_xmin:
            if xmax - abs_xmin >= (xmax - xmin) / 2:
                return ((abs_xmin, ymin, xmax, ymax), 1)
        else:
            if abs_xmax - xmin >= (xmax - xmin) / 2:
                return ((xmin, ymin, abs_xmax, ymax), 1)
    if xmax <= abs_xmax and xmin >= abs_xmin:
        if ymax > abs_ymin:
            if ymax - abs_ymin >= (ymax - ymin) / 2:
                return ((xmin, abs_ymin, xmax, ymax), 1)
        else:
            if abs_ymax - ymin >= (ymax - ymin) / 2:
                return ((xmin, ymin, xmax, abs_ymax), 1)
    if ymax > abs_ymin and xmax > abs_xmin:
        if (ymax - abs_ymin) * (xmax - abs_xmin) >= area:
            return ((abs_xmin, abs_ymin, xmax, ymax), 1)
    if ymin < abs_ymax and xmax > abs_xmin:
        if (abs_ymax - ymin) * (xmax - abs_xmin) >= area:
            return ((abs_xmin, ymin, xmax, abs_ymax), 1)
    if ymin < abs_ymax and xmin < abs_xmax:
        if (abs_ymax - ymin) * (abs_xmax - xmin) >= area:
            return ((xmin, ymin, abs_xmax, abs_ymax), 1)
    if ymax > abs_ymin and xmin < abs_xmax:
        if (ymax - abs_ymin) * (abs_xmax - xmin) >= area:
            return ((xmin, abs_ymin, abs_xmax, ymax), 1)
    return ((0, 0, 0, 0), 0)


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
        for idx, window_mat in windows.items():
            cord = cords[idx]
            if not create_xml(json_content, name, cord[0], cord[1], cord[2], cord[3], idx):
                continue
            new_filename = f"{name}_{idx}.npy"
            path_to_save = os.path.join(TARGET_IMAGES_DIR, new_filename)
            np.save(path_to_save, window_mat)

        print(f"Finished {name}")


create_dataset((300, 300), 300)
