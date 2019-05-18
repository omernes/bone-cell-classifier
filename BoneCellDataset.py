import pickle
import os

import datetime
import matplotlib.image as mpimg
import matplotlib.path as mplpath
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from image_manipulation import concat_n_images, augment_image


class Polygon:
    def __init__(self, points=[], label="default"):
        self.points = np.array(points)
        self.label = label
        self.label_int = self.convert_label_to_int(label)
        self.polygon_path = mplpath.Path(points)

    def get_min_max_block(self):
        # print(self.points[:,1])
        bottom = min(self.points[:, 1])
        top = max(self.points[:, 1])
        left = min(self.points[:, 0])
        right = max(self.points[:, 0])

        return left, right, bottom, top

    def contains_point(self, point):
        return self.polygon_path.contains_point(point)

    @staticmethod
    def convert_label_to_int(lbl_str):
        if lbl_str == "g":
            return 0
        elif lbl_str == "p":
            return 0
        elif lbl_str == "0_1":
            return 1
        elif lbl_str == "0_2":
            return 1
        elif lbl_str == "0_3":
            return 1
        else:
            return 0

    @staticmethod
    def convert_label_to_int_real(lbl_str):
        if lbl_str == "g":
            return 5
        elif lbl_str == "p":
            return 4
        elif lbl_str == "0_1":
            return 1
        elif lbl_str == "0_2":
            return 2
        elif lbl_str == "0_3":
            return 3
        else:
            return 0


def get_polygons(json_content):
    import json
    dict = json.loads(json_content)
    polygons = []
    for shape in dict["shapes"]:
        lbl = shape["label"]
        polygons.append(Polygon(shape["points"], lbl))
    shape = (dict["imageHeight"], dict["imageWidth"])
    return polygons, shape


def create_label_matrix(shape, polygons):
    lbl_mat = np.zeros(shape)
    for polygon in polygons:
        (left, right, bottom, top) = polygon.get_min_max_block()
        for x in range(left, right):
            for y in range(bottom, top):
                if polygon.contains_point([x, y]):
                    lbl_mat[y, x, 0] = polygon.label_int
                    lbl_mat[y, x, 1] = polygon.label_int
                    lbl_mat[y, x, 2] = polygon.label_int

    return lbl_mat


class BoneCellDataset:
    def __init__(self, name="default", tags={}):
        self.name = name
        self.data = []
        self.labels = []
        self.tags = {}
        self.image_metadata = {}
        self.is_augmented = False

    def populate_from_directory(self, path):
        names = set()
        for r, d, files in os.walk(path):
            for file in files:
                name = file.split('.')[0]
                if name not in names:
                    names.add(name)

        for name in names:
            img_path = os.path.join(path, f"{name}.png")
            json_path = os.path.join(path, f"{name}.json")

            try:
                img_mat = mpimg.imread(img_path)
                with open(json_path) as json_file:
                    json_content = json_file.read()
                polygons, shape = get_polygons(json_content)
                if len(shape) == 2:
                    shape = (*shape, 3)
                lbl_mat = create_label_matrix(shape, polygons)

                re_img_mat = resize(img_mat, output_shape=(800, 800, 3))
                re_img_mat = np.reshape(np.dot(re_img_mat, [0.2989, 0.5870, 0.1140]), newshape=(800, 800, 1))

                re_lbl_mat = resize(lbl_mat, output_shape=(800, 800, 1))

                self.data.append(re_img_mat)
                self.labels.append(re_lbl_mat)
                self.image_metadata[len(self.data) - 1] = {
                    "name": name,
                    "is_augmented": False
                }
                print(f"Parsed image {name}")
            except Exception as e:
                print(f"WARN :: Could not parse image {name}")

    def add_augmentation(self):
        if self.is_augmented:
            return

        n = len(self.data)
        for i in range(n):
            img_mat = self.data[i]
            lbl_mat = self.labels[i]
            (original, horizontal_flip, vertical_flip, rotation_270, rotation_180, rotation_90) = augment_image(img_mat, lbl_mat)

            self.data.append(horizontal_flip[0])
            self.labels.append(horizontal_flip[1])
            self.image_metadata[len(self.data) - 1] = {
                "name": f"{self.image_metadata[i]['name']}_aug_hor_flip",
                "is_augmented": True,
                "augmentation": "horizontal_flip"
            }

            self.data.append(vertical_flip[0])
            self.labels.append(vertical_flip[1])
            self.image_metadata[len(self.data) - 1] = {
                "name": f"{self.image_metadata[i]['name']}_aug_ver_flip",
                "is_augmented": True,
                "augmentation": "vertical_flip"
            }

            self.data.append(rotation_270[0])
            self.labels.append(rotation_270[1])
            self.image_metadata[len(self.data) - 1] = {
                "name": f"{self.image_metadata[i]['name']}_aug_rot_270",
                "is_augmented": True,
                "augmentation": "rotation_270"
            }

            self.data.append(rotation_180[0])
            self.labels.append(rotation_180[1])
            self.image_metadata[len(self.data) - 1] = {
                "name": f"{self.image_metadata[i]['name']}_aug_rot_180",
                "is_augmented": True,
                "augmentation": "rotation_180"
            }

            self.data.append(rotation_90[0])
            self.labels.append(rotation_90[1])
            self.image_metadata[len(self.data) - 1] = {
                "name": f"{self.image_metadata[i]['name']}_aug_rot_90",
                "is_augmented": True,
                "augmentation": "rotation_90"
            }

        self.is_augmented = True

    def get_full_dataset(self):
        pass

    def get_split_dataset(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(np.array(self.data), np.array(self.labels), test_size=test_size)
        return X_train, X_test, y_train, y_test

    def save_dataset(self, path):
        with open(os.path.join(path, f"{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d')}.pkl"),
                  "wb") as pickle_out:
            pickle.dump(self, pickle_out)

    @staticmethod
    def load_dataset(path):
        with open(path, "rb") as pkl:
            try:
                dataset = pickle.load(pkl)
                return dataset
            except:
                print("Could not load")
        return None


if __name__ == "__main__":
   # dataset = BoneCellDataset("test")
   # dataset.populate_from_directory("data/raw")
   dataset = BoneCellDataset.load_dataset("data/datasets/test_2019-05-18.pkl")
   dataset.add_augmentation()
   dataset.save_dataset("data/datasets")