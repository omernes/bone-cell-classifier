import os
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.image as mpimg
import logging
import numpy as np

from image_manipulation import concat_n_images, augment_image

_logger = logging.getLogger(__name__)


class Image:
    def __init__(self, img_mat, lbl_mat=None):
        self.rgb_mat = img_mat
        self.lbl_mat = lbl_mat

    def get_grayscale_image(self):
        return np.dot(self.rgb_mat, [0.2989, 0.5870, 0.1140])

    def get_averaged_image(self):
        return np.dot(self.rgb_mat, [1 / 3, 1 / 3, 1 / 3])


class Polygon:
    def __init__(self, points=[], label="default"):
        self.points = np.array(points)
        self.label = label
        self.label_int = self.convert_label_to_int(label)
        self.polygon_path = mplPath.Path(points)

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
    return polygons


def create_label_matrix(img_mat, polygons):
    lbl_mat = np.zeros(img_mat.shape)
    print(img_mat.shape)
    for polygon in polygons:
        block = (left, right, bottom, top) = polygon.get_min_max_block()
        print(block)
        for x in range(left, right):
            for y in range(bottom, top):
                if polygon.contains_point([x, y]):
                    lbl_mat[y, x, 0] = polygon.label_int
                    lbl_mat[y, x, 1] = polygon.label_int
                    lbl_mat[y, x, 2] = polygon.label_int

    return lbl_mat


def create_dataset(folder):
    """
    load images from files
    merge imgae slices to large image
    extend image list:
        rotate
        flip
        noise
    """
    raw_images = {}
    for r, d, files in os.walk(folder):
        for file in files:
            file_type = 'image' if 'png' in file else 'labels'
            img_id = file[:2]
            img_coor = [int(x) for x in file[3:8].split('_')]

            if file_type == 'image':
                print(img_id, img_coor, file_type)
                if img_id not in raw_images:
                    raw_images[img_id] = [[0] * 4 for i in range(4)]
                # raw_images[img_id][img_coor[0] - 1][img_coor[1] - 1] = os.path.join(folder, file)
                try:
                    img = mpimg.imread(os.path.join(folder, file))
                except:
                    img = np.zeros((913, 872))
                raw_images[img_id][img_coor[0] - 1][img_coor[1] - 1] = img

    final_images = {}

    # concat rows
    for img_id, img_parts in raw_images.items():
        img_rows = []
        for i in range(4):
            img_rows.append(concat_n_images(img_parts[i], 'horizontal'))
        final_images[img_id] = concat_n_images(img_rows, 'vertical')
        # plt.imshow(final_images[img_id])
        # plt.show()

        aug_images = augment_image(final_images[img_id], final_images[img_id])

        toshow = concat_n_images([aug_images[0][0], aug_images[6][0]], 'horizontal')
        plt.imshow(toshow)
        plt.show()
        # for img in aug_images:
        #     plt.imshow(img[0])
        #     plt.show()




    return raw_images


if __name__ == "__main__":
    create_dataset("data")

    # png = "C:\\Projects\\bone-cell-classifier\\data\\B2_01_02.png"
    # json = "C:\\Projects\\bone-cell-classifier\\data\\B2_01_02.json"
    #
    # file = open(json, "r")
    # json_content = file.read()
    # polygons = get_polygons(json_content)
    #
    # np_img = mpimg.imread(png)
    # image = Image(np_img)
    #
    # lbl_mat = create_label_matrix(image.rgb_mat, polygons)
    #
    # print("-------------")
    # # plt.imshow((lbl_mat * 5).astype(np.uint8))
    # plt.imshow(image.rgb_mat, cmap='gray')
    # plt.imshow(lbl_mat[:, :, :], alpha=0.5)
    # plt.show()
