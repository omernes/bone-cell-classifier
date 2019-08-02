import io
from PIL import Image
import base64
import os
from matplotlib import pyplot as plt
# from keras.preprocessing.image import array_to_img
import numpy as np
import json
from pascal_voc_writer import Writer

classes = ['background', '0_1', '0_2', '0_3', 'g', 'p']
n_classes = 5


# class BoundingBox:
#     def __init__(self, xmin, xmax, ymin, ymax, label=None):
#         self.xmin = xmin
#         self.xmax = xmax
#         self.ymin = ymin
#         self.ymax = ymax
#         self.label = label
#
#     def get_box_coordinates(self, is_closed=False):
#         points = [[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]]
#         if is_closed:
#             points.append([self.xmin, self.ymin])
#         return np.array(points)


class Polygon:
    def __init__(self, label=None, points=None):
        self.label = label
        self.points = points
        self.box_coordinates = None

    def get_box_coordinates(self, is_closed=False):
        if not self.box_coordinates:
            xmin, xmax, ymin, ymax = min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]), max(
                self.points[:, 1])
            points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            if is_closed:
                points.append([xmin, ymin])
            self.box_coordinates = np.array(points)

        return self.box_coordinates

    # def get_bounding_box(self):
    #     if self.points is None or len(self.points) == 0:
    #         return None
    #
    #     xmin, xmax, ymin, ymax = min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]), max(
    #         self.points[:, 1])
    #     return BoundingBox(xmin, xmax, ymin, ymax)

    def get_points(self):
        return self.points

    def get_bounding_box_for_display(self):
        coords = self.get_box_coordinates()
        return (self.label, *coords[0], *coords[2])


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
                lbl, score, xmin, ymin, xmax, ymax = polygon
                color = colors[classes.index(polygon[0])]
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, lbl, size='x-large', color='white',
                                  bbox={'facecolor': 'green', 'alpha': 1.0})

    plt.show()


def extract_data_from_json(json_path):
    try:
        with open(json_path, "r") as f:
            json_content = f.read()
        data = json.loads(json_content)

        polygons = []
        for shape in data["shapes"]:
            lbl = shape["label"]
            points = np.array(shape["points"])
            polygon = Polygon(label=lbl, points=points)
            polygons.append(polygon)

        image_data = data["imageData"]
        image_data = base64.b64decode(str(image_data))
        image = Image.open(io.BytesIO(image_data))
        np_img = np.array(image)

        return np_img, polygons
    except:
        print(f"Could not parse JSON file in {json_path}")
        return None


def load_dataset(data_path):
    dataset = {}
    for root, dir, files in os.walk(data_path):
        if root != data_path:
            continue
        for file in files:
            if file[-4:] != "json":
                continue
            img_id = file[:-5]
            data = extract_data_from_json(os.path.join(root, file))
            if not data:
                continue
            # img, polygons = data
            dataset[img_id] = {
                "image": data[0],  # image as np array
                "polygons": data[1]  # list of Polygon objects
            }
    return dataset


def get_overlapping_rectangle(polygon, abs_xmin, abs_ymin, abs_xmax, abs_ymax):
    lbl, xmin, ymin, xmax, ymax = polygon.get_bounding_box_for_display()
    dx = min(xmax, abs_xmax) - max(xmin, abs_xmin)
    dy = min(ymax, abs_ymax) - max(ymin, abs_ymin)
    if (dx >= 0) and (dy >= 0):
        return max(xmin, abs_xmin), max(ymin, abs_ymin), min(xmax, abs_xmax), min(ymax, abs_ymax)
    return None


def change_coordinates(point, new_root, angle):
    (x, y) = point
    a, b = new_root
    new_x = (x - a)
    new_y = (y - b)

    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    R = np.array(((c, -s), (s, c)))

    new_point = R.dot(np.array((new_x, new_y)))

    return new_point[0], new_point[1]


def cut_window_from_image(img, polygons, xy=(0, 0), window_size=(300, 300), angle=0):
    x, y = xy
    width, height = window_size
    if x + width > img.shape[0] or y + height > img.shape[1]:
        return None

    # window_img = img[x:x + width, y:y + height]
    window_img = img[y:y + width, x:x+height]
    window_polygons = []


    for polygon in polygons:
        points = polygon.get_points()
        new_points = np.array([change_coordinates(point, xy, angle) for point in polygon.get_points()])
        new_polygon = Polygon(label=polygon.label, points=new_points)
        overlap = get_overlapping_rectangle(new_polygon, 0, 0, *window_size)
        if not overlap:
            continue
        window_polygons.append([polygon.label, 1.0, *overlap])

    return window_img, window_polygons


def create_sliding_windows(img, polygons, size=(300, 300), step=300):
    width, height = img.shape[0], img.shape[1]
    win_width, win_height = size

    windows = {}
    index = 0
    for x in range(0, width - win_width, step):
        for y in range(0, height - win_height, step):
            win_img, win_poly = cut_window_from_image(img, polygons, xy=(x, y), window_size=size)
            windows[index] = (win_img, win_poly)
            index += 1
        win_img, win_poly = cut_window_from_image(img, polygons, xy=(x, height - win_height), window_size=size)
        windows[index] = (win_img, win_poly)
        index += 1
    x = width - win_width
    for y in range(0, height - win_height, step):
        win_img, win_poly = cut_window_from_image(img, polygons, xy=(x, y), window_size=size)
        windows[index] = (win_img, win_poly)
        index += 1
    win_img, win_poly = cut_window_from_image(img, polygons, xy=(x, height - win_height), window_size=size)
    windows[index] = (win_img, win_poly)

    return windows

def create_xml(polygons, width, height, path_to_save):
    writer = Writer('', width, height)

    for polygon in polygons:
        writer.addObject(polygon[0], int(polygon[2]), int(polygon[3]), int(polygon[4]), int(polygon[5]))

    writer.save(path_to_save)
    # return json.dumps(polygons)

def create_dataset(data_path, target_image_path, target_annotation_path):
    dataset = load_dataset(data_path)
    for img_id, record in dataset.items():
        print(f"Starting :: {img_id}")
        # display_image(record["image"], record["polygons"])
        # win_img, win_poly = cut_window_from_image(record["image"], record["polygons"], xy=(0, 310), window_size=(300, 300))
        # display_image(win_img, win_poly)
        windows = create_sliding_windows(record["image"], record["polygons"], size=(300,300), step=100)
        print(f"-- created {len(windows)} windows...")
        for idx, window in windows.items():
            if len(window[1]) == 0:
                print(f"-- window {idx} has no polygons... skipping")
                continue

            filename = f"{img_id}_{idx}"
            # display_image(window[0], window[1])

            from PIL import Image
            im = Image.fromarray(window[0])
            im.save(os.path.join(target_image_path, f"{filename}.jpg"))

            # img = array_to_img(window[0])
            # img.save(os.path.join(target_image_path, f"{filename}.jpg"), format="JPEG")

            create_xml(window[1], 300, 300, os.path.join(TARGET_ANNOTAIONS, f"{filename}.xml"))
        print("-- done")


if __name__ == "__main__":
    DATASET_PATH = "data"
    TARGET_IMAGES = os.path.join("data_xml", "images")
    TARGET_ANNOTAIONS = os.path.join("data_xml", "annotations")

    create_dataset(DATASET_PATH, TARGET_IMAGES, TARGET_ANNOTAIONS)


    # dataset = load_dataset(DATASET_PATH)
    # img = dataset["B2_03_03"]["image"]
    #
    # poly = [Polygon(label="0_1", points=[[100, 100], [200, 100], [200, 150], [100, 150]])]
    #
    # # display_image(img, poly)
    #
    # img2, poly2 = cut_window_from_image(img, poly, xy=(50, 0))
    #
    # display_image(img2, poly2)
    # display_image(img2, poly2)