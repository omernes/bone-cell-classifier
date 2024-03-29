import io
import random

from PIL import Image
import base64
import os
from matplotlib import pyplot as plt
# from keras.preprocessing.image import array_to_img
import numpy as np
import json
from pascal_voc_writer import Writer
from skimage.transform import rotate

classes = ['background', '0_1', '0_2', '0_3', 'g', 'p']
n_classes = 5


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

    if len(polygons) > 0:
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                color = colors[classes.index(polygon.label)]
                current_axis.add_patch(
                    plt.Polygon(xy=polygon.get_points(), closed=True, color=color, fill=False, linewidth=2))
            else:
                # print(polygon)
                if len(polygon) == 6:
                    lbl, score, xmin, ymin, xmax, ymax = polygon
                else:
                    lbl, xmin, ymin, xmax, ymax = polygon

                if type(lbl) == int :
                    lbl = classes[lbl]
                    color = colors[classes.index(lbl)]
                elif type(lbl) == str:
                    color = colors[classes.index(lbl)]
                elif np.issubdtype(lbl, float) or np.issubdtype(lbl, int):
                    lbl = classes[int(lbl)]
                    color = colors[classes.index(lbl)]
                else:
                    color = colors[classes.index(lbl)]

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
    # if x + width > img.shape[0] or y + height > img.shape[1]:
    if x + width > img.shape[1] or y + height > img.shape[0]:
        return None

    # window_img = img[x:x + width, y:y + height]
    window_img = img[y:y + height, x:x+width]
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
    height, width = img.shape[0], img.shape[1]
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


# polygons = list of lists
# img = ndarray
def create_augmentations(img, polygons, include_orig=True):
    augmentations = {}
    if include_orig:
        augmentations["orig"] = (img, polygons)

    horizontal_flip = img[:, ::-1]
    hor_new_polygons = []
    for polygon in polygons:
        lbl, score, xmin, ymin, xmax, ymax = polygon
        new_xmin = 300 - xmax
        new_xmax = 300 - xmin
        new_polygon = [lbl, score, new_xmin, ymin, new_xmax, ymax]
        hor_new_polygons.append(new_polygon)
    augmentations["hor"] = (horizontal_flip, hor_new_polygons)

    vertical_flip = img[::-1, :]
    ver_new_polygons = []
    for polygon in polygons:
        lbl, score, xmin, ymin, xmax, ymax = polygon
        new_ymin = 300 - ymax
        new_ymax = 300 - ymin
        new_polygon = [lbl, score, xmin, new_ymin, xmax, new_ymax]
        ver_new_polygons.append(new_polygon)
    augmentations["ver"] = (vertical_flip, ver_new_polygons)

    new_roots = {
        90: (0, 300),
        180: (300, 300),
        270: (300, 0)
    }

    rotations = [90, 180, 270]
    for angle in rotations:
        rotated_img = rotate(img, 360-angle) * 255
        rotated_img = rotated_img.astype('uint8')
        new_polygons = []
        for polygon in polygons:
            lbl, score, xmin, ymin, xmax, ymax = polygon
            new_xmin, new_ymin = change_coordinates((xmin, ymin), new_roots[angle], angle)
            new_xmax, new_ymax = change_coordinates((xmax, ymax), new_roots[angle], angle)
            new_polygon = [lbl, score, new_xmin, new_ymin, new_xmax, new_ymax]
            new_polygons.append(new_polygon)

        augmentations[f"rot{angle}"] = (rotated_img, new_polygons)

    return augmentations


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

            if window[0].shape != (300, 300, 3):
                print(f"!!! {filename} :: {window[0].shape}")

            # augs = create_augmentations(window[0], window[1])
            # print(f"-- window {idx} :: created {len(augs)} augmentations...")

            from PIL import Image

            img, polygons = window[0], window[1]

            im = Image.fromarray(img)
            im.save(os.path.join(target_image_path, f"{filename}.jpg"))

            create_xml(polygons, 300, 300, os.path.join(TARGET_ANNOTATIONS, f"{filename}.xml"))

            # for aug_title, aug in augs.items():
            #     img, polygons = aug
            #
            #     im = Image.fromarray(img)
            #     im.save(os.path.join(target_image_path, f"{filename}_{aug_title}.jpg"))
            #
            #     create_xml(polygons, 300, 300, os.path.join(TARGET_ANNOTATIONS, f"{filename}_{aug_title}.xml"))

        print(f"-- {img_id} done")


def create_dataset_split_files(TARGET_DIR, IMAGES_DIR, ANNOTATIONS_DIR):
    filenames = []

    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            filenames.append(file[:-4])

    final = []
    for filename in filenames:
        if os.path.exists(os.path.join(ANNOTATIONS_DIR, f"{filename}.xml")):
            final.append(filename)

    # final = filenames

    print(final[:5])
    print(len(final))
    idx1 = int(0.7 * len(final))
    idx2 = int(0.85 * len(final))

    random.shuffle(final)

    train_files = final[:idx1]
    val_files = final[idx1:idx2]
    test_files = final[idx2:]

    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    with open(os.path.join(TARGET_DIR, "train.txt"), "w") as f:
        f.write("\n".join(train_files) + "\n")

    with open(os.path.join(TARGET_DIR, "val.txt"), "w") as f:
        f.write("\n".join(val_files) + "\n")

    with open(os.path.join(TARGET_DIR, "test.txt"), "w") as f:
        f.write("\n".join(test_files) + "\n")

if __name__ == "__main__":
    DATASET_PATH = os.getenv("DATASET_DIR")

    TARGET_DATASET = os.getenv("TARGET_DATASET_DIR")
    TARGET_IMAGES = os.path.join(TARGET_DATASET, "images")
    TARGET_ANNOTATIONS = os.path.join(TARGET_DATASET, "annotations")

    os.makedirs(TARGET_DATASET, exist_ok=True)
    os.makedirs(TARGET_IMAGES, exist_ok=True)
    os.makedirs(TARGET_ANNOTATIONS, exist_ok=True)

    create_dataset(DATASET_PATH, TARGET_IMAGES, TARGET_ANNOTATIONS)
    create_dataset_split_files(TARGET_DATASET, TARGET_IMAGES, TARGET_ANNOTATIONS)
