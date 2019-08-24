from PIL import Image
import numpy
import matplotlib.pyplot as plt

from ssd_keras.bounding_box_utils.bounding_box_utils import iou, intersection_area
from model import Model

classes = ['background', '0_1', '0_2', '0_3', 'p', 'g']
n_classes = 5


def process_image(path):  # path should be path to image
    # create matrix from img
    im = Image.open(path)
    np_im = numpy.array(im)

    # send to sliding window, get list of windows and cords
    windows, cords = create_sliding_windows(img=np_im, size=(300, 300), step=200)

    # send each window to predict get back a list of boxes, put in index i of window in pred final list
    preds = []
    model = Model()
    for window in windows:
        pred = model.predict(window)
        preds.append(pred)

    # send to get abs boxes, get list of all boxes in abs axes
    boxes = get_absolute_boxes(cords, preds)

    best_boxes = find_best_boxes(boxes)

    # display_image(im, boxes)
    # display_image(im, best_boxes)

    print(f"boxes :: {len(boxes)}")
    print(f"best boxes :: {len(best_boxes)}")

    counts = count_cells(best_boxes)
    print(counts)

    # save results to json (counters and boxes)
    # save plot as image (with boxes)


def display_image(img, boxes):
    colors = plt.cm.hsv(numpy.linspace(0, 1, n_classes + 1)).tolist()
    plt.figure(figsize=(20, 12))
    plt.imshow(img)

    current_axis = plt.gca()

    if len(boxes) > 0:
        for box in boxes:
            if len(box) == 6:
                lbl, score, xmin, ymin, xmax, ymax = box
            else:
                lbl, xmin, ymin, xmax, ymax = box

            if type(lbl) == int:
                lbl = classes[lbl]
                color = colors[classes.index(lbl)]
            elif type(lbl) == str:
                color = colors[classes.index(lbl)]
            elif numpy.issubdtype(lbl, float) or numpy.issubdtype(lbl, int):
                lbl = classes[int(lbl)]
                color = colors[classes.index(lbl)]
            else:
                color = colors[classes.index(lbl)]

            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, lbl, size='x-large', color='white',
                              bbox={'facecolor': 'green', 'alpha': 1.0})

    plt.show()


# cords is a list of touples (one for each window, indexed by the windows) and holds the top left cords for each window [left,top]
# preds_by_window is a list of lists, each index i holds a  list with all the box predictions for window i
def get_absolute_boxes(cords, preds_by_window):  # cords holds the top left cords for each window [left,top]
    total_preds = []
    for i in range(len(cords)):
        x_abs = cords[i][0]
        y_abs = cords[i][1]
        for box in preds_by_window[i]:
            curr_pred = [box[0], box[1], box[2] + x_abs, box[3] + y_abs, box[4] + x_abs, box[5] + y_abs]
            total_preds.append(curr_pred)
    return total_preds


def cut_window_from_image(img, xy=(0, 0), window_size=(300, 300), angle=0):
    x, y = xy
    width, height = window_size
    return img[y:y + height, x:x + width]


def create_sliding_windows(img, size=(300, 300), step=300):
    height, width = img.shape[0], img.shape[1]
    win_width, win_height = size

    windows = []
    cords = []

    for x in range(0, width - win_width, step):
        for y in range(0, height - win_height, step):
            win_img = cut_window_from_image(img, xy=(x, y), window_size=size)
            windows.append(win_img)
            cords.append((x, y))
        win_img = cut_window_from_image(img, xy=(x, height - win_height), window_size=size)
        windows.append(win_img)
        cords.append((x, height - win_height))
    x = width - win_width
    for y in range(0, height - win_height, step):
        win_img = cut_window_from_image(img, xy=(x, y), window_size=size)
        windows.append(win_img)
        cords.append((x, y))
    win_img = cut_window_from_image(img, xy=(x, height - win_height), window_size=size)
    windows.append(win_img)
    cords.append((x, height - win_height))

    return windows, cords


def find_best_boxes(predictions, area_threshold=0.75, iou_threshold=0.5):
    predictions = predictions.copy()
    predictions.sort(key=lambda pred: (pred[4] - pred[2]) * (pred[5] - pred[3]), reverse=True)

    boxes = []

    while len(predictions) > 0:
        pred = predictions.pop(0)
        boxes.append(pred)
        pred_coords = numpy.array(pred[2:])
        boxes_to_remove = []
        for compare in predictions:
            compare_coords = numpy.array(compare[2:])
            area = intersection_area(numpy.array([pred_coords]), numpy.array([compare_coords]), coords="corners")[0][0]
            pred_area = (pred_coords[2] - pred_coords[0]) * (pred_coords[3] - pred_coords[1])
            pred_area_perc = area / pred_area
            compare_pred_area = (compare_coords[2] - compare_coords[0]) * (compare_coords[3] - compare_coords[1])
            compare_pred_area_perc = area / compare_pred_area

            if compare_pred_area_perc > area_threshold:
                boxes_to_remove.append(compare)

        for rem in boxes_to_remove:
            predictions.remove(rem)

    final_boxes = []
    iou_overlaps = []
    while len(boxes) > 0:
        pred = boxes.pop(0)
        pred_coords = numpy.array(pred[2:])

        for compare in boxes:
            compare_coords = numpy.array(compare[2:])
            if iou(pred_coords, compare_coords, 'corners', 'element-wise') >= iou_threshold:
                iou_overlaps.append(compare)

        if len(iou_overlaps) > 0:
            iou_overlaps.sort(key=lambda x: x[1], reverse=True)
            if iou_overlaps[1] > pred[1]:
                final_boxes.append(iou_overlaps[0])
                if iou_overlaps[0] in boxes:
                    boxes.remove(iou_overlaps[0])
            else:
                final_boxes.append(pred)
        else:
            final_boxes.append(pred)

    return final_boxes


def count_cells(boxes):
    ghost_cnt, pre_cnt, o1_cnt, o2_cnt, o3_cnt, total_o_cnt = 0, 0, 0, 0, 0, 0
    for box in boxes:
        label = classes[int(box[0])]
        if label == "p":
            pre_cnt = pre_cnt + 1
        elif label == "g":
            ghost_cnt = ghost_cnt + 1
        elif label == "0_1":
            o1_cnt = o1_cnt + 1
        elif label == "0_2":
            o2_cnt = o2_cnt + 1
        elif label == "0_3":
            o3_cnt = o3_cnt + 1
    total_o_cnt = o1_cnt + o2_cnt + o3_cnt
    return total_o_cnt, o1_cnt, o2_cnt, o3_cnt, ghost_cnt, pre_cnt


if __name__ == "__main__":
    img_path = "data/raw/C2_03_02.png"
    process_image(img_path)
