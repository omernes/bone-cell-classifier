from os import getenv, walk
import xml.etree.ElementTree as ET

IMAGES_DIR = getenv("IMAGES_DIR", "/a/home/cc/students/csguests/omernestor/data_xml/images")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR", "/a/home/cc/students/csguests/omernestor/data_xml/annotations")

IMAGESET_TEST = getenv("IMAGESET_TEST", "/a/home/cc/students/csguests/omernestor/data_xml/test.txt")

files_to_remove = []

for root, dirs, files in walk(ANNOTATIONS_DIR):
    for file in files:
        if root != ANNOTATIONS_DIR:
            continue

        with open(file, "r") as f:
            content = f.read()

        tree = ET.parse(content)
        root = tree.getroot()

        list_with_all_boxes = []

        print(file)

        has_objects = False
        for boxes in root.iter('object'):
            print(boxes)
            has_objects = True
            break

        if not has_objects:
            files_to_remove.append(file[:-4])

print(files_to_remove)

with open(IMAGESET_TEST, "r") as f:
    test_files = f.readlines()

new_test_files = []
for filename in test_files:
    if filename not in files_to_remove:
        new_test_files.append(filename)

with open(IMAGESET_TEST[:-4] + "_new.txt", "w") as f:
    f.write("\n".join(new_test_files))

print("\n".join(new_test_files))