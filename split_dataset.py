from os import getenv, walk, path
import random

IMAGES_DIR = getenv("IMAGES_DIR")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR")
TARGET_DIR = getenv("TARGET_DIR")

filenames = []

for root, dirs, files in walk(IMAGES_DIR):
  for file in files:
    filenames.append(file[:-4])

# final = []
# for filename in filenames:
#   if path.exists(path.join(ANNOTATIONS_DIR, f"{filename[:-3]}.xml")):
#     final.append(filename)

final = filenames

print(final[:5])
print(len(final))
idx = int(0.8 * len(final))

random.shuffle(final)

train_files = final[:idx]
test_files = final[idx:]

print(len(train_files))
print(len(test_files))

# lines = []
# for file in train_files:
#   lines.append(file)

with open(path.join(TARGET_DIR, "train.txt"), "w") as f:
  f.write("\n".join(train_files) + "\n")

# lines = []
# for file in test_files:
#   lines.append(path.join(raw_path, file) + "\t" + os.path.join(raw_lbls_path, file))

with open(path.join(TARGET_DIR, "test.txt"), "w") as f:
  f.write("\n".join(test_files) + "\n")