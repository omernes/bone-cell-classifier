from os import getenv, walk, path
import random

IMAGES_DIR = getenv("IMAGES_DIR")
ANNOTATIONS_DIR = getenv("ANNOTATIONS_DIR")
TARGET_DIR = getenv("TARGET_DIR")

filenames = []

for root, dirs, files in walk(IMAGES_DIR):
  for file in files:
    filenames.append(file[:-4])

final = []
for filename in filenames:
  if path.exists(path.join(ANNOTATIONS_DIR, f"{filename}.xml")):
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

# lines = []
# for file in train_files:
#   lines.append(file)

with open(path.join(TARGET_DIR, "train.txt"), "w") as f:
  f.write("\n".join(train_files) + "\n")

with open(path.join(TARGET_DIR, "val.txt"), "w") as f:
  f.write("\n".join(val_files) + "\n")

# lines = []
# for file in test_files:
#   lines.append(path.join(raw_path, file) + "\t" + os.path.join(raw_lbls_path, file))

with open(path.join(TARGET_DIR, "test.txt"), "w") as f:
  f.write("\n".join(test_files) + "\n")