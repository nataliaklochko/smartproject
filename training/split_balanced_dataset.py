import os
import shutil
import random


threshold = 0.1
count = 0

for t in os.listdir(os.path.join("image_dataset_balanced", "train")):
    print(t)
    for f in os.listdir(os.path.join("image_dataset_balanced", "train", t)):
        n = random.random()
        if n < threshold:
            source = os.path.join("image_dataset_balanced", "train", t, f)
            dist = os.path.join("image_dataset_balanced", "test", t)
            shutil.move(source, dist)
            count += 1

print(count)
