import os
from PIL import Image

os.makedirs("images/input", exist_ok=True)

for i in range(500):  # take 500 images
    img, label = dataset[i]
    img.save(f"images/input/img_{i}.png")
