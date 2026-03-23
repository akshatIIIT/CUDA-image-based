from torchvision import datasets
import torchvision.transforms as transforms
import os
from PIL import Image

dataset = datasets.CIFAR10(root='./data', download=True)

os.makedirs("images/input", exist_ok=True)

for i in range(500):  # take 500 images
    img, label = dataset[i]
    img.save(f"images/input/img_{i}.png")
