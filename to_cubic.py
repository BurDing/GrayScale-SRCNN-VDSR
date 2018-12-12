print("1")
from PIL import Image
print("2")
import numpy as np
print("3")
import matplotlib.pyplot as plt
print("4")
import os
print("5")
import cv2
print("6")
import torch
print("7")
import torch.nn as nn
print("8")
import torch.nn.functional as F
print("9")
import skimage.transform
print("10")
import imageio
print("11")

# Read all iamges
train_files = os.listdir('train_images_128x128')
test_files = os.listdir('test_images_64x64')

# Obtain 128x128 cubic image
for ima in train_files:
    img = imageio.imread('train_images_64x64/'+ima)
    out = skimage.transform.resize(img, (128, 128), mode = 'constant')
    imageio.imwrite('cubic_train/'+ima, out, format=None)

for ima in test_files:
    img = imageio.imread('test_images_64x64/'+ima)
    out = skimage.transform.resize(img, (128, 128), mode = 'constant')
    imageio.imwrite('cubic_test/'+ima, out, format=None)
