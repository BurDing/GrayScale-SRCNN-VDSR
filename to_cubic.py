from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.transform
import imageio

# Read all iamges
train_files = os.listdir('../train_images_128x128')
test_files = os.listdir('../test_images_64x64')

print(test_files)
# Obtain 128x128 cubic image
for ima in train_files:
    img = imageio.imread('../train_images_64x64/'+ima)
    out = skimage.transform.resize(img, (128, 128), mode = 'constant')
    imageio.imwrite('../cubic_train/'+ima, out, format=None)

for ima in test_files:
    img = imageio.imread('../test_images_64x64/'+ima)
    out = skimage.transform.resize(img, (128, 128), mode = 'constant')
    imageio.imwrite('../cubic_test/'+ima, out, format=None)
