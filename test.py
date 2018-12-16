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
import torch.optim as optim
from torch.utils.data import DataLoader

test_size = 3999
batch = 128
cuda = True

print("read")
test_files = os.listdir('cubic_test')
test_imgs = np.zeros((test_size, 128, 128))
for i in range(0, test_size):
    test_imgs[i] = np.array(Image.open('cubic_test/' + test_files[i]).convert('L'))
test_data = [torch.FloatTensor(i).view(1, 128, 128) for i in test_imgs]
test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

model = torch.load("model/500_train_model.pth")

# test
for step, input in enumerate(test_loader):
    print(step)
    if cuda:
        model = model.cuda()
        input = input.cuda()
    out = model(input).view(batch,128,128).detach().numpy()
    for j in range(0, batch):
        imageio.imwrite('upload/' + test_files[step * batch + j], out[j], format=None)
