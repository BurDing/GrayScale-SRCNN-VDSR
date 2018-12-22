from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.transform
import imageio
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from math import sqrt

test_size = 3999
cuda = True

# read the test file to tensor
print("read")
test_files = os.listdir('cubic_test')
test_imgs = np.zeros((test_size, 128, 128))
for i in range(0, test_size):
    test_imgs[i] = np.array(Image.open('cubic_test/' + test_files[i]).convert('L'))
test_data = [torch.FloatTensor(i).view(1, 128, 128) for i in test_imgs]

# model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=9,padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(32,1,kernel_size=5,padding=2);

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

# load the model
model = torch.load("model/1083_train_model.pth", map_location='cpu')
# output the result of test
for t in range(0, test_size):
    print(t)
    I = np.rint(model(test_data[t].view(1, 1, 128, 128)).view(128,128).detach().numpy())
    # deal with the negative and larger than 255 pixel
    for i in range(0, 128):
        for j in range (0, 128):
            if I[i][j] < 0:
                I[i][j] = 0
            if I[i][j] > 255:
                I[i][j] = 255
    I = I.astype(np.uint8)
    img = Image.fromarray(I)
    img.save('upload/' + test_files[t])
