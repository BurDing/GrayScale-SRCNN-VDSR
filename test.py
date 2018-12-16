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
