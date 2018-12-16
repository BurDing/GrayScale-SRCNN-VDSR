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
from math import sqrt

epoch = 500
cuda = True
batch = 1
train_size = 16000

print("read")
# Read all iamges
train_files = os.listdir('cubic_train')
train_labels_files = os.listdir('train_images_128x128')
# Obtain greyscale arrays with given size
train_imgs = np.zeros((train_size, 128, 128))
train_imgs_labels = np.zeros((train_size, 128, 128))
for i in range(0, train_size):
    train_imgs[i] = np.array(Image.open('cubic_train/' + train_files[i]).convert('L'))
    train_imgs_labels[i] = np.array(Image.open('train_images_128x128/' + train_labels_files[i]).convert('L'))
# data to tensor
train_data = [(torch.FloatTensor(train_imgs[i]).view(1, 128, 128),  torch.FloatTensor(train_imgs_labels[i]).view(1, 1, 128, 128)) for i in range(0, train_size)]
data_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)

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

# Build model
net = VDSR()
criterion = nn.MSELoss()
if cuda:
    net.cuda()
    criterion = criterion.cuda()
# create optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)
# Train
print("Begin")
for i in range(0, epoch):
    loss_sum= 0
    loss_input_sum = 0
    for step, (input, target) in enumerate(data_loader):
        if cuda:
            input = input.cuda()
            target = target.cuda()
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss_sum += loss.item()
        loss_input = criterion(input, target)
        loss_input_sum += loss_input.item()
        loss.backward()
        optimizer.step()    # Does the update
    print("epoch: " + str(i) + " loss: " + str(loss_sum / len(data_loader)) + " loss_input: " + str(loss_input_sum / len(data_loader)))
    if i % 10 == 0:
        file_name = str(i) + "_" + "train_model.pth"
        print("Save loss: " + str(loss_sum / len(data_loader)) + " Name: " + file_name)
        torch.save(net, "model/" + file_name)

torch.save(net, "final_train_model.pth")
