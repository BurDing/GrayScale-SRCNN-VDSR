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

epoch = 50
cuda = True
batch = 64
train_size = 1600
test_size = 5

# Read all iamges
train_files = os.listdir('cubic_train')
train_labels_files = os.listdir('train_images_128x128')
test_files = os.listdir('cubic_test')
# Obtain greyscale arrays with given size
train_imgs = np.zeros((train_size, 128, 128))
train_imgs_labels = np.zeros((train_size, 128, 128))
test_imgs = np.zeros((test_size, 128, 128))
for i in range(0, train_size):
    train_imgs[i] = np.array(Image.open('cubic_train/' + train_files[i]).convert('L'))
    train_imgs_labels[i] = np.array(Image.open('train_images_128x128/' + train_labels_files[i]).convert('L'))
for i in range(0, test_size):
    test_imgs[i] = np.array(Image.open('cubic_test/' + test_files[i]).convert('L'))
# data to tensor
train_data = torch.FloatTensor(train_imgs).view(train_size, 1, 128, 128)
target_data = torch.FloatTensor(train_imgs_labels).view(train_size, 1, 128, 128)
test_data = torch.FloatTensor(test_imgs).view(test_size, 1, 128, 128)
data_loader = DataLoader(dataset=(train_data,target_data), batch_size=batch, shuffle=True)

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

# Build model
net = SRCNN()
criterion = nn.MSELoss()
if cuda:
    srcnn.cuda()
    criterion = criterion.cuda()
# create optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)
# Train
for i in range(0, epoch):
    for step, (input, target) in enumerate(data_loader):
        if cuda:
            input = input.cuda()
            target = target.cuda()
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
        print("# of batch:" + step)
    if i%1 == 0:
        print("epoch:" + i + "loss:" + loss)

torch.save(net, "train_model.pth")
# model = torch.load("train_model.pth")


#  test
# for i in range(0, test_size):
#     t = train_data[i].view(1, 1, 128, 128)
#     if cuda:
#         model = model.cuda()
#         t = t.cuda()
#     out = model(t).view(128,128).detach().numpy()
#     imageio.imwrite('upload/' + test_files[i], out, format=None)
