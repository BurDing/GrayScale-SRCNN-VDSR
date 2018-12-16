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
data_loader = DataLoader(dataset=train_data, batch_size=batch, num_workers = 12, shuffle=True)

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
    if i % 10 == 0:
        print("epoch: " + str(i) + " loss: " + str(loss_sum) + " loss_input: " + str(loss_input_sum))
    if i % 100 == 0:
        file_name = str(i) + "_" + "train_model.pth"
        print("Save loss: " + str(loss_sum) + " Name: " + file_name)
        torch.save(net, "model/" + file_name)

torch.save(net, "final_train_model.pth")
# model = torch.load("train_model.pth")


#  test
# for i in range(0, test_size):
#     t = train_data[i].view(1, 1, 128, 128)
#     if cuda:
#         model = model.cuda()
#         t = t.cuda()
#     out = model(t).view(128,128).detach().numpy()
#     imageio.imwrite('upload/' + test_files[i], out, format=None)
