# GrayScale-SRCNN

## Data Preprocessing
Use bilinear interpolation to transfer the 64x64 images to 128x128 images for both train and test

    python3 preprocess.py
## Train 
### Train by SRCNN
Use SRCNN model to train the 16000 images with 500 epoch, 1 batch and 1e-4 learning rate (about 10 hours)

    python3 train_srcnn.py
### Train by VDSR
Use VDSR model to train the 16000 images with 500 epoch, 1 batch and 1e-4 learning rate (about 50 hours)

    python3 train_vdsr.py
## Test
Load the final_train_model.pth (trained by VDSR) to output 3999 test images

    python3 output.py
