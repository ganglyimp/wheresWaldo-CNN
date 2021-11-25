# ================
#  PREPROCESSING
# ================
import cv2 
import os
import glob
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as tf 
from torchvision import datasets
#torch.set_default_tensor_type(torch.FloatTensor)

print("Loading dataset...")

def loadDirectory(filepath, isOrig):
    # load directory
    files = glob.glob(filepath)

    arr = []
    for fl in files:
        img = cv2.imread(fl)

        # resize image to 256x256
        if(isOrig != True and img.shape[0] != 256): 
            img = cv2.resize(img, (256, 256))

        tenImg = torch.Tensor(img)

        arr.append(tenImg.T)

    # if dataset is original puzzles, return a list of tensors (images are not equal sizes)
    if(isOrig):
        return arr
    
    tensorList = torch.stack(arr)

    # shuffle data
    index = torch.randperm(tensorList.shape[0])
    tensorList = tensorList[index].view(tensorList.size()) 

    return tensorList

originalImg = loadDirectory("./original-images/*.jpg", True)

waldo64 = loadDirectory("./64/waldo/*.jpg", False)
notWaldo64 = loadDirectory("./64/notwaldo/*.jpg", False)

waldo128 = loadDirectory("./128/waldo/*.jpg", False)
notWaldo128 = loadDirectory("./128/notwaldo/*.jpg", False)

waldo256 = loadDirectory("./256/waldo/*.jpg", False)
notWaldo256 = loadDirectory("./256/notwaldo/*.jpg", False)

# Combining into two lists: waldos and not waldos
waldos = torch.cat((waldo64, waldo128, waldo256), 0)
notWaldos = torch.cat((notWaldo64, notWaldo128, notWaldo256), 0)

# Issue: 97 Waldo : 6940 Not Waldo
# Need to augment dataset to inflate Waldo instances

# Create labels & combine
waldoLabels = torch.cat((torch.ones(len(waldos)), torch.zeros(len(notWaldos))), 0)
allWaldos = torch.cat((waldos, notWaldos), 0)

waldoDataset = torch.utils.data.TensorDataset(allWaldos, waldoLabels)
print("Dataset loaded")

# ============
#   THE CNN
# ============
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

# Network should output whether or not the input image has waldo in it
class WaldoFinder(nn.Module):
    def __init__(self):
        super(WaldoFinder, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.batchNorm1 = nn.BatchNorm2d(num_features=256)
        self.dropout1 = nn.Dropout2d(p=0.1)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(num_features=128)
        self.dropout2 = nn.Dropout2d(p=0.1)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(num_features=64)
        self.dropout3 = nn.Dropout2d(p=0.1)

        # Block 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=32, stride=1, padding=0)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    #Forward function - convolvs down to 16x16 image and ultimately outputs 1 or 0
    def forward(self, t):
        t = self.conv1(t)
        t = self.batchNorm1(t)
        t = F.relu(t)
        t = self.dropout1(t)
        t = self.maxPool(t)

        t = self.conv2(t)
        t = self.batchNorm2(t)
        t = F.relu(t)
        t = self.dropout2(t)
        t = self.maxPool(t)

        t = self.conv3(t)
        t = self.batchNorm3(t)
        t = F.relu(t)
        t = self.dropout3(t)
        t = self.maxPool(t)

        t = self.conv4(t)

        t = torch.round(t)

        return t

waldoFinder = WaldoFinder()
print("Network Initialized")

# Divide into training and test set
tenPercent = int(len(waldoDataset) * 0.1)
ninetyPercent = len(waldoDataset) - tenPercent
trainSet, testSet = torch.utils.data.random_split(waldoDataset, [ninetyPercent, tenPercent])

trainLoader = torch.utils.data.DataLoader(trainSet, shuffle=True, batch_size=10)
testLoader = torch.utils.data.DataLoader(testSet, shuffle=True, batch_size=10)
print("Dataset shuffled")

# Train Loop
optimizer = optim.Adam(waldoFinder.parameters(), lr=.01)
#lossFunc = nn.CrossEntropyLoss()
lossFunc = nn.BSELoss()

print("Begining training...")
for items, labels in trainLoader:
    optimizer.zero_grad()
    preds = waldoFinder(items).unsqueeze(dim=0)

    loss = lossFunc(preds, labels.unsqueeze(dim=0))
    loss.backward()
    optimizer.step()
    print("One loop completed")

# Test Loop
#correct = 0
#for items, labels in testLoader:
#    preds = waldoFinder(items)
#    if preds == labels:
#        correct = correct + 1

# Output stats for AI