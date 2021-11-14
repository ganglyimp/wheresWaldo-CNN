# ================
#  PREPROCESSING
# ================
import cv2 
import os
import glob
import torch
import random
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

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

        arr.append(torch.Tensor(img))

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


# ============
#   THE CNN
# ============
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as tf 
import torch.optim as optim

# Network should output whether or not the input image has waldo in it
class WaldoFinder(nn.Module):
    def __init__(self):
        super(WaldoFinder, self).__init__()

        C = 256 #in/out channels
        K = 3 #kernel size

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=C, kernel_size=K, stride=2, padding=1)

        # Scales weights by gain parameter
        nn.init.xavier_uniform_(self.conv1.weight)
    
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)

        return t

waldoFinder = WaldoFinder()

# Divide into training and test set
tenPercent = int(len(waldoDataset) * 0.1)
ninetyPercent = len(waldoDataset) - tenPercent
trainSet, testSet = torch.utils.data.random_split(waldoDataset, [ninetyPercent, tenPercent])

trainLoader = torch.utils.data.DataLoader(trainSet, shuffle=True, batch_size=10)
testLoader = torch.utils.data.DataLoader(testSet, shuffle=True, batch_size=10)

# Train Loop
optimizer = optim.Adam(waldoFinder.parameters(), lr=.01)
lossFunc = nn.MSELoss()

'''
for items, labels in trainLoader:
    preds = waldoFinder(items)

    loss = lossFunc(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

# Test Loop
