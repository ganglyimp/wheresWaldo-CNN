# ================
#  PREPROCESSING
# ================
import glob
import random
import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as tf 
from torchvision import datasets
#torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda:0")

def loadDirectory(filepath):
    # load directory
    files = glob.glob(filepath)

    arr = []
    for fl in files:
        img = cv2.imread(fl)

        # resize image to 256x256
        if(img.shape[0] != 256): 
            img = cv2.resize(img, (256, 256))

        arr.append(img)

    return arr

def numpyToTensor(arr):
    arr = np.array(arr)
    arr = arr.transpose((0, 3, 1, 2))
    tensorList = torch.FloatTensor(arr)

    return tensorList

def createNewWaldoSamples(notWaldos, overlay):
    newWaldos = []
    oRows, oCols, oChannels = overlay.shape
    for img in notWaldos:
        newImg = img
        iRows, iCols, iChannels = newImg.shape
        newImg = np.dstack([newImg, np.ones((iRows, iCols), dtype='uint8') * 255])

        randX = random.randint(0, iRows-oRows)
        randY = random.randint(0, iCols-oCols)

        aOverlay = overlay[:, :, 3] / 255.0
        aImg = 1.0 - aOverlay

        for c in range(0, 3):
            newImg[randX:randX+oRows, randY:randY+oCols, c] = (aOverlay * overlay[:, :, c] + aImg * newImg[randX:randX+oRows, randY:randY+oCols, c])

        newWaldos.append(newImg[:, :, :3])
    
    return newWaldos

def createWaldoDataset():
    # Note: WaldoOverlay is a png that will be overlayed on top of notWaldo samples in order to generate more Waldo samples

    print("Loading in Waldo images...")
    # Loading in 64x64 images
    waldo64 = loadDirectory("./64/waldo/*.jpg")
    notWaldo64 = loadDirectory("./64/notwaldo/*.jpg")
    waldoOverlay64 = cv2.imread('./waldo64.png', cv2.IMREAD_UNCHANGED)
    moreWaldo64 = createNewWaldoSamples(notWaldo64, waldoOverlay64)

    # Converting 64x64 images into tensors
    waldo64Tensor = numpyToTensor(waldo64)
    moreWaldo64Tensor = numpyToTensor(moreWaldo64)
    notWaldo64Tensor = numpyToTensor(notWaldo64)

    # Free up memory
    del waldo64
    del notWaldo64
    del waldoOverlay64
    del moreWaldo64

    # Loading in 128x128 images
    waldo128 = loadDirectory("./128/waldo/*.jpg")
    notWaldo128 = loadDirectory("./128/notwaldo/*.jpg")
    waldoOverlay128 = cv2.imread('./waldo128.png', cv2.IMREAD_UNCHANGED)
    moreWaldo128 = createNewWaldoSamples(notWaldo128, waldoOverlay128)

    # Converting 128x128 images into tensors
    waldo128Tensor = numpyToTensor(waldo128)
    moreWaldo128Tensor = numpyToTensor(moreWaldo128)
    notWaldo128Tensor = numpyToTensor(notWaldo128)

    # Freeing up memory
    del waldo128
    del notWaldo128
    del waldoOverlay128
    del moreWaldo128

    # Loading in 256x256 images
    waldo256 = loadDirectory("./256/waldo/*.jpg")
    notWaldo256 = loadDirectory("./256/notwaldo/*.jpg")
    waldoOverlay256 = cv2.imread('./waldo256.png', cv2.IMREAD_UNCHANGED)
    moreWaldo256 = createNewWaldoSamples(notWaldo256, waldoOverlay256)

    # Converting 256x256 images into tensors
    waldo256Tensor = numpyToTensor(waldo256).to(device)
    moreWaldo256Tensor = numpyToTensor(moreWaldo256).to(device)
    notWaldo256Tensor = numpyToTensor(notWaldo256).to(device)

    # Freeing up memory
    del waldo256
    del notWaldo256
    del waldoOverlay256
    del moreWaldo256

    # Combining into one list
    waldos = torch.cat((waldo64Tensor, waldo128Tensor, waldo256Tensor, moreWaldo64Tensor, moreWaldo128Tensor, moreWaldo256Tensor), 0)
    notWaldos = torch.cat((notWaldo64Tensor, notWaldo128Tensor, notWaldo256Tensor), 0)

    # Freeing up ALL that memory
    del waldo64Tensor
    del waldo128Tensor
    del waldo256Tensor
    del moreWaldo64Tensor
    del moreWaldo128Tensor
    del moreWaldo256Tensor
    del notWaldo64Tensor
    del notWaldo128Tensor
    del notWaldo256Tensor

    # All values between 0 and 1
    waldos = waldos / 255.0
    notWaldos = notWaldos / 255.0

    # Create labels & combine
    waldoLabels = torch.cat((torch.ones(len(waldos)), torch.zeros(len(notWaldos))), 0)
    allWaldos = torch.cat((waldos, notWaldos), 0)

    waldoSet = torch.utils.data.TensorDataset(allWaldos, waldoLabels)

    return waldoSet

waldoDataset = createWaldoDataset()
print("Dataset loaded.")

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

        return t

waldoFinder = WaldoFinder()
waldoFinder = WaldoFinder().to(device)
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
lossFunc = nn.BCEWithLogitsLoss()

i = 0
print("Starting training loop...")
for items, labels in trainLoader:
    optimizer.zero_grad()
    preds = waldoFinder(items).squeeze()

    loss = lossFunc(preds, labels)
    loss.backward()
    optimizer.step()

    print("Batch ", i, ": ", loss.item())
    i += 1

# Test Loop
print("Starting test loop...")
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

totalLoss = 0.0
epochCount = 0

waldoFinder.eval()
for items, labels in testLoader:
    preds = waldoFinder(items).squeeze()

    loss = lossFunc(preds, labels)
    totalLoss += loss.item()
    epochCount += 1

    preds = torch.round(preds)
    
    for i in range(len(labels)):
        if(labels[i] == 1):
            if(labels[i] == preds[i]):
                truePositive += 1
            else:
                falseNegative += 1
        else:
            if(labels[i] == preds[i]):
                trueNegative += 1
            else:
                falsePositive += 1


# Output stats for AI
print("True Positive: ", truePositive)
print("True Negative: ", trueNegative)
print("False Positive: ", falsePositive)
print("False Negative: ", falseNegative)
print()

totalCorrect = truePositive + trueNegative
totalIncorrect = falsePositive + falseNegative
meanLoss = totalLoss / epochCount
print("Total Correct: ", totalCorrect)
print("Total Incorrect: ", totalIncorrect)
print("Average Loss: ", meanLoss)
