import glob
import random
import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as tf 
from torchvision import datasets

# Load in original Waldo puzzle set.
def loadDirectory(filepath):
    # load directory
    files = glob.glob(filepath)

    arr = []
    for fl in files:
        img = cv2.imread(fl)

        arr.append(img)

    return arr

def numpyToTensor(arr):
    arr = np.array(arr)
    arr = arr.transpose((2, 0, 1))
    tensorList = torch.FloatTensor(arr)
    tensorList = tensorList / 255.0

    return tensorList

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

# Break up into grid squares and then run through the network. 
# Take square with highest waldo probability, and then break that square up into smaller squares
# Repeat process until no squares have a positive Waldo in them. Take coordinate of last Waldo square.
# Figure out a way to draw a square around where Waldo is likely to be. 

def solveTheWaldo(thePuzzle):
    # Divide an image into equal sized tiles 
    height = thePuzzle.shape[0]
    width = thePuzzle.shape[1]

    numRows = int(np.ceil(height / 256.0))
    numCols = int(np.ceil(width / 256.0))

    thePuzzle = cv2.resize(thePuzzle, (256*numRows, 256*numCols))

    for y in range(0, height, 256):
        for x in range(0, width, 256):
            square = thePuzzle[y:y+256, x:x+256]
            cv2.imshow("A square", square)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    # Resize tiles to 256x256
    # Run tiles through CNN
    # Find tile with highest probability of containing waldo.
    # Repeat loop

print("INITIALIZING")

print("\tLoading in image set...")
waldoPuzzles = loadDirectory("./original-images/*.jpg")

print("\tLoading in weights for network...")
waldoFinder = WaldoFinder()
waldoFinder.load_state_dict(torch.load("waldoWeights.pth", map_location=torch.device("cpu")))
waldoFinder.eval()

print()
print("=====================")
print("   WHERE IS WALDO?")
print("=====================")

exitProgram = False
while exitProgram == False:
    print("Select an action: ")
    print("1. Select a Waldo puzzle to solve.")
    print("2. Exit.")
    print()

    command = input('Enter Number: ')
    print()

    if command == "1":
        exitLoop = False
        while exitLoop == False:
            print("Current number of Waldo puzzles: ", len(waldoPuzzles))
            print("Enter a number [ 1 -", len(waldoPuzzles), "] to select the Waldo puzzle to solve.")
            print()

            wallyNum = -1
            try:
                wallyNum = int(input("Enter Number: "))
            except:
                print("Invalid input. Please enter a number.")
                print()
                continue

            if wallyNum < 0 or wallyNum > len(waldoPuzzles):
                print("Input out of range.")
                print()
                continue
            
            solveTheWaldo(waldoPuzzles[wallyNum-1])

            exitLoop = True
    elif command == "2":
        print("Exiting program...")
        print()
        exitProgram = True
    else:
        print("Invalid command.")
        print()