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