import cv2 
import os
import glob
import torch
import random
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

print("Loading dataset...")

# All image directory paths
originalImg_dir = "./original-images/*.jpg"

waldo64_dir = "./64/waldo/*.jpg"
notWaldo64_dir = "./64/notwaldo/*.jpg"

waldo128_dir = "./128/waldo/*.jpg"
notWaldo128_dir = "./128/notwaldo/*.jpg"

waldo256_dir = "./256/waldo/*.jpg"
notWaldo256_dir = "./256/notwaldo/*.jpg"

# Loading directories
origImg_files = glob.glob(originalImg_dir)

waldo64_files = glob.glob(waldo64_dir)
waldo128_files = glob.glob(waldo128_dir)
waldo256_files = glob.glob(waldo256_dir)

notWaldo64_files = glob.glob(notWaldo64_dir)
notWaldo128_files = glob.glob(notWaldo128_dir)
notWaldo256_files = glob.glob(notWaldo256_dir)

def loadDirectory(files):
    arr = []
    for fl in files:
        img = cv2.imread(fl)
        arr.append(img)

    return arr 

# Arrays containing all image info
origImg_data = loadDirectory(origImg_files)

waldo64_data = loadDirectory(waldo64_files)
waldo128_data = loadDirectory(waldo128_files)
waldo256_data = loadDirectory(waldo256_files)

notWaldo64_data = loadDirectory(notWaldo64_files)
notWaldo128_data = loadDirectory(notWaldo128_files)
notWaldo256_data = loadDirectory(notWaldo256_files)

# Converting each array into a tensor
origImg_tensor = torch.tensor(origImg_data)

waldo64_tensor = torch.tensor(waldo64_data)
waldo128_tensor = torch.tensor(waldo128_data)
waldo256_tensor = torch.tensor(waldo256_data)

notWaldo64_tensor = torch.tensor(waldo64_data)
notWaldo128_tensor = torch.tensor(waldo128_data)
notWaldo256_tensor = torch.tensor(waldo256_data)

# Randomly shuffle the data using torch.randperm
def shuffleData(theTensor):
    index = torch.randperm(theTensor.shape[0])
    theTensor = theTensor[index].view(theTensor.size())
    return theTensor

origImg_tensor = shuffleData(origImg_tensor)

waldo64_tensor = shuffleData(waldo64_tensor)
waldo128_tensor = shuffleData(waldo128_tensor)
waldo256_tensor = shuffleData(waldo256_tensor)

notWaldo64_tensor = shuffleData(notWaldo64_tensor)
notWaldo128_tensor = shuffleData(notWaldo128_tensor)
notWaldo256_tensor = shuffleData(notWaldo256_tensor)