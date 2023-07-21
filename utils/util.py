from __future__ import print_function
import torch
import numpy as np
import os
import torch.nn.functional as F
from torchmetrics import Dice
import torch.nn as nn

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dice_coefficient(prediction,label):
    label = label.to(prediction.device)
    dice = Dice(average='micro').to(prediction.device)
    return dice(prediction,label)

#takes in one image at a time
def pad(image):
    pad = nn.ZeroPad2d(1)
    return pad(image)

#takes in a batch of meshes
def unpad(padded_image):
    return padded_image[:, :, 1:-1, 1:-1]