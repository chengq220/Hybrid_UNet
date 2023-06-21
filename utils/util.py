from __future__ import print_function
import torch
import numpy as np
import os
import torch.nn.functional as F
from torchmetrics import Dice

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dice_coefficient(prediction,label):
    label = label.to(prediction.device)
    dice = Dice(average='micro').to(prediction.device)
    return dice(prediction,label)