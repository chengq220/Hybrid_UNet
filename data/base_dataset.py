import torch.utils.data as data
import numpy as np
import pickle
import os

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        super(BaseDataset, self).__init__()

def collate_fn(batch):
    images = []
    labels = []

    # Collect images and labels from the batch
    for item in batch:
        images.append(item[0])
        labels.append(item[1])

    # Stack images and labels into tensors
    images = np.stack(images)
    labels = np.stack(labels)

    return {"features":images, "labels":labels}