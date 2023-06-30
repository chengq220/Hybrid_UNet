import torch
import torch.nn as nn
import numpy as np
from utils import pad

class MeshUnpool(nn.Module):
    def __init__(self,pad):
        super(MeshUnpool, self).__init__()
        self.__pad = pad

    def __call__(self, out,mask, in_images, vc_order):
        for i in range(len(vc_order)):
            vc_order[i] = pad(vc_order[i],self.__pad)
        vc_order = np.stack(vc_order)
        return self.forward(out,mask,in_images,vc_order)

    def forward(self, out,mask,in_images,vc_order):
        out_images = torch.zeros_like(out)
        # mask = torch.from_numpy(mask)
        for idx in range(out.shape[0]):
            out_images[idx][mask[idx]] = in_images[idx]
        #reconstruct the image in reverse order
        for edge in vc_order[::-1]:
            out_images[:,edge[:,1]] = out_images[:,edge[:,0]]
        return out_images