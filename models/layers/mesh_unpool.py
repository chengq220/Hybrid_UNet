import torch
import torch.nn as nn
import numpy as np

class MeshUnpool(nn.Module):
    def __init__(self):
        super(MeshUnpool, self).__init__()

    def __call__(self, out,mask, in_images, vc_order):
        vc_order = np.stack(vc_order)
        return self.forward(out,mask,in_images,vc_order)

    def forward(self, out,mask,in_images,vc_order):
        out_images = torch.zeros_like(out)
        #reconstruct the image
        for idx in range(out.shape[0]):
            out_images[idx][mask[idx]] = in_images[idx]
        #reconstruct the image in reverse order
        for edge in vc_order[::-1]:
            out_images[:,edge[:,1]] = out_images[:,edge[:,0]]
        return out_images