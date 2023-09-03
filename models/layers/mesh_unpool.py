import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import SplineConv

class MeshUnpool(nn.Module):
    def __init__(self):
        super(MeshUnpool, self).__init__()

    def __call__(self, meshes, images):
        return self.forward(meshes, images)

    def forward(self, meshes, images):
        out_image = []

        for idx, mesh in enumerate(meshes): #iterate over each mesh
            img = images[idx]
            mask, order = mesh.unroll()
            v_f = torch.zeros(mask.shape[0],img.shape[1]).to(img.device)
            v_f[mask] = img
            #reconstruct the image in reverse order
            for idx in range(len(order[0])):
                t = order[1, len(order[0]) - idx - 1]
                f = order[0, len(order[0]) - idx - 1]
                v_f[t] = v_f[f]
            out_image.append(v_f)
        return meshes, torch.stack(out_image)