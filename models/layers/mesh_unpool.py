import torch
import torch.nn as nn
import numpy as np

class MeshUnpool(nn.Module):
    def __init__(self):
        super(MeshUnpool, self).__init__()

    def __call__(self, meshes, adjs, image):
        return self.forward(meshes, adjs, image)

    def forward(self, meshes, adjs, images):
        out_image = []

        for idx, mesh in enumerate(meshes): #iterate over each mesh
            image = images[idx]
            mask, order = mesh.unroll()
            v_f = torch.zeros(mask.shape[0],image.shape[1]).to(image.device)
            v_f[mask] = image
            #reconstruct the image in reverse order
            for idx in range(len(order[0])):
                t = order[1, len(order[0]) - idx - 1]
                f = order[0, len(order[0]) - idx - 1]
                v_f[t] = v_f[f]
            out_image.append(v_f)
        
        return meshes, torch.stack(out_image)