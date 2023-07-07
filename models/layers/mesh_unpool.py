import torch
import torch.nn as nn
import numpy as np

class MeshUnpool(nn.Module):
    def __init__(self):
        super(MeshUnpool, self).__init__()

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        for mesh in meshes: #iterate over each mesh
            img = mesh.get_feature()
            vertex, mask, order, edge = mesh.unroll()
            v_f = torch.zeros(mask.shape[0],img.shape[1]).to(img.device)
            v_f[mask] = img
            #reconstruct the image in reverse order
            for idx in range(len(order[0])):
                t = order[1,len(order[0])-idx-1]
                f = order[0,len(order[0])-idx-1]
                v_f[t] = v_f[f]
            mesh.update_feature(v_f)
            mesh.edge = edge
            mesh.vertex = vertex
        return meshes