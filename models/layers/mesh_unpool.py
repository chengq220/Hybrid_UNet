import torch
import torch.nn as nn
import numpy as np

class MeshUnpool(nn.Module):
    def __init__(self):
        super(MeshUnpool, self).__init__()

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        edges = []
        edge_features = []
        for mesh in meshes: #iterate over each mesh
            img = mesh.image
            mask, order, edge, edge_feature = mesh.unroll()
            edges.append(edge)
            edge_features.append(edge_feature)
            v_f = torch.zeros(mask.shape[0],img.shape[1]).to(img.device)
            v_f[mask] = img
            #reconstruct the image in reverse order
            for idx in range(len(order[0])):
                t = order[1,len(order[0])-idx-1]
                f = order[0,len(order[0])-idx-1]
                v_f[t] = v_f[f]
            mesh.image = v_f
        return edges, edge_features