from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv
import torch.nn.functional as F


def recConvBlock(in_channel,out_channel,kernel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel,padding="same"),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=kernel,padding="same"),
        nn.ReLU()
    )
    return conv

class MeshDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pool):
        super(DownConv, self).__init__()
        self.conv1 = SplineConv(in_channels,out_channels,dim=2, kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels, dim=2 ,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.pool = None
        if(pool):
            self.pool = MeshPool()

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        before_pool = []
        #Spline Convolution
        for idx,mesh in enumerate(meshes):     
            v_f = mesh.image
            edges = mesh.get_undirected_edges()
            edge_attribute = mesh.get_attributes(edges).cuda()
            v_f = self.conv1(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            if self.pool is not None:
                before_pool.append(v_f)
                mesh.update_dictionary(edges,"edge")
            mesh.image = v_f

        if self.pool is not None:
            meshes = self.pool(meshes)
            before_pool = torch.stack(before_pool)
            return before_pool

class MeshUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv1 = SplineConv(in_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.unpool = MeshUnpool()

    def __call__(self, meshes, skips):
        return self.forward(meshes,skips)

    def forward(self, meshes,skips):
        meshes = self.unpool(meshes)
        for idx,mesh in enumerate(meshes): 
            v_f = mesh.image
            edge = mesh.get_undirected_edges()
            edge_attribute = mesh.get_attributes(edge).cuda()
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = torch.cat((v_f,skips[idx]),1)
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            mesh.image = v_f
        return meshes