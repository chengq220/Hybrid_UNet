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
        super(MeshDownConv, self).__init__()
        self.conv1 = SplineConv(in_channels,out_channels,dim=2, kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels, dim=2 ,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.pool = None
        if(pool):
            self.pool = MeshPool()

    def __call__(self, meshes, adjs, images):
        return self.forward(meshes, adjs, images)

    def forward(self, meshes, adjs, images):
        before_pool = []
        #Spline Convolution
        for idx,mesh in enumerate(meshes):     
            v_f = images[idx]
            adj = adjs[idx]
            edges = mesh.get_undirected_edges(adj)
            edge_attribute = mesh.get_attributes(edges).cuda()
            v_f = self.conv1(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            before_pool.append(v_f)
        
        before_pool = torch.stack(before_pool)
        if self.pool is not None:
            meshes, adjs, images = self.pool(meshes, adjs, before_pool)
        else:
            meshes = meshes
            adjs = adjs
            images = before_pool
            before_pool = None
        return before_pool, meshes, adjs, images

class MeshUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MeshUpConv, self).__init__()
        self.conv1 = SplineConv(in_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.unpool = MeshUnpool()

    def __call__(self, meshes, adjs, images, skips):
        return self.forward(meshes, adjs, images, skips)

    def forward(self, meshes, adjs, images, skips):
        out_image = []
        meshes = self.unpool(meshes, adjs, images)
        for idx,mesh in enumerate(meshes): 
            v_f = images[idx]
            adj = adjs[idx]
            edge = mesh.get_undirected_edges(adj)
            edge_attribute = mesh.get_attributes(edge).cuda()
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = torch.cat((v_f,skips[idx]),1)
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            out_image.append(v_f)
        return meshes, torch.stack(out_image)