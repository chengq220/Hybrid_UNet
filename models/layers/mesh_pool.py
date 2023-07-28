import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify
from torch.profiler import profile, record_function, ProfilerActivity
import heapq


class MeshPool(nn.Module):
    def __init__(self, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__multi_thread = multi_thread
        self.__meshes = None

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        pool_threads = []
        num_vertex_delete = meshes[0].before_pad_vertices - meshes[0].before_pad_vertices // 2
        self.__out_target = meshes[0].vertex_count - num_vertex_delete
        self.out = []
        self.__meshes = meshes
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        return self.__meshes


    def __pool_main(self, mesh_index):
        # iterate over batch
        mesh = self.__meshes[mesh_index]
        mesh.initiateUpdate()
        image = mesh.image
        features = torch.transpose(torch.cat((image[mesh.edges[0, :]], image[mesh.edges[1, :]]), dim=1), 0, 1)
        queue = self.__build_queue(features, mesh.edge_counts)
        self.is_boundary = self.get_boundary(mesh)
        while mesh.vertex_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mesh.vertex_mask[mesh.edges[1,edge_id]] and mesh.vertex_mask[mesh.edges[0,edge_id]]:
                items = self.__pool_edge(mesh, edge_id)
        mesh.clean_up()

    @staticmethod
    def get_boundary(mesh):
        v_0, v_1 = mesh.vs[mesh.edges[0,:]], mesh.vs[mesh.edges[1,:]]
        epsilon = mesh.epsilon
        is_boundary = ((v_0[:,0] < epsilon) | (v_0[:,0] > 1 - epsilon) |
                          (v_0[:,1] < epsilon) | (v_0[:,1] > 1 - epsilon) |
                          (v_1[:,0] < epsilon) | (v_1[:,0] > 1 - epsilon) |
                          (v_1[:,1] < epsilon) | (v_1[:,1] > 1 - epsilon))
        return is_boundary

    def __pool_edge(self, mesh, edge_id):
        if self.is_boundary[edge_id]:
            return None
        else:
            return mesh.merge_vertex(edge_id)

    def __build_queue(self, features, edges_count):
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap
