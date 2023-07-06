import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify


class MeshPool(nn.Module):
    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__meshes = None

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        pool_threads = []
        self.out = []
        self.__meshes = meshes
        # iterate over batch
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
        mesh = self.__meshes[mesh_index]
        image = mesh.get_feature()
        features = torch.transpose(torch.cat((image[mesh.edges[0, :]], image[mesh.edges[1, :]]), dim=1), 0, 1)
        queue = self.__build_queue(features, mesh.edge_counts)
        while mesh.vertex_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            # make sure that the vertex have not been merged already
            if mesh.vertex_mask[mesh.edges[1,edge_id]] and mesh.vertex_mask[mesh.edges[0,edge_id]]:
                self.__pool_edge(mesh, edge_id)
        mesh.clean_up()

    def __pool_edge(self, mesh, edge_id):
        if self.is_valid(mesh,edge_id):
            mesh.merge_vertex(edge_id)
            return True
        else:
            return False

    @staticmethod
    def is_valid(mesh, edge_id):
        # edges in coo
        v_0, v_1 = mesh.edges[0, edge_id], mesh.edges[1, edge_id]
        n_0 = set(mesh.edges[1, mesh.edges[0, :] == v_0].tolist())
        n_f_0 = set(mesh.edges[0, mesh.edges[1, :] == v_0].tolist())  # find all the undirected neighbors
        neighbor_v0 = n_0.union(n_f_0)
        n_1 = set(mesh.edges[1, mesh.edges[0, :] == v_1].tolist())
        n_f_1 = set(mesh.edges[0, mesh.edges[1, :] == v_1].tolist())
        neighbor_v1 = n_1.union(n_f_1)
        shared = neighbor_v0 & neighbor_v1 - set([v_0, v_1])
        return len(shared) == 2

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap
