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

    def __call__(self, meshes):
        return self.forward(meshes)

    #image is the image,fe is the image rep, meshes contains the mesh information
    def forward(self, meshes):
        # self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__meshes = meshes
        # self.__out_image = []
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
        # return torch.stack(self.__out_image)


    def __pool_main(self, mesh_index):
        self.idx_vertex = []
        mesh = self.__meshes[mesh_index]
        image = mesh.image
        features = torch.transpose(torch.cat((image[mesh.edges[0,:]],image[mesh.edges[1,:]]),dim=1),0,1)
        queue = self.__build_queue(features, mesh.edge_counts)
        mask = np.ones(mesh.edge_counts, dtype=bool)
        while mesh.vertex_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id)
                mask[edge_id] = False
        #update all the information and apply all the masks

        mesh.clean_up()

    def __pool_edge(self, mesh, edge_id):
        if self.is_boundaries(mesh, edge_id):
            return False
        elif self.is_valid(mesh, edge_id):
            mesh.merge_vertex(edge_id)
            return True
        else:
            return False

    @staticmethod
    def is_boundaries(mesh, edge_id):
        # Compute row sums and column sums of the adjacency matrix
        row_sums = mesh.adj_matrix.sum(dim=1)
        col_sums = mesh.adj_matrix.sum(dim=0)

        # Check if the edge's vertices have row sum or column sum less than the maximum
        vertex1, vertex2 = mesh.edges[0,edge_id],mesh.edges[1,edge_id]
        is_boundary = row_sums[vertex1] < row_sums.max() and col_sums[vertex2] < col_sums.max()
        return is_boundary.item()


    @staticmethod
    def is_valid(mesh, edge_id):
        #edges in coo
        v_0,v_1 = mesh.edges[0,edge_id], mesh.edges[1,edge_id]
        neighbor_v0 = set(mesh.edges[1,mesh.edges[0, :] == v_0].tolist())
        neighbor_v1 = set(mesh.edges[1, mesh.edges[0, :] == v_1].tolist())
        shared = neighbor_v0 & neighbor_v1 - set([v_0,v_1])
        
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
