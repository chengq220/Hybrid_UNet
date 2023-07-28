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
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
        #     with record_function("merge_vertices"):
        while mesh.vertex_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mesh.vertex_mask[mesh.edges[1,edge_id]] and mesh.vertex_mask[mesh.edges[0,edge_id]]:
                items = self.__pool_edge(mesh, edge_id)
                # if items is not None:
                #     queue = self.update_q(queue,items)
        mesh.clean_up()
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)) 
    def __pool_edge(self, mesh, edge_id):
        if self.is_boundary(mesh,edge_id):
            return None
        elif self.is_valid(mesh,edge_id):
            return mesh.merge_vertex(edge_id)
        else:
            return None

    @staticmethod
    def is_valid(mesh, edge_id):
        v_0, v_1 = mesh.edges[0, edge_id], mesh.edges[1, edge_id]
        v_0_n = mesh.adj_matrix[:, v_0] + mesh.adj_matrix[v_0]
        v_1_n = mesh.adj_matrix[:, v_1] + mesh.adj_matrix[v_1]
        shared = v_0_n & v_1_n
        return (shared.sum() == 2).item()

    @staticmethod
    def is_boundary(mesh, edge_id):
        v_0, v_1 = mesh.vs[mesh.edges[0, edge_id]], mesh.vs[mesh.edges[1, edge_id]]  # get the spacial position of the vertex
        epsilon = mesh.epsilon
        boundary_check = ((v_0[0] < epsilon) | (v_0[0] > 1 - epsilon) |
                          (v_0[1] < epsilon) | (v_0[1] > 1 - epsilon) |
                          (v_1[0] < epsilon) | (v_1[0] > 1 - epsilon) |
                          (v_1[1] < epsilon) | (v_1[1] > 1 - epsilon))
        return boundary_check

    @staticmethod
    def update_q(q, items):
        queue = q
        queue.extend(items)
        heapq.heapify(queue)
        return queue

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap
