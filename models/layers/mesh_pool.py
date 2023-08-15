import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify
import heapq
import time


class MeshPool(nn.Module):
    def __init__(self, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__multi_thread = multi_thread
        self.__meshes = None
        self.images = None
        self.adjs = None

    def __call__(self, meshes, adjs, images):
        return self.forward(meshes, adjs, images)

    def forward(self, meshes, adjs, images):
        pool_threads = []
        self.adjs = adjs
        self.images = images
        self.__meshes = meshes

        #return values
        out_mesh = []
        out_adj = []
        out_img = []

        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                mesh, adj, img = self.__pool_main(mesh_index)
                out_mesh.append(mesh)
                out_adj.append(adj)
                out_img.append(img)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()

        return np.asarray(out_mesh), torch.stack(out_adj), torch.stack(out_img)


    def __pool_main(self, index):
        mesh = self.__meshes[index]
        image = self.images[index]
        adj = self.adjs[index]

        out_target = mesh.vertex_count // 2
        edges = mesh.get_directed_edges(adj)

        features = torch.transpose(torch.cat((image[edges[0, :]], image[edges[1, :]]), dim=1), 0, 1)
        queue = self.__build_queue(features, edges.shape[1])
        boundary_mask = ~self.is_boundary(mesh,edges)

        vertex_mask = torch.ones(image.shape[0], dtype=torch.bool)
        pool_order = []

        while mesh.vertex_count > out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            v_1 = edges[1,edge_id]
            v_0 = edges[0,edge_id]
            if boundary_mask[edge_id] and vertex_mask[v_1] and vertex_mask[v_0]:
                vertex_mask[v_1] = False
                pool_order.append(edge_id)
                mesh.vertex_count = mesh.vertex_count - 1
        pool_order = torch.tensor(pool_order)
        adj, update_matrix = mesh.merge_vertex(adj, edges, pool_order, image)
        adj = adj[vertex_mask][:,vertex_mask]
        adj = adj * ~torch.eye(adj.size(0), dtype=bool) #remove self-loop
        image = image + update_matrix
        mesh.update_history(vertex_mask, pool_order)
        return mesh, adj, image[vertex_mask]

    @staticmethod
    def is_boundary(mesh,edges):
        v_0, v_1 = mesh.vs[edges[0,:]], mesh.vs[edges[1,:]]
        epsilon = mesh.epsilon
        bound = ((v_0[:,0] < epsilon) | (v_0[:,0] > 1 - epsilon) |
                          (v_0[:,1] < epsilon) | (v_0[:,1] > 1 - epsilon) |
                          (v_1[:,0] < epsilon) | (v_1[:,0] > 1 - epsilon) |
                          (v_1[:,1] < epsilon) | (v_1[:,1] > 1 - epsilon))
        return bound

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap