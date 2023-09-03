import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify
import heapq


class MeshPool(nn.Module):
    def __init__(self, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__multi_thread = multi_thread
        self.__meshes = None
        self.__images = None
        self.__adjs = None

    def __call__(self, meshes, adjs, images):
        return self.forward(meshes, adjs, images)

    def forward(self, meshes, adjs, images):
        pool_threads = []
        self.out = []
        self.__meshes = meshes
        self.__images = images
        self.__adjs = adjs

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
        image = self.__images[index]
        adj = self.__adjs[index]

        out_target = mesh.vertex_count // 2
        edges = mesh.get_directed_edges(adj)

        features = torch.transpose(torch.cat((image[edges[0, :]], image[edges[1, :]]), dim=1), 0, 1)
        queue = self.__build_queue(features, len(edges[0]))

        vertex_mask = torch.ones(image.shape[0],dtype=torch.bool)
        pool_indices = []
        update_matrix = torch.zeros_like(image)

        while mesh.vertex_count > out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)

            v_0 = edges[0,edge_id]
            v_1 = edges[1,edge_id]

            if vertex_mask[edges[1,edge_id]] and vertex_mask[edges[0,edge_id]] and not self.is_boundary(mesh, edges, edge_id):

                pool_indices.append(edge_id)
                vertex_mask[v_1] = False
                adj, update_val = mesh.merge_vertex(adj, edges, edge_id, image)
                update_matrix[v_0] = update_val

        image = image + update_matrix
        adj = adj[vertex_mask][:,vertex_mask]
        pool_order = torch.stack((edges[0,pool_indices],edges[1,pool_indices]))
        mesh.update_history(vertex_mask, pool_order)
        return mesh, adj, image[vertex_mask]

    def __pool_edge(self, mesh, edge_id):
        if self.is_boundary(mesh,edge_id):
            return None
        else:
            return mesh.merge_vertex(edge_id)

    @staticmethod
    def is_boundary(mesh, edges, edge_id):
        v_0, v_1 = mesh.vs[edges[0, edge_id]], mesh.vs[edges[1, edge_id]]  # get the spacial position of the vertex
        epsilon = mesh.epsilon
        boundary_check = ((v_0[0] < epsilon) | (v_0[0] > 1 - epsilon) |
                          (v_0[1] < epsilon) | (v_0[1] > 1 - epsilon) |
                          (v_1[0] < epsilon) | (v_1[0] > 1 - epsilon) |
                          (v_1[1] < epsilon) | (v_1[1] > 1 - epsilon))
        return boundary_check

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap
