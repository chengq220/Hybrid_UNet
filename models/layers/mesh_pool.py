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
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]
        self.v_collapse = []
        self.v_mask = []

    def __call__(self, image, fe, meshes):
        return self.forward(image, fe, meshes)

    #image is the image,fe is the image rep, meshes contains the mesh information
    def forward(self, image, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        self.out_image = []
        self.image = image
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
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return torch.stack(self.out_image), torch.stack(self.v_mask),self.v_collapse, out_features


    def __pool_main(self, mesh_index):
        self.idx_vertex = []
        mesh = self.__meshes[mesh_index]
        image = self.image[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=bool)
        # edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        self.v_collapse.append(np.array(self.idx_vertex))
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        image = image[mesh.v_mask]
        self.out_image.append(image)
        self.v_mask.append(torch.from_numpy(mesh.v_mask))
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__is_one_ring_valid(mesh, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            self.idx_vertex.append(mesh.merge_vertices(edge_id))
            # a = mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    @staticmethod
    def is_boundaries(mesh, edge_id):
        # Compute row sums and column sums of the adjacency matrix
        row_sums = mesh.adj_matrix.sum(dim=1)
        col_sums = mesh.adj_matrix.sum(dim=0)

        # Check if the edge's vertices have row sum or column sum less than the maximum
        vertex1, vertex2 = edge_id[0], edge_id[1]
        is_boundary = row_sums[vertex1] < row_sums.max() and col_sums[vertex2] < col_sums.max()
        return is_boundary.item()


    @staticmethod
    def is_valid(mesh, edge_id):
        #make sure that it's one ring is also valid
        # edges in coo
        v_0, v_1 = edge_id[0], edge_id[1]
        neighbor_v0 = set(mesh.edges[1, mesh.edges[0, :] == v_0].tolist())
        neighbor_v1 = set(mesh.edges[1, mesh.edges[0, :] == v_1].tolist())
        shared = neighbor_v0 & neighbor_v1 - set([v_0, v_1])
        return len(shared) == 2

    def __build_queue(self, features, edge_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap
