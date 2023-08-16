import torch
import numpy as np
from torch import nn


class Mesh:
    def __init__(self, shape):
        self.shape = shape
        self.epsilon = self.calcEpsilon(shape[0], shape[1])
        self.vs = None
        self.vertex_count = None
        self.history_data = {
            'vertex': [],
            'vertex_mask': [],
            'collapse_order': [],
        }

    def get_adjacency(self):
        self.vs, faces = self.__fill_mesh(self.shape[0],self.shape[1])
        self.vertex_count = (self.shape[0]-2) * (self.shape[1]-2)
        edges, edge_counts = self.__get_edges(faces)
        num_nodes = edges.max().item() + 1
        adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.size(1), dtype=torch.bool),
                                             size=(num_nodes, num_nodes))
        return adj_matrix.to_dense()

    def __fill_mesh(self,length, width):
        size = length * width
        # The vertices position of the regular triangular mesh
        # h is constant since image is 2D

        row = (torch.arange(width) / (width - 1))
        row = row.view((-1, 1)).repeat(1, length)
        row = row.view(-1)

        col = torch.arange(length) / (length - 1)
        col = col.repeat(width)

        vertex = torch.stack((row, col), -1)

        # The faces of the regular triangular mesh
        # The vertex index of the first element in the triplet
        c_pre = torch.arange(size - width)
        c1 = c_pre.view((-1, 1)).repeat(1, 2)
        c1 = c1.view(-1)
        c1_mask = ~((width - 1) == c1 % width)
        c1 = c1[c1_mask]

        # the vertex index of the second element in the triplet
        c2 = torch.arange(size - width)
        c2_mask = ~(c_pre % width == 0)
        c2 = c2[c2_mask]
        c2_diagonal = c2 + width - 1
        c2, c2_diagonal = c2.view((-1, 1)), c2_diagonal.view((-1, 1))
        c2 = torch.cat((c2, c2_diagonal), 1).view(-1)

        # the vertex index of the third element in the triplet
        c3 = torch.arange(width, size)
        c3 = c3.view((-1, 1)).repeat(1, 2)
        c3 = c3.view(-1)
        c3_mask = ~(0 == c3 % width)
        c3 = c3[c3_mask]

        faces = torch.stack((c1, c2, c3), -1)
        return vertex, np.asarray(faces)

    #Generate the initial edges for mesh
    def __get_edges(self, faces):
        edge2key = dict()
        edges = []
        edges_count = 0
        for face_id, face in enumerate(faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edges_count += 1
        d_e = torch.tensor(edges,dtype=torch.int64)
        edge = torch.stack((d_e[:, 0], d_e[:, 1]))
        edge = torch.cat((edge, torch.flip(edge, dims=[0])),dim=1)
        return edge, edge.shape[1]


    def get_attributes(self,u_e):
        v_1 = self.vs[u_e[0, :]]
        v_2 = self.vs[u_e[1, :]]
        return v_1-v_2

    @staticmethod
    def get_undirected_edges(adj):
        edges = adj.nonzero()
        edges = torch.stack([edges[:, 0], edges[:, 1]], dim=0)
        return edges

    @staticmethod
    def get_directed_edges(adj):
        directed = torch.triu(adj)
        edges = directed.nonzero()
        edges = torch.stack([edges[:, 0], edges[:, 1]], dim=0)
        return edges

    #Variation 2: Vectorize Merging vertices
    def merge_vertex(self, adj, edges, order, image):
        v_0, v_1 = edges[0, order], edges[1, order]
        update_matrix = torch.zeros_like(image)

        max_tensor = torch.maximum(image[v_0], image[v_1])
        update_matrix[v_0] = max_tensor - image[v_0]
        
        adj[v_0] = adj[v_0] | adj[v_1]
        adj[:,v_0] = adj[:,v_0] | adj[:,v_1]
        
        self.vs[v_0] = (self.vs[v_0] + self.vs[v_1]) / 2

        return adj, update_matrix

    #update the dictionary that contains all the information
    def update_history(self, vertex_mask, pool_order):
        #update vertex mask history
        v_mask_history = self.history_data.get('vertex_mask', [])
        v_mask_history.append(vertex_mask)

        #update vertex history
        vertex_history = self.history_data.get('vertex',[])
        vertex_history.append(self.vs)
        self.vs = self.vs[vertex_mask]

        #update pool history
        pool_history = self.history_data.get('collapse_order', [])
        pool_history.append(pool_order)

        self.history_data['vertex_mask'] = v_mask_history
        self.history_data['collapse_order'] = pool_history
        self.history_data['vertex'] = vertex_history


    #get the information needed for the unpool operation
    def unroll(self):
        vertex_mask = self.history_data['vertex_mask'].pop()
        pool_order = self.history_data['collapse_order'].pop()
        self.vs = self.history_data['vertex'].pop()
        return vertex_mask, pool_order

    @staticmethod
    def calcEpsilon(length,width):
        return float(1)/(length*width)