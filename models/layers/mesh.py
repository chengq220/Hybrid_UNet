import torch
import numpy as np
# from mesh_pool import MeshPool
# from mesh_unpool import MeshUnpool
# import time


class Mesh:
    def __init__(self, file=None):
        self.image = file.view(file.shape[0]*file.shape[1],file.shape[2])
        self.vertex_count = None
        self.vs, self.faces = self.__fill_mesh2(file.shape[0],file.shape[1])
        #return edges in the coo format
        self.edges, self.edge_counts = self.__get_edges(self.faces)
        self.adj_matrix = self.__adjacency(self.edges)
        self.vertex_mask = torch.ones(self.vertex_count, dtype=torch.bool)
        self.vertex_collapse = []
        self.history = []

    def __fill_mesh2(self,length, width):
        size = length * width
        self.vertex_count = size
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
        return np.asarray(vertex), np.asarray(faces)

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
        edges = np.asarray(edges,dtype=np.int64)
        u_e = np.vstack([edges, edges[:, ::-1]]) #undirected edges
        edges = torch.stack((torch.from_numpy(u_e[:, 0]), torch.from_numpy(u_e[:, 1])))
        return edges, edges.shape[1]

    #Generate adjacency matrix give edges in coo format
    def __adjacency(self,edges):
        num_nodes = edges.max().item() + 1
        #num_nodes = self.vertex_count
        adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.size(1)), size=(num_nodes, num_nodes))
        adj_matrix = adj_matrix.to_dense()
        return adj_matrix

    #update the edges stored in the mesh
    def __update_edges(self):
        edges = self.adj_matrix.nonzero()
        edges = torch.stack([edges[:, 0],edges[:,1]],dim=0)
        return edges

    def get_attributes(self):
        attribute = self.vs[self.edges[0, :]] - self.vs[self.edges[1, :]]
        return attribute

    #does merge vertex update the matrix every time a vertex merge or until all the items to be merged
    #have been determined
    def merge_vertex(self,edge_id):
        v_0 = edge_id[0]
        v_1 = edge_id[1]

        max_tensor = torch.max(self.image[v_0], self.image[v_1])
        with torch.no_grad():
            self.image[v_0] = max_tensor

        neighbors = torch.nonzero(self.adj_matrix[v_1]).squeeze(1)
        for neighbor in neighbors:
            if(neighbor != v_0): #update the neighbors and its adjacent vertices
                self.adj_matrix[neighbor,v_1] = False
                self.adj_matrix[neighbor,v_0] = True
                self.adj_matrix[v_0,neighbor] = True
        self.vertex_mask[v_1] = False
        self.vertex_count = self.vertex_count - 1
        self.vertex_collapse.append(edge_id)

    #clean up the adjacency matrix (vertex/edges) pooled
    def clean_up(self):
        self.adj_matrix = self.adj_matrix[self.vertex_mask][:, self.vertex_mask]
        self.image = self.image[self.vertex_mask]
        self.edges = self.__update_edges()
        self.edge_counts = self.edges.shape[1]
        self.history.append(np.vstack(self.vertex_collapse))
        self.vertex_mask = torch.ones(self.vertex_count, dtype=torch.bool)
        self.vertex_collapse = []

    #getter method: to get vertex collapse
    def vertex_collapse_order(self):
        order = torch.cat((torch.asarray(self.vertex_collapse[:,0]),torch.asarray(self.vertex_collapse[:,1])))
        return order