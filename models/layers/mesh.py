import torch
import numpy as np
# from models.layers.mesh_pool import MeshPool
from utils.util import pad


class Mesh:
    def __init__(self, file=None):
        self.before_pad_vertices = file.shape[1] * file.shape[2]
        file = pad(file)
        self.image = torch.transpose(file.view(file.shape[0],file.shape[1]*file.shape[2]),0,1)
        self.epsilon = self.calcEpsilon(file.shape[1],file.shape[2])
        self.vertex_count = None
        self.vs, self.faces = self.__fill_mesh(file.shape[1],file.shape[2])
        #return the directed edges in the coo format
        self.edges, self.edge_counts = self.__get_edges(self.faces)
        self.adj_matrix = self.__adjacency(self.edges)
        # self.neighbor = self.compute_neighbor()
        self.vertex_mask = torch.ones(self.vertex_count, dtype=torch.bool)
        self.collapse_order = []
        self.history_data = {
            'vertex': [],
            'vertex_mask': [],
            'collapse_order': [],
            'edge':[]
        }

    #create a mesh
    def __fill_mesh(self,length, width):
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
        d_e = np.asarray(edges,dtype=np.int64)
        edges = torch.stack((torch.from_numpy(d_e[:, 0]), torch.from_numpy(d_e[:, 1])))
        return edges, edges.shape[1]

    #Generate adjacency matrix give edges in coo format
    def __adjacency(self,edges):
        num_nodes = edges.max().item() + 1
        adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.size(1),dtype=torch.bool), size=(num_nodes, num_nodes))
        adj_matrix = adj_matrix.to_dense()
        return adj_matrix

    #update the edges stored in the mesh
    def __update_edges(self):
        edges = self.adj_matrix.nonzero()
        edges = torch.stack([edges[:, 0],edges[:,1]],dim=0)
        return edges

    #the edge stored is directed, this gives the undirected edges
    def get_undirected_edges(self):
        edge_flipped = torch.stack((self.edges[1, :], self.edges[0, :]))
        u_e = torch.cat((self.edges,edge_flipped),dim=1)
        return u_e

    #calculate the attributes in 2d space of each edges
    def get_attributes(self,u_e):
        feature = self.vs[u_e[1, :]] - self.vs[u_e[0, :]]
        return feature

    #collapse the two vertices together and update the matrix 
    def merge_vertex(self,edge_id):
        # start_time = time.time()
        v_0, v_1 = self.edges[0,edge_id], self.edges[1,edge_id]
        max_tensor = torch.dot(self.image[v_0], self.image[v_1])
        self.image[v_0].data = max_tensor
        
        neighbors = torch.cat((torch.nonzero(self.adj_matrix[v_1]).squeeze(1),torch.nonzero(self.adj_matrix[:,v_1]).squeeze(1)))
        valid_neighbors = neighbors[neighbors != v_0]

        # Update the adjacency matrix
        # print(valid_neighbors)

        self.adj_matrix[:, v_1] = False
        self.adj_matrix[v_1,:] = False
        self.adj_matrix[v_0, valid_neighbors] = True

        new_edges = v_0.repeat(valid_neighbors.size(0))
        new_edges = torch.stack((new_edges, valid_neighbors))
        features = torch.cat((self.image[new_edges[0]],self.image[new_edges[1]]),dim=1)
        squared_magnitude = torch.sum(features * features, 1)
        edge_ids = torch.arange(self.edge_counts, self.edge_counts+len(features), device=squared_magnitude.device, dtype=torch.float32)
        self.edges = torch.cat((self.edges,new_edges),dim=1)
        self.edge_counts += new_edges[0].size(0)
        heap_items = torch.stack((squared_magnitude, edge_ids),dim=1).tolist()

        self.vertex_mask[v_1] = False
        self.vertex_count = self.vertex_count - 1
        self.collapse_order.append(edge_id)

        return heap_items
        # return None

    #clean up the adjacency matrix (vertex/edges) pooled
    def clean_up(self):
        self.adj_matrix = self.adj_matrix[self.vertex_mask][:, self.vertex_mask]
        # self.neighbor = self.compute_neighbor()
        self.image = self.image[self.vertex_mask]
        self.update_history()
        self.vs = self.vs[self.vertex_mask]
        self.edges = self.__update_edges()
        self.edge_counts = self.edges.shape[1]
        self.vertex_mask = torch.ones(self.vertex_count, dtype=torch.bool)
        self.collapse_order = []

    #update the dictionary that contains all the information
    def update_history(self):
        #update vertex mask history
        v_mask_history = self.history_data.get('vertex_mask', [])
        v_mask_history.append(self.vertex_mask)

        #update vertex history
        vertex_history = self.history_data.get('vertex',[])
        vertex_history.append(self.vs)

        #update pool history
        pool_order = torch.stack((self.edges[0,self.collapse_order],self.edges[1,self.collapse_order]))
        pool_history = self.history_data.get('collapse_order', [])
        pool_history.append(pool_order)

        self.history_data['vertex_mask'] = v_mask_history
        self.history_data['collapse_order'] = pool_history
        self.history_data['vertex'] = vertex_history

    #update dictionary with informations like edge connectivity and vertex values
    def update_dictionary(self,list,category):
        history = self.history_data.get(category,[])
        history.append(list)
        self.history_data[category] = history

    #get the information needed for the unpool operation
    def unroll(self):
        vertex = self.history_data['vertex'].pop()
        vertex_mask = self.history_data['vertex_mask'].pop()
        pool_order = self.history_data['collapse_order'].pop()
        edges = self.history_data['edge'].pop()
        return vertex, vertex_mask, pool_order, edges

    @staticmethod
    def calcEpsilon(length,width):
        return float(1)/(length*width)