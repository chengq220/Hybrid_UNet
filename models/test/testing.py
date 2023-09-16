import torch

def checkVertexCoordinateBound(mesh):
    zero = torch.Tensor([0]).to(mesh.image.device)
    one = torch.Tensor([1]).to(mesh.image.device)
    for attr in mesh.vs:
        if torch.lt(attr[0],zero):
            print(attr)
            print("Vertex Bound Test failed")
            exit()
        if torch.gt(attr[0],one):
            print(attr)
            print("Vertex Bound Test failed")
            exit()
        if torch.lt(attr[1],zero):
            print(attr)
            print("Vertex Bound Test failed")
            exit()
        if torch.gt(attr[1],one):
            print(attr)
            print("Vertex Bound Test failed")
            exit()
    print("Passed: Vertex bounds are all between 0 and 1 inclusive")

def checkEdgeBound(mesh):
    maxBound = mesh.image.shape[0]-1
    for edges in mesh.edges[0,:]:
        if edges.to(torch.int) > maxBound:
            print(edges)
            print("Edge test failed")
            exit()
    for edges in mesh.edges[1,:]:
        if edges.to(torch.int) > maxBound:
            print(edges)
            print("Edge test failed")
            exit()
    print("Passed: All edges are in bound")
    
