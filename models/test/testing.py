import torch

def checkVertexCoordinateBound(mesh,adjs,image):
    zero = torch.Tensor([0]).to(image.device)
    one = torch.Tensor([1]).to(image.device)
    for i,attr in enumerate(mesh[0].vs):
        if torch.lt(attr[0],zero):
            print(mesh[0].vs[i])
            print("Vertex Bound Test failed")
            exit()
        if torch.gt(attr[0],one):
            print(mesh[0].vs[i])
            print("Vertex Bound Test failed")
            exit()
        if torch.lt(attr[1],zero):
            print(mesh[0].vs[i])
            print("Vertex Bound Test failed")
            exit()
        if torch.gt(attr[1],one):
            print(mesh[0].vs[i])
            print("Vertex Bound Test failed")
            exit()
    print("Passed: Vertex bounds are all between 0 and 1 inclusive")

def checkEdgeBound(mesh, adjs, image):
    maxBound = image.shape[1]-1
    for edges in mesh[0].get_undirected_edges(adjs)[0,:]:
        if edges.to(torch.int) > maxBound:
            print(edges)
            print("Edge test failed")
            exit()
    for edges in mesh[0].get_undirected_edges(adjs)[1,:]:
        if edges.to(torch.int) > maxBound:
            print(edges)
            print("Edge test failed")
            exit()
    print("Passed: All edges are in bound")