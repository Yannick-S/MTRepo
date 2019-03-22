import torch
from utility.tictoc import TicToc

def diag(A, nr_iterations, device):
    A_ = A.clone()
    n = A.size(0)

    for i in range(nr_iterations):
        #corners
        r = torch.sqrt(torch.pow(A[:,1,0],2) + torch.pow(A[:,2,0],2))
        c =  A[:,2,0] / r
        s = -A[:,1,0] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,0,0] = torch.ones((n))
        G[:,1,1] = c
        G[:,2,1] = -s 
        G[:,1,2] = s
        G[:,2,2] = c
        V = G 
        A = torch.bmm(G, A)
        
        #top
        r = torch.sqrt(torch.pow(A[:,0,0],2) + torch.pow(A[:,2,0],2))
        c = A[:,0,0] / r
        s = -A[:,2,0] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,1,1] = torch.ones((n))
        G[:,0,0] = c
        G[:,2,0] = s 
        G[:,0,2] = -s
        G[:,2,2] = c
        V = torch.bmm(G,V)
        A = torch.bmm(G, A)
    
        #top
        r = torch.sqrt(torch.pow(A[:,1,1],2) + torch.pow(A[:,2,1],2))
        c = A[:,1,1] / r
        s = -A[:,2,1] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,0,0] = torch.ones((n))
        G[:,1,1] = c
        G[:,2,1] = s 
        G[:,1,2] = -s
        G[:,2,2] = c
        V = torch.bmm(G,V)
        A = torch.bmm(G, A)

        A = torch.bmm(A, torch.transpose(V,1,2))

        if i == 0:
            W = V
        else:
            W = torch.bmm(V,W)

    #if torch.isnan(A).any():
    #    diag_loud(A_, nr_iterations, device)
    #    quit()
    if torch.isnan(W).any():
        diag_loud(A_, nr_iterations, device)
        quit()

    #TODO: This sorting is shit. How to do it good?
    diag = torch.cat(
        (A[:,0,0].view(-1,1),A[:,1,1].view(-1,1),A[:,2,2].view(-1,1)),
        dim=1)
    diag = torch.abs(diag)

    sort = torch.argsort(diag, dim=1, descending=True)
    sort = sort.view(-1,1,3)
    sort = sort.repeat(1,3,1)
    
    W = torch.gather(W, dim=2, index=sort)

    return A, W

def diag_loud(A, nr_iterations, device):
    print("\n\n\n HAMMER TIME \n\n\n")

    n = A.size(0)
    print("A", A)

    for i in range(nr_iterations):
        print(i)
        #corners
        r = torch.sqrt(torch.pow(A[:,1,0],2) + torch.pow(A[:,2,0],2))
        c =  A[:,2,0] / r
        s = -A[:,1,0] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,0,0] = torch.ones((n))
        G[:,1,1] = c
        G[:,2,1] = -s 
        G[:,1,2] = s
        G[:,2,2] = c
        V = G 
        A = torch.bmm(G, A)
        print("G", G)
        print("A", A)
        print("A size: ", torch.norm(A))
        
        #top
        r = torch.sqrt(torch.pow(A[:,0,0],2) + torch.pow(A[:,2,0],2))
        c = A[:,0,0] / r
        s = -A[:,2,0] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,1,1] = torch.ones((n))
        G[:,0,0] = c
        G[:,2,0] = s 
        G[:,0,2] = -s
        G[:,2,2] = c
        V = torch.bmm(G,V)
        A = torch.bmm(G, A)
    
        print("G", G)
        print("A", A)
        print("A size: ", torch.norm(A))

        #top
        r = torch.sqrt(torch.pow(A[:,1,1],2) + torch.pow(A[:,2,1],2))
        c = A[:,1,1] / r
        s = -A[:,2,1] / r
        G = torch.zeros((n,3,3), device=device)
        G[:,0,0] = torch.ones((n))
        G[:,1,1] = c
        G[:,2,1] = s 
        G[:,1,2] = -s
        G[:,2,2] = c
        V = torch.bmm(G,V)
        A = torch.bmm(G, A)

        print("G", G)
        print("A", A)
        print("A size: ", torch.norm(A))
        A = torch.bmm(A, torch.transpose(V,1,2))

        print("V", V)
        print("A", A)
        print("A size: ", torch.norm(A))
        if i == 0:
            W = V
        else:
            W = torch.bmm(V,W)

    #TODO: This sorting is shit. How to do it good?
    diag = torch.cat(
        (A[:,0,0].view(-1,1),A[:,1,1].view(-1,1),A[:,2,2].view(-1,1)),
        dim=1)
    diag = torch.abs(diag)
    sort = torch.argsort(diag, dim=1, descending=True)


    for i in range(W.size(0)):
        W[i] = W[i,sort[i]]

    return A, W

if __name__ == "__main__":
    X = torch.rand((1000,10,3))
    
    A = torch.bmm(torch.transpose(X,1,2) , X)

    S, V = diag(A, nr_iterations=7,device='cpu')