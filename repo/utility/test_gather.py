import torch

n=2
X = torch.rand((n,3,3))

indices = torch.tensor((1,0,2))
indices = indices.long().view(-1,1,3)
indices = indices.repeat(n,3,1)

out = torch.gather(X, dim=2, index=indices)
print(X)
print(indices)
print(out)

