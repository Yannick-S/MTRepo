import torch
from torch_geometric.nn import MessagePassing, SplineConv

from torch.nn import BatchNorm1d

from torch.nn import Sequential , Linear , ReLU

from utility.diag import diag
from utility.utility import plot_point_cloud

class DirectionalDense(MessagePassing):
    def __init__(self, l, k, in_size, mlp, out_size, with_pos=True):
        super(DirectionalDense, self).__init__()
        self.k = k
        self.l = l if l <= k else k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.in_size = in_size
        self.net = mlp
        self.out_size = out_size

        self.with_pos = with_pos

    def forward(self, pos, edge_index, features):
        # center the clusters, make view
        clusters = pos[edge_index[1,:]] - pos[edge_index[0,:]]
        clusters = clusters.view(-1,self.k,3)
        nr_points = clusters.size(0)
        
        # get covariance matrices:
        clusters_t = torch.transpose(clusters,dim0=1,dim1=2)
        cov_mat = torch.bmm( clusters_t[:,:,:self.l], clusters[:,:self.l,:])

        # get the projections
        S, V = diag(cov_mat, nr_iterations=5, device=self.device)
        V_t = torch.transpose(V, 1,2)

        # apply projections to clusters
        if self.with_pos:
           directional_clusters = torch.bmm(clusters, V_t) 
           signs = directional_clusters[:,:,2].sum(dim=1).sign()
           directional_clusters[:,:,2] = directional_clusters[:,:,2] * signs.view(-1,1)

        # apply projection to features
        clusters_feature = features[edge_index[1,:]]
        clusters_feature = clusters_feature.view(nr_points, -1 ,3)
        directional_features = torch.bmm(clusters_feature, V_t)
        directional_features = directional_features.view(nr_points,self.k,-1)

        # concatenate the features and positions
        if self.with_pos:
            concat = torch.cat((directional_clusters, directional_features), dim=2).view(-1,self.in_size)
        else:
            concat = directional_features.view(-1, self.in_size)

        # do the inner NN
        out = self.net(concat)

        # agregate
        out = out.view(-1, self.k, self.out_size)
        out = out.sum(dim=1)

        return pos, edge_index, out
