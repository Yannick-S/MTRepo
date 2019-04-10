import torch
from torch_geometric.nn import MessagePassing, SplineConv

from torch.nn import BatchNorm1d

from torch.nn import Sequential , Linear , ReLU

from utility.diag import diag
from utility.utility import plot_point_cloud
from utility.tictoc import TicToc
class DirectionalDense(MessagePassing):
    def __init__(self, l, k,  mlp,
                 conv_p = True,
                 conv_fc = True,
                 conv_fn = True,
                 out_3d = True):
        super(DirectionalDense, self).__init__()
        self.k = k
        self.l = l if l <= k else k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = mlp

        self.conv_p  = conv_p
        self.conv_fc = conv_fc
        self.conv_fn = conv_fn

        self.out_3d = out_3d 

        self.ttlist = []
        for i in range(9):
            self.ttlist.append(TicToc(str(i)))

        self.ttcounter = 0

    def forward(self, pos, edge_index, features):
        self.ttcounter += 1
        if self.ttcounter % 1000 == 0:
            for i in range(9):
                print(self.ttlist[i])

        # center the clusters, make view
        self.ttlist[0].tic()
        clusters = pos[edge_index[1,:]] - pos[edge_index[0,:]]
        clusters = clusters.view(-1,self.k,3)
        nr_points = clusters.size(0)
        self.ttlist[0].toc()
        
        # get covariance matrices:
        self.ttlist[1].tic()
        clusters_t = torch.transpose(clusters,dim0=1,dim1=2)
        cov_mat = torch.bmm( clusters_t[:,:,:self.l], clusters[:,:self.l,:])
        self.ttlist[1].toc()

        # get the projections
        self.ttlist[2].tic()
        print(self.device)
        print(cov_mat.size())
        S, V = diag(cov_mat, nr_iterations=5, device=self.device)
        V_t = torch.transpose(V, 1,2)
        self.ttlist[2].toc()

        # apply projections to clusters
        self.ttlist[3].tic()
        if self.conv_p:
           directional_clusters = torch.bmm(clusters, V_t) 
           signs = directional_clusters[:,:,2].sum(dim=1).sign()
           directional_clusters[:,:,2] = directional_clusters[:,:,2] * signs.view(-1,1)
        self.ttlist[3].toc()

        # apply projection to features

        self.ttlist[4].tic()
        if self.conv_fn:
            clusters_feature_neighbor = features[edge_index[1,:]]
            clusters_feature_neighbor = clusters_feature_neighbor.view(nr_points, -1 ,3)
            directional_features_neighbor = torch.bmm(clusters_feature_neighbor, V_t)
            directional_features_neighbor = directional_features_neighbor.view(nr_points,self.k,-1)
        if self.conv_fc:
            clusters_feature_central = features[edge_index[1,:]]
            clusters_feature_central = clusters_feature_central.view(nr_points, -1 ,3)
            directional_features_central = torch.bmm(clusters_feature_central, V_t)
            directional_features_central = directional_features_central.view(nr_points,self.k,-1)
        self.ttlist[4].toc()

        # concatenate the features and positions

        self.ttlist[5].tic()
        if self.conv_p:
            concat = directional_clusters 
            if self.conv_fn:
                concat = torch.cat((concat, directional_features_neighbor), dim=2)
            if self.conv_fc:
                concat = torch.cat((concat, directional_features_central), dim=2)
        elif self.conv_fn:
            concat = directional_features_neighbor 
            if self.conv_fc:
                concat = torch.cat((concat, directional_features_central), dim=2)
        else:
            concat = directional_features_central
        concat = concat.view(nr_points * self.k, -1) # -1 = in_size to conv
        self.ttlist[5].toc()

        # do the inner NN

        self.ttlist[6].tic()
        out = self.net(concat)
        out_size = out.size(1)
        self.ttlist[6].toc()

        # agregate
        self.ttlist[7].tic()
        out = out.view(nr_points, self.k,-1)
        out = out.sum(dim=1)
        self.ttlist[7].toc()

        if self.out_3d == False:
            return pos, edge_index, out

        # rotate results back
        self.ttlist[8].tic()
        out = out.view(-1,out_size,1).repeat(1,1,3)
        out_V = V[:,2].view(-1,1,3).repeat(1,out_size,1)
        out = torch.mul(out, out_V)
        self.ttlist[8].toc()

        return pos, edge_index, out