import torch
import numpy as np
from torch_geometric.nn import EdgeConv, knn_graph, MessagePassing, SplineConv
from torch_geometric.utils import scatter_
from utility.utility import get_orthog, plot_point_cloud, get_plane

from torch.nn import Sequential as Seq, Linear, ReLU
from torch.nn import BatchNorm1d



class DirectionalSplineConvNoF(MessagePassing):
    def __init__(self, filter_nr, l, k, activation, nr_points):
        super(DirectionalSplineConvNoF, self).__init__()
        self.nr_points = nr_points
        self.k = k
        self.l = l if l <= k else k
        self.activation = activation
        self.filter_nr = filter_nr
        self.conv = SplineConv(1, self.filter_nr, dim=3, kernel_size=15)
        self.bn = BatchNorm1d(self.filter_nr)

    def forward(self, x, edge_index):
        # center the clusters
        clusters = x[edge_index[1,:]] - x[edge_index[0,:]]

        #prepare output
        out_dir = torch.zeros(x.size(0), self.filter_nr, 3)
        out_nondir = torch.zeros(x.size(0),self.filter_nr)

        ### for every center we do:
        ones  = torch.ones(self.k)
        for i in range(x.size(0)):
            cluster = clusters[i*self.k:(i+1)*self.k]

            _, S, V = torch.svd(cluster[:self.l,:])

            # rotate to face correctly
            plot_point_cloud(cluster)
            directional_cluster = torch.matmul(cluster, V)
            plot_point_cloud(directional_cluster)
            if directional_cluster[:,2].sum() < 0:
                directional_cluster = -directional_cluster

            # move into [0,1]^3
            directional_cluster = directional_cluster/directional_cluster.abs().max() * 0.5 + 0.5

            # prepare fake edges. All poins are connected to the center point. 
            cluster_edge      = torch.zeros(2, self.k, dtype=torch.long)
            cluster_edge[0,:] = torch.zeros(self.k)
            cluster_edge[1,:] = torch.linspace(0, self.k-1, steps=self.k)

            # Spline Convolution
            #in [k, 3] out [k, 16]
            conv_out = self.conv(ones, cluster_edge, directional_cluster)[0,:]

            #TODO: remove this later. This is just for visualizing
            out_nondir[i] =  conv_out.view(-1)

            # Rotate back to original direction
            #v = V[2,:].view(1,3)
            #int_out = torch.matmul(conv_out.view(-1,1),v)
            #out_dir[i] = int_out
        

        out_bn = self.bn(out_nondir)

        #return out_dir, out_nondir
        return out_bn

class DirectionalEdgeConv(MessagePassing):
    def __init__(self, in_features, out_features, k):
        super(DirectionalEdgeConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.mlp = Seq(Linear(3+2*in_features, out_features*2),
                       ReLU(),
                       Linear(out_features*2, out_features))

        self.econv = EdgeConv(self.mlp, aggr='max')

    def forward(self, pos, edge_index, x):
        print("pos:", pos.size())
        print("edge_index:", edge_index.size())
        print("x:", x.size())

        clusters_pos = pos[edge_index[1,:]] - pos[edge_index[0,:]]

        #prepare output
        out_dir = torch.zeros(x.size(0), self.out_features, 3)

        ### for every center we do:
        for i in range(x.size(0)):
            cluster_pos = clusters_pos[i*self.k:(i+1)*self.k]
            cluster_x   = x[edge_index[1,i*self.k:(i+1)*self.k]]
            center_x    = x[i] 

            _, _, V = torch.svd(cluster_pos)
            # TODO: make sure (-1) is correct, later by centor of mass i guess

            # rotate to face correctly
            directional_cluster_pos = torch.matmul(cluster_pos, V)
            directional_cluster_x   = torch.matmul(cluster_x.view(-1,3), V).view(self.k,-1,3)
            directional_center_x    = torch.matmul(center_x, V).view(-1,3)

            for j in range(self.k):
                for n in range(directional_center_x.size(0)):
                    input = torch.cat((directional_cluster_pos[j,:],
                                       directional_center_x[n,:],
                                       directional_cluster_x[j,n,:]))

                    print("\n\n\n hi")
                    print(input.size())

                    x = self.mlp(input)

                    print(x)

                    quit()

