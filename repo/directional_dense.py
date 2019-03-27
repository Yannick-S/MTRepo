import torch
from torch_geometric.nn import MessagePassing, SplineConv

from torch.nn import BatchNorm1d

from torch.nn import Sequential , Linear , ReLU

from utility.diag import diag
from utility.utility import plot_point_cloud

class DirectionalDense(MessagePassing):
    def __init__(self, l, k, kernel_size):
        super(DirectionalDense, self).__init__()
        self.k = k
        self.l = l if l <= k else k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        layers = []
        layers.append(Linear(18, 256))
        layers.append(ReLU())
        layers.append(Linear(256, 256))
        layers.append(ReLU())

        self.net = Sequential(*layers)

    def forward(self, pos, features, edge_index):
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
        directional_clusters = torch.bmm(clusters, V_t) 
        signs = directional_clusters[:,:,2].sum(dim=1).sign()
        directional_clusters[:,:,2] = directional_clusters[:,:,2] * signs.view(-1,1)

        # apply projection to features
        clusters_feature = features[edge_index[1,:]]
        clusters_feature = clusters_feature.view(nr_points, -1 ,3)
        directional_features = torch.bmm(clusters_feature, V_t)
        directional_features = directional_features.view(nr_points,self.k,-1)

        # concatenate the features and positions
        concat = torch.cat((directional_clusters, directional_features), dim=2).view(-1,18)

        # do the inner NN
        out = self.net(concat)

        # agregate
        out = out.view(-1, self.k, 256)
        out = out.sum(dim=1)

        # outer NN
        # blank now

        return out


#### Define Model ####
from torch_geometric.nn import knn_graph
import torch.nn.functional as F
from directional_spline_conv_3d import DirectionalSplineConv3D

class SampleNetDD(torch.nn.Module):
    def __init__(self, nr_points, k,l, nr_filters, filter_size,  nr_classes, out_y=False):
        super(SampleNetDD, self).__init__()
        self.out_y = out_y

        self.k = k
        self.l = l
        self.nr_points = nr_points
        self.nr_classes = nr_classes
        
        self.filter_nr= nr_filters
        self.kernel_size = filter_size

        self.dc3 = DirectionalSplineConv3D(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=self.l,
                                            k=self.k)

        self.dd = DirectionalDense(l=self.l,
                                   k=self.k,
                                   kernel_size=10)

        self.nn1 = torch.nn.Linear(256, 256)
        self.nn2 = torch.nn.Linear(256, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        edge_index = knn_graph(pos, self.k, batch, loop=False)

        features_3d = self.dc3(pos, edge_index)  #output: batch_size*nr_points, nr_filters,3 
        features_3d = torch.sigmoid(features_3d)
        features_dd = self.dd(pos, features_3d, edge_index)

        ys = features_dd.view(-1,self.nr_points,256)

        ys = ys.mean(dim=1).view(-1, 256)
        y1 = self.nn1(ys)
        y1 = F.elu(y1)
        y2 = self.nn2(y1)
        y2 = self.sm(y2) 
            
        if self.out_y:
            return y2, y
        return y2

