import torch
from torch_geometric.nn import MessagePassing, SplineConv

from torch.nn import BatchNorm1d

from utility.diag import diag
from utility.utility import plot_point_cloud

class DirectionalSplineConv3D(MessagePassing):
    def __init__(self, filter_nr, l, k, kernel_size):
        super(DirectionalSplineConv3D, self).__init__()
        self.k = k
        self.l = l if l <= k else k

        self.filter_nr = filter_nr
        self.kernel_size=kernel_size
        self.conv = SplineConv(1, self.filter_nr, dim=3, kernel_size=self.kernel_size)

        self.bn = BatchNorm1d(self.filter_nr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index):
        # center the clusters, make view
        clusters = x[edge_index[1,:]] - x[edge_index[0,:]]
        clusters = clusters.view(-1,self.k,3)

        # get covariance matrices:
        clusters_t = torch.transpose(clusters,dim0=1,dim1=2)
        cov_mat = torch.bmm( clusters_t[:,:,:self.l], clusters[:,:self.l,:])

        # get the projections
        S, V = diag(cov_mat, nr_iterations=5, device=self.device)

        # apply projections to clusters
        directional_clusters = torch.bmm(clusters, torch.transpose(V, 1,2)) 
        signs = directional_clusters[:,:,2].sum(dim=1).sign()
        directional_clusters[:,:,2] = directional_clusters[:,:,2] * signs.view(-1,1)
        max_abs = directional_clusters.abs(
                    ).view(-1,self.k*3
                    ).max(dim=1)[0]
        directional_clusters = directional_clusters/max_abs.view(-1,1,1)
        directional_clusters = directional_clusters*0.5 +0.5

        #plot_point_cloud(clusters[0,:,:])
        #plot_point_cloud(directional_clusters[0,:,:])
        
        #prepare output
            #out_dir = torch.zeros(x.size(0), self.filter_nr, 3).to(self.device)
        out_nondir = torch.zeros((x.size(0),self.filter_nr),device=self.device)

        # prepare edges
        ones  = torch.ones((self.k),device=self.device).view(1, self.k)
        linsp = torch.linspace(0,x.size(0) - 1, steps=x.size(0), device=self.device).view(x.size(0),1)
        linsp = linsp * self.k

        cluster_edge      = torch.zeros((2, x.size(0)*self.k), device=self.device, dtype=torch.long)
        cluster_edge[0,:] = torch.matmul(linsp, ones).view(-1)
        cluster_edge[1,:] = torch.linspace(0, x.size(0)*self.k-1, steps=x.size(0)*self.k, device=self.device)

        # prepare pseudo 
        ones  = torch.ones((self.k*x.size(0)), device=self.device)

        # conv
        conv_out = self.conv(ones, cluster_edge, directional_clusters.view(-1,3))

        # extract important results
        # conv does a lot of useles shit for me... :(
        linsp = torch.linspace(0, x.size(0) - 1, steps=x.size(0), device=self.device) * self.k
        linsp = linsp.long()
        out_nondir = conv_out[linsp,:]

        # batch NR
        out_bn = self.bn(out_nondir)

        # rotate results back
        out_bn = out_bn.view(-1,self.filter_nr,1).repeat(1,1,3)
        out = V[:,2].view(-1,1,3).repeat(1,self.filter_nr,1)
        out = torch.mul(out_bn, out)

        return out 



#### Define Model ####
from torch_geometric.nn import knn_graph
import torch.nn.functional as F

class SampleNetDC3D(torch.nn.Module):
    def __init__(self, nr_points, k,l, nr_filters, filter_size,  nr_classes, out_y=False):
        super(SampleNetDC3D, self).__init__()
        self.out_y = out_y

        self.k = k
        self.l = l
        self.nr_points = nr_points
        self.nr_classes = nr_classes
        
        self.filter_nr= nr_filters
        self.kernel_size = filter_size

        self.dsc = DirectionalSplineConv3D(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=self.l,
                                            k=self.k)

        self.nn1 = torch.nn.Linear(self.filter_nr, 256)
        self.nn2 = torch.nn.Linear(256, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        edge_index = knn_graph(pos, self.k, batch, loop=False)

        y = self.dsc(pos, edge_index) 

        y = torch.sigmoid(y)
        ys = y.view(-1, self.nr_points , self.filter_nr)
        ys = ys.mean(dim=1).view(-1, self.filter_nr)
        y1 = self.nn1(ys)
        y1 = F.elu(y1)
        y2 = self.nn2(y1)
        y2 = self.sm(y2) 
            
        if self.out_y:
            return y2, y
        return y2
