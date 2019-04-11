import torch
from torch_geometric.nn import MessagePassing, SplineConv

from torch.nn import BatchNorm1d

from utility.diag import diag
from utility.utility import plot_point_cloud

class DirectionalSplineConv(MessagePassing):
    def __init__(self, filter_nr, l, k, kernel_size, out_3d = True):
        super(DirectionalSplineConv, self).__init__()
        self.k = k
        self.l = l if l <= k else k

        self.filter_nr = filter_nr
        self.kernel_size=kernel_size
        self.conv = SplineConv(1, self.filter_nr, dim=3, kernel_size=self.kernel_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.out_3d = out_3d 

    def forward(self, pos, edge_index, V_t = None):
        # center the clusters, make view
        clusters = pos[edge_index[1,:]] - pos[edge_index[0,:]]
        clusters = clusters.view(-1,self.k,3)
        nr_points = clusters.size(0)
        
        with torch.no_grad():
            # get covariance matrices:
            if V_t is None:
                clusters_t = torch.transpose(clusters,dim0=1,dim1=2)
                cov_mat = torch.tensor((nr_points,3,3),dtype=torch.float, requires_grad=False, device=self.device)
                torch.bmm( clusters_t[:,:,:self.l], clusters[:,:self.l,:], out=cov_mat)

            # get the projections
            if V_t is None:
                S, V = diag(cov_mat, nr_iterations=5, device=self.device)
                V_t = torch.transpose(V, 1,2) 
            else:
                V = torch.transpose(V_t, 1,2)

        # apply projections to clusters
        directional_clusters = torch.bmm(clusters, V_t) 
        signs = directional_clusters[:,:,2].sum(dim=1).sign()
        directional_clusters[:,:,2] = directional_clusters[:,:,2] * signs.view(-1,1)
        
        # move to [0,1] box
        max_abs = directional_clusters.abs(
                    ).view(-1,self.k*3
                    ).max(dim=1)[0]
        directional_clusters = directional_clusters/max_abs.view(-1,1,1)
        directional_clusters = directional_clusters*0.5 +0.5
        
        # prepare edges
        with torch.no_grad():
            ones  = torch.ones((self.k),device=self.device).view(1, self.k)
            linsp = torch.linspace(0,pos.size(0) - 1, steps=pos.size(0), device=self.device).view(pos.size(0),1)
            linsp = linsp * self.k

        cluster_edge      = torch.zeros((2, pos.size(0)*self.k), device=self.device, dtype=torch.long)
        cluster_edge[0,:] = torch.matmul(linsp, ones).view(-1)
        cluster_edge[1,:] = torch.linspace(0, pos.size(0)*self.k-1, steps=pos.size(0)*self.k, device=self.device)

        # prepare pseudo 
        ones  = torch.ones((self.k*pos.size(0)), device=self.device)

        # conv
        conv_out = self.conv(ones, cluster_edge, directional_clusters.view(-1,3))

        # extract important results
        linsp = torch.linspace(0, pos.size(0) - 1, steps=pos.size(0), device=self.device) * self.k
        linsp = linsp.long()
        out_nondir = conv_out[linsp,:]

        if self.out_3d == False:
            pos, edge_index, out_nondir

        # rotate results back
        out_nondir = out_nondir.view(-1,self.filter_nr,1).repeat(1,1,3)
        out_V = V[:,2].view(-1,1,3).repeat(1,self.filter_nr,1)
        out = torch.mul(out_nondir, out_V)

        return pos, edge_index, out, V_t


