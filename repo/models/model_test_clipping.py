import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, fps

import torch.nn.functional as F
from .layers.directional_spline_conv_3d import DirectionalSplineConv3D
from .layers.directional_dense_1 import DirectionalDense

from utility.cyclic_lr import CyclicLR

from torch.nn import Sequential , Linear , ReLU
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #name
        self.name = "test_clip"
        #optimizer
        self.lr = 0.001
        self.optimizer_name = 'Adam'

        #data
        self.data_name = "Geometry"
        self.batch_size = 20
        self.nr_points = 1024
        self.nr_classes = 10 if self.data_name == 'ModelNet10' else 40

        #train_info
        self.max_epochs = 5000
        self.save_every = 500

        #model
        self.k = 20
        self.l = 7
        
        self.filter_nr= 8 
        self.kernel_size = 7
        self.dsc3d = DirectionalSplineConv3D(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=self.l,
                                            k=self.k)
        
        #
        self.in_size = self.filter_nr*3 * 2
        self.out_size = 64
        layers = []
        layers.append(Linear(self.in_size, 128))
        layers.append(ReLU())
        layers.append(Linear(128, self.out_size))
        layers.append(ReLU())
        dense3dnet = Sequential(*layers)
        self.dd = DirectionalDense(l = self.l,
                                   k = self.k,
                                   in_size = self.in_size,
                                   mlp = dense3dnet,
                                   out_size = self.out_size,
                                   with_pos=True)

        self.nn1 = torch.nn.Linear(self.out_size, 512)
        self.nn2 = torch.nn.Linear(512, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)



    def forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        # Build first edges
        edge_index = knn_graph(pos, self.k, batch, loop=False)

        #extract features in 3d
        _,_,features_3d = self.dsc3d(pos, edge_index) 
        features_3d = torch.sigmoid(features_3d)
        _,_,features_dd = self.dd(pos, edge_index, features_3d)
        features_dd = torch.sigmoid(features_dd)


        ys = features_dd.view(self.batch_size, -1 , self.out_size)
        ys = ys.mean(dim=1).view(-1, self.out_size)
        y1 = self.nn1(ys)
        y1 = F.elu(y1)
        y2 = self.nn2(y1)
        y2 = self.sm(y2) 
        
        return y2

    
    def get_info(self):
        model_info = {
            "name": self.name,
            "opt": {
                "name": self.optimizer_name,
                "lr": self.lr
            },
            "data":{
                "name": self.data_name,
                "nr_points": self.nr_points,
                "batch_size": self.batch_size
            },
            "training": {
                "max_epochs": self.max_epochs,
                "save_every": self.save_every
            }

        }
        return model_info

    def get_optimizer(self):
        if self.optimizer_name == 'Adam':
            opt = torch.optim.Adam(self.parameters(), 
                            lr=self.lr)
            sch = CyclicLR(opt, 
                           base_lr=1e-4,
                           max_lr=5e-4,
                           step_size=200,
                           mode='triangular'
                           )
            return opt, sch
                                         
        if self.optimizer_name == 'SGD':
            opt = torch.optim.SGD(self.parameters(),
                                          lr=self.lr)
            sch = CyclicLR(opt, 
                           base_lr=1e-4,
                           max_lr=5e-4,
                           step_size=20,
                           mode='triangular'
                           )
            return opt, sch
        else:
            raise NotImplementedError

