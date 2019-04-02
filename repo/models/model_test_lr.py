import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, fps

import torch.nn.functional as F
from .layers.directional_spline_conv import DirectionalSplineConv
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
        self.batch_size = 1
        self.nr_points = 30
        self.nr_classes = 10 if self.data_name == 'ModelNet10' else 40

        #train_info
        self.max_epochs = 5000
        self.save_every = 500

        #model
        self.k = 20
        self.l = 7
        
        self.filter_nr= 10
        self.kernel_size = 5

        self.dsc = DirectionalSplineConv(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=self.l,
                                            k=self.k)


        self.nn1 = torch.nn.Linear(self.filter_nr, 256)
        self.nn2 = torch.nn.Linear(256, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)



    def forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        edge_index = knn_graph(pos, self.k, batch, loop=False)

        _,_, y = self.dsc(pos, edge_index) 

        y = torch.sigmoid(y)
        ys = y.view(-1, self.nr_points , self.filter_nr)
        ys = ys.mean(dim=1).view(-1, self.filter_nr)
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

