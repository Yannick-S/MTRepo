import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, fps

import torch.nn.functional as F
from .layers.directional_dense import DirectionalDense as DD

from torch.nn import Sequential , Linear , ReLU
from utility.cyclic_lr import CyclicLR

from utility.tictoc import TicToc

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #name
        self.name = "testFast"
        #optimizer
        self.lr = 0.001
        self.optimizer_name = 'Adam-Exp'

        #data
        self.data_name = "ModelNet40"
        #self.data_name = "Geometry"
        self.batch_size = 10
        self.nr_points = 1024
        self.nr_classes = 10 if self.data_name == 'ModelNet10' else 40

        #train_info
        self.max_epochs = 60
        self.save_every = 60
        
        self.nn1 = torch.nn.Linear(3, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)


    def forward(self, data):

        pos, edge_index, batch = data.pos, data.edge_index, data.batch
        real_batch_size = pos.size(0) /self.nr_points
        real_batch_size = int(real_batch_size)

        y1 = self.nn1(pos)
        y1 = y1.view(real_batch_size, -1 , self.nr_classes)
        y1 = torch.max(y1, dim=1)[0]
        y1 = torch.nn.functional.relu(y1)
    
        out = self.sm(y1)

        return out
    
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
        if self.optimizer_name == 'Adam-Exp':
            opt = torch.optim.Adam(self.parameters(), 
                            lr=self.lr)
            sch = torch.optim.lr_scheduler.StepLR(opt,
                                                  step_size=6,
                                                  gamma=0.9)

            return opt, sch
        if self.optimizer_name == 'Adam-Tri':
            opt = torch.optim.Adam(self.parameters(), 
                            lr=self.lr)
            sch = CyclicLR(opt, 
                           base_lr=0.00005,
                           max_lr= 0.0002,
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