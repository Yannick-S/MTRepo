import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, fps

import torch.nn.functional as F
from .layers.directional_spline_conv import DirectionalSplineConv
from .layers.directional_spline_conv_3d import DirectionalSplineConv3D
from .layers.directional_dense import DirectionalDense
from .layers.directional_dense_3d import DirectionalDense3D

from torch.nn import Sequential , Linear , ReLU
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #name
        self.name = "1layer"
        #optimizer
        self.lr = 0.001
        self.optimizer_name = 'Adam'

        #data
        self.data_name = "Geometry"
        self.batch_size = 20
        self.nr_points = 1000
        self.nr_classes = 10 if self.data_name == 'ModelNet10' else 40

        #train_info
        self.max_epochs = 100
        self.save_every = 10

        #model
        self.k = 15
        self.l = 7
        
        self.filter_nr= 10
        self.kernel_size = 5
        self.dsc3d = DirectionalSplineConv3D(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=self.l,
                                            k=self.k)
        
        #
        self.in_size = self.filter_nr*3 + 3
        self.out_size = 20
        layers = []
        layers.append(Linear(self.in_size, 20))
        layers.append(ReLU())
        layers.append(Linear(20, self.out_size))
        layers.append(ReLU())
        dense3dnet = Sequential(*layers)
        self.dd = DirectionalDense3D(l = self.l,
                                   k = self.k,
                                   in_size = self.in_size,
                                   mlp = dense3dnet,
                                   out_size = self.out_size,
                                   with_pos=True)


        #
        self.in_size_2 = self.out_size*3
        self.out_size_2 = 32
        layers2 = []
        layers2.append(Linear(self.in_size_2, 64))
        layers2.append(ReLU())
        layers2.append(Linear(64, self.out_size_2))
        layers2.append(ReLU())
        dense3dnet2 = Sequential(*layers2)
        self.dd2 = DirectionalDense(l = self.l,
                                   k = self.k,
                                   in_size = self.in_size_2,
                                   mlp = dense3dnet2,
                                   out_size = self.out_size_2,
                                   with_pos=False)


        self.nn1 = torch.nn.Linear(self.out_size_2, 256)
        self.nn2 = torch.nn.Linear(256, self.nr_classes)

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

        # pooling 80%
        index = fps(pos, batch=batch, ratio=0.2)
        pos = pos[index]
        features = features_dd[index]
        batch = batch[index]
        edge_index = knn_graph(pos, self.k, batch, loop=False) #change pos to features for test later!

        # extract features in 3d again
        _,_,features_dd2 = self.dd2(pos, edge_index, features_dd)
        features_dd2 = torch.sigmoid(features_dd2)

        ys = features_dd2.view(self.batch_size, -1 , self.out_size_2)
        ys = ys.mean(dim=1).view(-1, self.out_size_2)
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
            return torch.optim.Adam(self.parameters(),
                                          lr=self.lr)
        else:
            raise NotImplementedError

