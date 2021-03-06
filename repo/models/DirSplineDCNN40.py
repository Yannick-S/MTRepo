import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, graclus, avg_pool_x, max_pool_x 

import torch.nn.functional as F
from .layers.directional_dense import DirectionalDense as DD
from .layers.directional_dense_minus import DirectionalDense as DDm
from .layers.directional_spline_conv import DirectionalSplineConv as DS

from torch.nn import Sequential , Linear , ReLU
from utility.cyclic_lr import CyclicLR

from utility.tictoc import TicToc

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #name
        self.name = "DirSplineDCNN40"
        #optimizer
        self.lr = 0.001*0.440126669
        self.optimizer_name = 'Adam-Exp'

        #data
        #self.data_name = "ModelNet40"
        self.data_name = "Geometry"
        self.batch_size = 40

        self.nr_points = 1024
        self.nr_classes = 10 if self.data_name == 'ModelNet10' else 40

        #train_info
        self.max_epochs = 100
        self.save_every = 2

        #model
        self.k = 20
        self.l = 7
        
        # DD1
        self.filter_nr = 64
        self.kernel_size = 7

        self.ds1 = DS(self.filter_nr,
                      self.l,
                      self.k,
                      self.kernel_size,
                      out_3d=True)

        self.ds1bn= torch.nn.BatchNorm1d(self.filter_nr)

        # DD2
        self.in_size_2 = self.filter_nr * 6
        self.out_size_2 = 128
        layers2 = []
        layers2.append(Linear(self.in_size_2, self.out_size_2))
        layers2.append(ReLU())
        layers2.append(torch.nn.BatchNorm1d(self.out_size_2))
        dense3dnet2 = Sequential(*layers2)
        self.dd2 = DDm(l = self.l,
                        k = self.k,
                        mlp = dense3dnet2,
                        conv_p  = False,
                        conv_fc = True,
                        conv_fn = True,
                        out_3d  = False)


        self.nn1 = torch.nn.Linear(self.out_size_2, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.nn2 = torch.nn.Linear(1024, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.nn3 = torch.nn.Linear(512, 265)
        self.bn3 = torch.nn.BatchNorm1d(265)
        self.nn4 = torch.nn.Linear(265, self.nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)


    def forward(self, data):

        pos, edge_index, batch = data.pos, data.edge_index, data.batch
        real_batch_size = pos.size(0) /self.nr_points
        real_batch_size = int(real_batch_size)

        # Build first edges
        edge_index = knn_graph(pos, self.k, batch, loop=False)

        #extract features in 3d
        _,_,features_dd, _ =  self.ds1(pos, edge_index, None)

        #graclus
        cluster = graclus(edge_index)

        pos_gra, batch_gra = avg_pool_x(cluster, pos, batch)
        features_gra, _ = max_pool_x(cluster, features_dd, batch)

        #knn(f)
        with torch.no_grad():
            edge_index_gra = knn_graph(features_gra.norm(dim=2), self.k, batch_gra, loop=False)


        # DD2
        _,_,features_dd2, _  = self.dd2(pos_gra, edge_index_gra, features_gra)

        y1 = self.nn1(features_dd2)

        y1_pool, _ = max_pool_x(batch_gra, y1, batch_gra)

        y1_pool = torch.nn.functional.relu(y1_pool)
        y1_pool = self.bn1(y1_pool)

        y2 = self.nn2(y1_pool)
        y2 = torch.nn.functional.relu(y2)
        y2 = self.bn2(y2)

        y3 = self.nn3(y2)
        y3 = torch.nn.functional.relu(y3)
        y3 = self.bn3(y3)

        y4 = self.nn4(y3)
        out = self.sm(y4)

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
                                                  step_size=5,
                                                  gamma=0.95)

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