import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn as nn

from torch_geometric.datasets import  ModelNet
from torch_geometric.transforms import SamplePoints, Compose, RandomRotate
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import EdgeConv, knn_graph, SplineConv, graclus, fps, GraphConv
from torch_geometric.nn import global_mean_pool as gavgp, max_pool_x

from torch_geometric.datasets.geometry import GeometricShapes
from utility.utility import plot_point_cloud, graclus_out
from my_nn_viz import DirectionalSplineConvNoF, DirectionalEdgeConv

#### Load Data ####
batch_size = 5
nr_points = 1000
k = 50

trans = Compose((SamplePoints(nr_points),
        RandomRotate(180)))

#dataset = ModelNet(root='MN', name="10", train=True, transform=trans)
dataset = GeometricShapes('data/geometric', train=True, transform=trans)
nr_clases = len(dataset)

dataset = dataset.shuffle()

test_loader = DataLoader(dataset, batch_size=batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size)

#### Define Model ####

class Net(torch.nn.Module):
    def __init__(self):
        self.k = k
        super(Net, self).__init__()
        
        self.filter_nr= 15

        self.dsc = DirectionalSplineConvNoF(filter_nr=self.filter_nr,
                                            l=6,
                                            k=self.k,
                                            activation=F.tanh,
                                            nr_points=nr_points)
        self.sm = torch.nn.LogSoftmax(dim=1)
        #self.dec = DirectionalEdgeConv(16, 32,k=self.k)
        self.nn1 = Linear(self.filter_nr, 256)
        self.nn2 = Linear(256, nr_clases)
        self.counter = 0

    def forward(self, data):
        print("counter:" , self.counter)
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        edge_index = knn_graph(pos, self.k, batch, loop=False)

        y = self.dsc(pos, edge_index) 

        y = torch.sigmoid(y)

        #y = gavgp(y , batch)
        if (self.counter+1) % 600 == 0 or self.counter > 600:
            y3 = y.view(-1, nr_points,self.filter_nr)
            print(y3.std(dim=1).view(-1, self.filter_nr))
            print(y3.std(dim=1).mean())
            print(y3.max(dim=1)[0].view(-1, self.filter_nr))
            color = y[:nr_points,:3].detach().numpy()
            #color = color - color.min()
            #color = color / color.max()
            plot_point_cloud(pos[:nr_points,:].detach().numpy(),color=color)
        y = y.view(-1, nr_points,self.filter_nr)
        y = y.mean(dim=1).view(-1, self.filter_nr)
        y1 = self.nn1(y)
        y1 = F.elu(y1)
        y2 = self.nn2(y1)
        y2 = self.sm(y2) 
            
        self.counter += 1
        return y2


### Setup Experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_f = nn.NLLLoss()

#### Define Train and Eval Funcs ####
def train(epoch):
    model.train()

    loss_all = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        
        loss = loss_f(output, data.y)

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

        for i in range(output.size(0)): 
            if output[i,:].max(dim=0)[1] == data.y[i]:
                correct += 1

    print("correct: ", correct, "loss: ", loss_all)
    return loss_all / len(dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred =  out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

#### execute Training ####
for epoch in range(1, 20001):
    loss = train(epoch)
    #train_acc = test(train_loader)
    #test_acc = test(test_loader)
    #train_acc = 0
    #print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc:
    #{:.5f}'.format(epoch, loss, train_acc, test_acc))
    

