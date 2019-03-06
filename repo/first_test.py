import torch
import torch.nn.functional as F

from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import knn_graph

from torch_geometric.datasets.geometry import GeometricShapes
from directional_spline_conv import DirectionalSplineConv

from utility.utility import plot_point_cloud
#### Load Data ####
batch_size = 6
nr_points = 500
k = 20

trans = Compose((SamplePoints(nr_points),
        NormalizeScale(),
        RandomTranslate(0.01),
        RandomRotate(180)))

#dataset = ModelNet(root='MN', name="10", train=True, transform=trans)
dataset = GeometricShapes('data/geometric', train=True, transform=trans)
nr_classes = len(dataset)

dataset = dataset.shuffle()
test_loader = DataLoader(dataset, batch_size=batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size)

#### Define Model ####
class Net(torch.nn.Module):
    def __init__(self):
        self.k = k
        super(Net, self).__init__()
        
        self.filter_nr= 15
        self.kernel_size = 10

        self.dsc = DirectionalSplineConv(filter_nr=self.filter_nr,
                                            kernel_size=self.kernel_size,
                                            l=9,
                                            k=self.k)

        self.nn1 = torch.nn.Linear(self.filter_nr, 256)
        self.nn2 = torch.nn.Linear(256, nr_classes)

        self.sm = torch.nn.LogSoftmax(dim=1)

        self.counter = 0 

    def forward(self, data):
        self.counter += 1
        pos, edge_index, batch = data.pos, data.edge_index, data.batch

        edge_index = knn_graph(pos, self.k, batch, loop=False)

        y = self.dsc(pos, edge_index) 

        y = torch.sigmoid(y)
        if self.counter > 300:
            y3 = y.view(-1, nr_points,self.filter_nr)
            color = y[:nr_points,:3].detach().numpy()
            plot_point_cloud(pos[:nr_points,:].detach().numpy(),color=color)
        y = y.view(-1, nr_points,self.filter_nr)
        y = y.mean(dim=1).view(-1, self.filter_nr)
        y1 = self.nn1(y)
        y1 = F.elu(y1)
        y2 = self.nn2(y1)
        y2 = self.sm(y2) 
            
        return y2

### Setup Experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_f = torch.nn.NLLLoss()

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

#### execute Training ####
for epoch in range(1, 20001):
    print("Epoch:", epoch)
    loss = train(epoch)
    

