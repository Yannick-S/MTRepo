import torch
import torch.nn.functional as F

from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import knn_graph

from torch_geometric.datasets.geometry import GeometricShapes
from directional_spline_conv import DirectionalSplineConv

from utility.utility import plot_point_cloud
#### Load Data ####
batch_size = 1
nr_points = 2500
k = 2

trans = Compose((SamplePoints(nr_points),
        NormalizeScale(),
        RandomTranslate(0),
        RandomRotate(180)))

#dataset = ModelNet(root='MN', name="10", train=True, transform=trans)
dataset = GeometricShapes('data/geometric', train=True,  transform=trans)
nr_classes = len(dataset)

dataset = dataset.shuffle()
test_loader = DataLoader(dataset, batch_size=batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size)



#### Define Train and Eval Funcs ####
from utility.utility import plot_point_cloud
import matplotlib.pyplot as plt

def train(epoch):

    for data in train_loader:
        plot_point_cloud(data.pos, color='angle', show=False)

        plt.savefig('img/'+str(data.y.item()) + '.png', format='png', dpi=1000)
        plt.clf()
        plt.cla()
    quit()
    return 0

#### execute Training ####
for epoch in range(1, 20001):
    print("Epoch:", epoch)
    loss = train(epoch)
    
