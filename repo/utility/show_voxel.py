import numpy as np
from torch_geometric.transforms import SamplePoints, Compose, RandomRotate
from pyntcloud import PyntCloud
from torch_geometric.datasets.geometry import GeometricShapes

nr_points = 512

trans = Compose((SamplePoints(nr_points),
        RandomRotate(180)))

#dataset = ModelNet(root='MN', name="10", train=True, transform=trans)
dataset = GeometricShapes('gm', train=True, transform=trans)
dataset = dataset.shuffle()

x = dataset[0]['pos']

from utility import plot_point_cloud, plot_voxel

plot_voxel(x, d=32)




















