import torch
import torch.nn.functional as F

from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import knn_graph

from torch_geometric.datasets.geometry import GeometricShapes
from directional_spline_conv import DirectionalSplineConv

from utility.utility import plot_point_cloud

#### Load Experiment ####
import os
from plot_results.setup_file import save_file, load_file

all_dirs = os.listdir('checkpoint')
sorted_dirs = sorted(all_dirs)
last_dir = sorted_dirs[-1]
all_files = os.listdir('checkpoint/'+last_dir)
all_files = [f for f in all_files if "epoch_" in f]
all_files = [f for f in all_files if f[-4:] == '.pth']
all_files = sorted(all_files)
load_path = 'checkpoint/' + last_dir + '/' + all_files[-1]
print(load_path)
checkpoint = torch.load(load_path, map_location='cpu')

experiment = load_file(os.path.dirname(load_path) + '/experiment.json')
experiment['batch_size'] = 1
########### Load DataSet ###############
trans = Compose((SamplePoints(experiment['nr_points']),
        NormalizeScale(),
        RandomTranslate(0.01),
        RandomRotate(180)))

#dataset = ModelNet(root='MN', name="10", train=True, transform=trans)
dataset = GeometricShapes('data/geometric', train=True, transform=trans)
nr_classes = len(dataset)
#experiment['nr_classes'] = len(dataset)

dataset = dataset.shuffle()
test_loader  = DataLoader(dataset,  batch_size =experiment['batch_size'])
train_loader = DataLoader(dataset, batch_size=experiment['batch_size'])


### Setup Experiment
import torch
from directional_spline_conv import SampleNetDC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SampleNetDC(
    k=experiment['k'],
    l=experiment['l'],
    filter_size=experiment['filter_size'],
    nr_filters=experiment['nr_filters'],
    nr_points=experiment['nr_points'],
    nr_classes=experiment['nr_classes'],
    out_y=True
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=experiment['learning_rate'])
loss = torch.nn.NLLLoss()

### load model


experiment_loaded = load_file(os.path.dirname(load_path) + '/experiment.json')
assert experiment_loaded['nr_points'] == experiment['nr_points'], "Not the same amount of 'nr_points'. Loading has {} and now has {}".format(experiment_loaded['nr_points'], experiment['nr_points'])
assert experiment_loaded['k'] == experiment['k'], "Not the same 'k'. Loading has {} and now has {}".format(experiment_loaded['k'], experiment['k'])
assert experiment_loaded['l'] == experiment['l'], "Not the same 'l'. Loading has {} and now has {}".format(experiment_loaded['l'], experiment['l'])
assert experiment_loaded['filter_size'] == experiment['filter_size'], "Not the same 'filter_size'. Loading has {} and now has {}".format(experiment_loaded['filter_size'], experiment['filter_size'])
assert experiment_loaded['nr_filters'] == experiment['nr_filters'], "Not the same 'nr_filters'. Loading has {} and now has {}".format(experiment_loaded['nr_filters'], experiment['nr_filters'])
#assert experiment_loaded['nr_classes'] == experiment['nr_classes'], "Not the same 'nr_classes'. Loading has {} and now has {}".format(experiment_loaded['nr_classes'], experiment['nr_classes'])
path = os.path.dirname(load_path) + '/'

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
training_history = checkpoint['training_history']
start_epoch = checkpoint['epoch']

#### Define Train and Eval Funcs ####
def show(epoch):
    model.eval()

    for data in train_loader:
        data = data.to(device)

        output, y = model(data)

        plot_point_cloud(data.pos.detach().numpy(), color=y[:,3:6].detach().numpy(), path='img_conv/'+str(data.y.item()) + '.png')

    quit()

#### execute Training ####
for epoch in range(1, 20001):
    print("Epoch:", epoch)
    show(epoch)    

