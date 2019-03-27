load_from_file = False
start_epoch = 0
#### prepare model
from test_model import Net
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model_info = model.get_info()
optimizer = model.get_optimizer()
loss = torch.nn.NLLLoss()

#### load model
if load_from_file:
    from model_loader import load_model
    model, optimizer, training_history, start_epoch, path = load_model(
        model_info,
        model
    )
else:
    training_history = {'nll': [], 'acc': []}
    import datetime, os
    now = datetime.datetime.now()
    path = 'checkpoint/' + model_info["name"]
    path = path +'/{:04d}-{:02d}-{:02d}'.format(now.year, now.month , now.day)
    path = path + '_{:02d}:{:02d}:{:02d}/'.format(now.hour, now.minute, now.second)
    if not os.path.isdir('checkpoint/'): os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/' + model_info["name"]): os.mkdir('checkpoint/' + model_info["name"])
    os.mkdir(path)

#### load data
from data_loader import data_from_data_info
train_loader, val_loader = data_from_data_info(model_info["data"])

#### train
from ignite_train import run



run(model, 
    optimizer,
    loss, 
    device, 
    train_loader, 
    training_history,
    model_info,
    start_epoch,
    path
    )
