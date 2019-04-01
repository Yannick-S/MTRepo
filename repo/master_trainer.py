from utility.checknotebook import in_ipynb
if in_ipynb():
    import os
    os.chdir("..")
    #! git pull
    os.chdir("repo")

load_from_file = False
start_epoch = 0
#### prepare model
import models.model_test_clipping as mod
if in_ipynb():
    import importlib
    importlib.reload(mod)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = mod.Net().to(device)
model_info = model.get_info()
optimizer = model.get_optimizer()
loss = torch.nn.NLLLoss()

#### load model
import model_loader
if in_ipynb(): importlib.reload(model_loader)
if load_from_file:
    model, optimizer, training_history, param_history, start_epoch, path = model_loader.load_model(model_info,model, optimizer)
else:
    training_history, param_history, path = model_loader.else_load(model_info, model)

#### load data
import data_loader
if in_ipynb(): importlib.reload(data_loader)
train_loader, val_loader = data_loader.data_from_data_info(model_info["data"])

#### train
import ignite_train
if in_ipynb(): importlib.reload(ignite_train)
ignite_train.run(model, 
    optimizer,
    loss,
    device,
    train_loader,
    training_history,
    param_history,
    model_info,
    start_epoch,
    path)
