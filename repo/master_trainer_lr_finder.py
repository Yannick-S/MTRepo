#import os
#os.chdir("/content")
#os.chdir("MTRepo/repo/")

from utility.checknotebook import in_ipynb
if in_ipynb():
    os.chdir("..")
    #! git pull
    os.chdir("repo")
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

load_from_file = False
start_epoch = 0
#### prepare model
import models.model_test_lr as mod
if in_ipynb():
    import importlib
    importlib.reload(mod)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = mod.Net().to(device)
model_info = model.get_info()
optimizer, scheduler = model.get_optimizer()
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

### pre train
import ignite_train
if in_ipynb(): importlib.reload(ignite_train)

print("Pre training")
model_info["training"]["max_epochs"] = 100
ignite_train.run(model, 
    optimizer,
    scheduler,
    loss,
    device,
    train_loader,
    training_history,
    param_history,
    model_info,
    start_epoch,
    path)

#### get LR
print("Get LR")
import numpy as np
    
all_histories = []
all_lrs = []

import utility.lr_getter

lr_g = utility.lr_getter.lr_exp(start=0.001, decay=0.5)

with tqdm(total=5) as pbar:
    for i in range(5):
        model_info["training"]["max_epochs"] = 300

        training_history['nll'] = []
        training_history['acc'] = []

        optimizer, scheduler = model.get_optimizer()
        for param_group in optimizer.param_groups:
            new_lr = lr_g.next() 
            all_lrs.append(new_lr)
            param_group['lr'] = all_lrs[-1]

        out = ignite_train.run_LR_find(model, 
                optimizer,
                loss,
                device,
                train_loader,
                training_history,
                param_history,
                model_info,
                start_epoch,
                path)

        out_tensor = np.array(out['nll'])
        all_histories.append(out_tensor)

        pbar.update(1)

import plot_results.plot_hist 
if in_ipynb(): importlib.reload(plot_results.plot_hist)

plot_results.plot_hist.plot_lr(all_histories, all_lrs)