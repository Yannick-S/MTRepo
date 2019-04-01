import os
import torch

def load_model(model_info, model, optimizer):
    name = model_info["name"]

    # find latest model
    all_dirs = os.listdir('checkpoint' + '/' + name)
    sorted_dirs = sorted(all_dirs)
    last_dir = sorted_dirs[-1]
    all_files = os.listdir('checkpoint/'+name  + '/' + last_dir)
    all_files = [f for f in all_files if "epoch_" in f]
    all_files = [f for f in all_files if f[-4:] == '.pth']
    all_files = sorted(all_files)
    load_path = 'checkpoint/' + name + '/' + last_dir + '/' + all_files[-1]

    # load latest model and optimzier
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_history = checkpoint['training_history']
    param_history = checkpoint['param_history']
    start_epoch = checkpoint['epoch']

    # 
    path = 'checkpoint' + '/' + name + '/' + last_dir + '/'
    return model, optimizer, training_history, param_history, start_epoch, path 

def else_load(model_info, model):
    training_history = {'nll': [], 'acc': []}
    import datetime, os
    now = datetime.datetime.now()
    path = 'checkpoint/' + model_info["name"]
    path = path +'/{:04d}-{:02d}-{:02d}'.format(now.year, now.month , now.day)
    path = path + '_{:02d}:{:02d}:{:02d}/'.format(now.hour, now.minute, now.second)
    if not os.path.isdir('checkpoint/'): os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/' + model_info["name"]): os.mkdir('checkpoint/' + model_info["name"])
    os.mkdir(path)

    param_history = {}
    for i, param in enumerate(model.parameters()):
        param_history[i] = []
    
    return training_history, param_history, path