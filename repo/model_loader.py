import os
import torch

def load_model(model_info, model):
    name = model_info["name"]

    # set up optimizer
    if model_info[""]

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
    start_epoch = checkpoint['epoch']

    return model, optimizer, training_history, start_epoch, last_dir 