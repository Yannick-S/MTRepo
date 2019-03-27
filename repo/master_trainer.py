load_from_file = False
start_epoch = 0
#### prepare model
from model_1layer import Net
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model_info = model.get_info()
optimizer = model.get_optimizer()
loss = torch.nn.NLLLoss()

#### load model
from model_loader import load_model, else_load
if load_from_file:
    model, optimizer, training_history, start_epoch, path = load_model(model_info,model)
else:
    training_history, path = else_load(model_info)

#### load data
from data_loader import data_from_data_info
train_loader, val_loader = data_from_data_info(model_info["data"])

#### train
from ignite_train import run

run(model, optimizer, loss, device, train_loader, training_history, model_info, start_epoch, path)
