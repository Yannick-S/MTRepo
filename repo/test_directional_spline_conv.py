import os
load_from_file = True 
if load_from_file:
    all_dirs = os.listdir('checkpoint')
    sorted_dirs = sorted(all_dirs)
    last_dir = sorted_dirs[-1]
    all_files = os.listdir('checkpoint/'+last_dir)
    all_files = [f for f in all_files if "epoch_" in f]
    all_files = [f for f in all_files if f[-4:] == '.pth']
    all_files = sorted(all_files)
    load_path = 'checkpoint/' + last_dir + '/' + all_files[-1]
    print(load_path)

experiment = {
    ### Data
    "nr_points": 100,
    ### Net
    "k": 20,
    "l": 9,
    "nr_filters": 15,
    "filter_size": 10,
    ### Learning
    "batch_size": 7,
    "learning_rate": 0.001
}

max_epochs = 30
save_every = 3

### don't touch this:
start_epoch = 0

###############################################################
#### Load Data ################################################
from torch_geometric.datasets.geometry import GeometricShapes
from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader


trans = Compose((
        SamplePoints(experiment['nr_points']),
        NormalizeScale(),
        RandomTranslate(0.01),
        RandomRotate(180)))

dataset = GeometricShapes('data/geometric', train=True, transform=trans)
experiment['nr_classes'] = len(dataset)

dataset = dataset.shuffle()

val_loader   = DataLoader(dataset, batch_size=experiment['batch_size'])
train_loader = DataLoader(dataset, batch_size=experiment['batch_size'])

################################################################
### Setup Model ################################################
import torch
from directional_spline_conv import SampleNetDC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SampleNetDC(
    k=experiment['k'],
    l=experiment['l'],
    filter_size=experiment['filter_size'],
    nr_filters=experiment['nr_filters'],
    nr_points=experiment['nr_points'],
    nr_classes=experiment['nr_classes']
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=experiment['learning_rate'])
loss = torch.nn.NLLLoss()

###############################################################
### setup LR ##################################################
if False:
    from utility.lr_finder import LRFinder

    lr_finder = LRFinder(model, optimizer, loss, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=10, step_mode="exp")
    lr_finder.plot(skip_start=0, skip_end=0)

###############################################################
### Setup Ignite ##############################################
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from utility.checknotebook import in_ipynb
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import datetime 
import os
from plot_results.setup_file import save_file, load_file

if load_from_file:
    checkpoint = torch.load(load_path)
    experiment_loaded = load_file(os.path.dirname(load_path) + '/experiment.json')
    assert experiment_loaded['nr_points'] == experiment['nr_points'], "Not the same amount of 'nr_points'. Loading has {} and now has {}".format(experiment_loaded['nr_points'], experiment['nr_points'])
    assert experiment_loaded['k'] == experiment['k'], "Not the same 'k'. Loading has {} and now has {}".format(experiment_loaded['k'], experiment['k'])
    assert experiment_loaded['l'] == experiment['l'], "Not the same 'l'. Loading has {} and now has {}".format(experiment_loaded['l'], experiment['l'])
    assert experiment_loaded['filter_size'] == experiment['filter_size'], "Not the same 'filter_size'. Loading has {} and now has {}".format(experiment_loaded['filter_size'], experiment['filter_size'])
    assert experiment_loaded['nr_filters'] == experiment['nr_filters'], "Not the same 'nr_filters'. Loading has {} and now has {}".format(experiment_loaded['nr_filters'], experiment['nr_filters'])
    assert experiment_loaded['nr_classes'] == experiment['nr_classes'], "Not the same 'nr_classes'. Loading has {} and now has {}".format(experiment_loaded['nr_classes'], experiment['nr_classes'])
    path = os.path.dirname(load_path) + '/'

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_history = checkpoint['training_history']
    start_epoch = checkpoint['epoch']

else:
    training_history = {'nll': [], 'acc': []}
    now = datetime.datetime.now()
    path = 'checkpoint/{:04d}-{:02d}-{:02d}'.format(now.year, now.month , now.day)
    path = path + '_{:02d}:{:02d}:{:02d}/'.format(now.hour, now.minute, now.second)
    if not os.path.isdir('checkpoint/'):
        os.mkdir('checkpoint')
    os.mkdir(path)
    save_file(os.path.dirname(path) + '/experiment.json', experiment)

def prep_batch(batch, device=device, non_blocking=False):
    return batch.to(device), batch.y.to(device)
trainer = create_supervised_trainer(model,
                                    optimizer,
                                    loss,
                                    device=device,
                                    prepare_batch=prep_batch)
evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': Accuracy(),
                                        'nll': Loss(loss)},
                                        device=device,
                                        prepare_batch=prep_batch)

from event_handlers.log_training import log_training_results
trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_training_results,
            evaluator, train_loader, training_history)

from event_handlers.save_model import handler_save_model
trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            handler_save_model, 
            save_every, model, optimizer, training_history, path, start_epoch)

from event_handlers.save_img import log_img 
trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_img, 
            training_history, path, start_epoch)


pbar = tqdm(total=max_epochs)
@trainer.on(Events.EPOCH_COMPLETED)
def show_bar(engine):
    pbar.update(1)
trainer.run(train_loader, max_epochs=max_epochs)
pbar.close()