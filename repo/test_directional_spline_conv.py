load_path = "/home/ys/Documents/MTRepo/repo/checkpoint/2019-03-07_11:55:28/epoch_5"
load_from_file = True

batch_size = 7
nr_points = 100
k = 20

#### Load Data ####
from torch_geometric.datasets.geometry import GeometricShapes
from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader


trans = Compose((SamplePoints(nr_points),
        NormalizeScale(),
        RandomTranslate(0.01),
        RandomRotate(180)))

dataset = GeometricShapes('data/geometric', train=True, transform=trans)
nr_classes = len(dataset)

dataset = dataset.shuffle()

val_loader   = DataLoader(dataset, batch_size=batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size)

### Setup Model
import torch
from directional_spline_conv import SampleNetDC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SampleNetDC(
    k=k,
    nr_points=nr_points,
    nr_classes=nr_classes
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
loss = torch.nn.NLLLoss()

if load_from_file:
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_history = checkpoint['training_history']
    print(training_history)
    #quit()

### setup LR
if False:
    from utility.lr_finder import LRFinder

    lr_finder = LRFinder(model, optimizer, loss, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=10, step_mode="exp")
    lr_finder.plot(skip_start=0, skip_end=0)

### Setup Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from utility.checknotebook import in_ipynb
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import datetime 
import os

max_epochs = 6
save_every = 3

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


if not load_from_file: 
    training_history = {'nll': [], 'acc': []}
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']

        training_history['nll'].append(avg_nll)
        training_history['acc'].append(avg_accuracy)
        print(training_history)

#########
now = datetime.datetime.now()
path = 'checkpoint/{:04d}-{:02d}-{:02d}'.format(now.year, now.month , now.day)
path = path + '_{:02d}:{:02d}:{:02d}/'.format(now.hour, now.minute, now.second)
if not os.path.isdir('checkpoint/'):
    os.mkdir('checkpoint')
os.mkdir(path)
@trainer.on(Events.EPOCH_COMPLETED)
def save_model(engine):
    if not engine.state.epoch  % save_every == 0:
        return
    torch.save({
        'epoch': engine.state.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history
        },
        path + 'epoch_{}'.format(engine.state.epoch)
    )

@trainer.on(Events.EPOCH_STARTED)
def save_info(engine):
    if engine.state.epoch == 0:
        print("Nr_Points:", nr_points)

############

pbar = tqdm(total=max_epochs)
@trainer.on(Events.EPOCH_COMPLETED)
def show_bar(engine):
    pbar.update(1)

trainer.run(train_loader, max_epochs=max_epochs)
pbar.close()


### Evaluate
from plot_results.plot_hist import plot_hist

plot_hist(training_history)
