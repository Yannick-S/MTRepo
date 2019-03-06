#### Load Data ####
from torch_geometric.datasets.geometry import GeometricShapes
from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.data import DataLoader

batch_size = 6
nr_points = 1000
k = 20

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
loss = torch.nn.NLLLoss()


### setup LR
from utility.lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, loss, device=device)
lr_finder.range_test(train_loader, end_lr=0.1, num_iter=100, step_mode="exp")
lr_finder.plot(skip_end=1)

### Setup Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from tqdm import tqdm

max_epochs = 6

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


training_history = {'nll': [], 'acc': []}
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']

        training_history['nll'].append(avg_nll)
        training_history['acc'].append(avg_accuracy)

pbar = tqdm(
    total=max_epochs
)
@trainer.on(Events.EPOCH_COMPLETED)
def show_bar(engine):
    pbar.update(1)

trainer.run(train_loader, max_epochs=max_epochs)

### Evaluate
import matplotlib.pyplot as plt

plt.plot(range(max_epochs), training_history['nll'], 'dodgerblue', label='nll')
plt.plot(range(max_epochs), training_history['acc'], 'orange', label='acc')
plt.xlim(0, max_epochs)
plt.xlabel('Epoch')
plt.ylabel('BCE')
plt.title('Binary Cross Entropy on Training/Validation Set')
plt.legend()
plt.show()