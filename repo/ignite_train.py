from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from utility.checknotebook import in_ipynb
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import datetime 
import os
from plot_results.setup_file import save_file, load_file
from utility.tictoc import TicToc
import torch

def run(model, 
    optimizer,
    scheduler,
    loss_fn, 
    device, 
    train_loader, 
    training_history,
    param_history,
    model_info,
    start_epoch,
    path
    ):

    expected_batch_size = model_info["data"]["batch_size"]

    def prep_batch(batch, device=device, non_blocking=False):
        return batch.to(device), batch.y.to(device)
    
    def update(trainer, batch):
        nonlocal expected_batch_size

        model.train()
        optimizer.zero_grad()
        x, y = prep_batch(batch, device=device, non_blocking=False)

        if expected_batch_size != x.num_graphs:
            print(expected_batch_size)
            print(type(x))
            print(x.num_graphs)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        ### do clipping here
        for param in model.parameters():
            if param.grad is None:
                continue
            param.grad.data.clamp_(-1,1)

        optimizer.step()

        return loss.item()    

    trainer = Engine(update)

    evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': Accuracy(),
                                        'nll': Loss(loss_fn)},
                                        device=device,
                                        prepare_batch=prep_batch)

    optimizer_history = []
    from event_handlers.log_lr import log_lr
    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        log_lr,
        optimizer,
        optimizer_history)

    from event_handlers.scheduler import do_scheduler
    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        do_scheduler,
        optimizer, scheduler)
    
    from event_handlers.log_gradient import log_gradient
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_gradient,
            model, param_history)

    from event_handlers.log_training import log_training_results
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_training_results,
            evaluator, train_loader, training_history)

    from event_handlers.save_model import handler_save_model
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            handler_save_model, 
            model_info["training"]["save_every"], model, optimizer, training_history, param_history, path, start_epoch)

    from event_handlers.save_img import log_img 
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_img, 
            model_info["training"]["save_every"], training_history, param_history, path, start_epoch,optimizer_history)

    pbar = tqdm(total=model_info["training"]["max_epochs"])
    @trainer.on(Events.EPOCH_COMPLETED)
    def show_bar(engine):
        pbar.update(1)
    trainer.run(train_loader, max_epochs=model_info["training"]["max_epochs"])
    pbar.close()


def run_LR_find(model, 
    optimizer,
    loss_fn, 
    device, 
    train_loader, 
    training_history,
    param_history,
    model_info,
    start_epoch,
    path
    ):
    def prep_batch(batch, device=device, non_blocking=False):
        return batch.to(device), batch.y.to(device)
    
    def update(trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prep_batch(batch, device=device, non_blocking=False)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        ### do clipping here
        for param in model.parameters():
            param.grad.data.clamp_(-1,1)

        optimizer.step()
        return loss.item()    

    trainer = Engine(update)

    torch.save(model.state_dict(), '/tmp/model.pth')

    evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': Accuracy(),
                                        'nll': Loss(loss_fn)},
                                        device=device,
                                        prepare_batch=prep_batch)

    from event_handlers.log_training import log_training_results
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_training_results,
            evaluator, train_loader, training_history)

    trainer.run(train_loader, max_epochs=model_info["training"]["max_epochs"])

    model.load_state_dict(torch.load('/tmp/model.pth'))

    return training_history











