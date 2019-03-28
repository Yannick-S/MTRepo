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

def run(model, 
    optimizer,
    loss, 
    device, 
    train_loader, 
    training_history,
    model_info,
    start_epoch,
    path
    ):
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
            model_info["training"]["save_every"], model, optimizer, training_history, path, start_epoch)

    from event_handlers.save_img import log_img 
    trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_img, 
            model_info["training"]["save_every"], training_history, path, start_epoch)

    pbar = tqdm(total=model_info["training"]["max_epochs"])
    @trainer.on(Events.EPOCH_COMPLETED)
    def show_bar(engine):
        pbar.update(1)
    trainer.run(train_loader, max_epochs=model_info["training"]["max_epochs"])
    pbar.close()






