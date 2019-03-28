import torch

def handler_save_model(engine, save_every, model, optimizer, training_history, path, start_epoch):
    true_epoch = engine.state.epoch  + start_epoch
    if not engine.state.epoch  % save_every == 0:
        return
    print("Saving: ", path + 'epoch_{:05d}.pth'.format(true_epoch))
    torch.save({
        'epoch': true_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history
        },
        path + 'epoch_{:05d}.pth'.format(true_epoch)
    )
