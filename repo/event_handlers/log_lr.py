def log_lr(engine, optimizer, optimizer_history):
    for param_group in optimizer.param_groups:
        optimizer_history.append(param_group['lr'])
