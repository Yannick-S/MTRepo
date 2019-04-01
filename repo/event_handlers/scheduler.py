def do_scheduler(engine, optimizer, scheduler):
    scheduler.batch_step()
    for param_group in optimizer.param_groups:
        print(param_group['lr'])