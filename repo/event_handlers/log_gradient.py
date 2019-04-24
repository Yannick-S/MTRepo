def log_gradient(engine, model, param_history):

    for i, param in enumerate(model.parameters()):
        if param.grad is None:
                continue
        param_history[i].append(param.grad.abs().max())

