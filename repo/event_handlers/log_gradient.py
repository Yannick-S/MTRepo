def log_gradient(engine, model, param_history):
    for i, param in enumerate(model.parameters()):
        param_history[i].append(param.grad.abs().max())

