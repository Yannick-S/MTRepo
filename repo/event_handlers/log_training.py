def log_training_results(engine, evaluator, train_loader, training_history):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']

        training_history['nll'].append(avg_nll)
        training_history['acc'].append(avg_accuracy)
