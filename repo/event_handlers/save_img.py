from plot_results.plot_hist import plot_grad,plot_grad2, plot_hist_acc, plot_hist_loss, save_plot, plot_hist_lr

def log_img(engine, save_every, training_history, param_history, path, start_epoch, optimizer_history):
    if not engine.state.epoch  % save_every == 0:
        return

    true_epoch = engine.state.epoch  + start_epoch

    plot_grad(param_history, show=False)
    save_plot(path + 'epoch_{:05d}_grad.eps'.format(true_epoch))

    plot_grad2(param_history, show=False)
    save_plot(path + 'epoch_{:05d}_grad2.eps'.format(true_epoch))

    plot_hist_acc(training_history['acc'])
    save_plot(path + 'epoch_{:05d}_acc.eps'.format(true_epoch))

    plot_hist_loss(training_history['nll'])
    save_plot(path + 'epoch_{:05d}_nll.eps'.format(true_epoch))

    plot_hist_lr(optimizer_history)
    save_plot(path + 'epoch_{:05d}_lr.eps'.format(true_epoch))