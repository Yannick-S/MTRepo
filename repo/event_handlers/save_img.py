from plot_results.plot_hist import plot_hist_acc, plot_hist_loss, save_plot

def log_img(engine, save_every, training_history, path, start_epoch):
    if not engine.state.epoch  % save_every == 0:
        return

    true_epoch = engine.state.epoch  + start_epoch

    plot_hist_acc(training_history['acc'])
    save_plot(path + 'epoch_{:05d}_acc.eps'.format(true_epoch))

    plot_hist_loss(training_history['nll'])
    save_plot(path + 'epoch_{:05d}_nll.eps'.format(true_epoch))
