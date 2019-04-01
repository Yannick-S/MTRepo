import matplotlib.pyplot as plt
import torch 

def plot_grad(param_history, show=False):
    for i in param_history:
        tensor = torch.tensor(param_history[i]).numpy()
        n = tensor.shape[0]
        tmax = tensor.max()
        plt.plot(range(n), tensor/tmax,  color='C'+str(i%10))

    plt.xlabel('Epoch')
    plt.ylabel('Normailzed absolute max gradient')
    plt.title('Gradients')
    if show:
        plt.show()

def plot_grad2(param_history, show=False):
    for i in param_history:
        tensor = torch.tensor(param_history[i]).numpy()
        n = tensor.shape[0]
        plt.plot(range(n), tensor,  color='C'+str(i%10))

    plt.xlabel('Epoch')
    plt.ylabel('Normailzed absolute max gradient')
    plt.title('Gradients')
    if show:
        plt.show()

def plot_hist_loss(training_history, color='dodgerblue', show=False):
    n = len(training_history)
    plt.plot(range(n), training_history, color)
    plt.xlim(0, n)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.title('Negative Log Likelyhood')
    if show:
        plt.show()

def plot_hist_acc(training_history, color='orange', show=False):
    n = len(training_history)
    plt.plot(range(n), training_history, color)
    plt.xlim(0, n)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    if show:
        plt.show()

def save_plot(path):
    plt.savefig(path, format='eps', dpi=1000)
    plt.clf()
    plt.cla()