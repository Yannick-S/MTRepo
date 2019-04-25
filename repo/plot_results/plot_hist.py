import matplotlib.pyplot as plt
import torch 
import numpy as np 

def plot_lr(all_history, all_lrs, show=True):
    tensor = np.zeros((len(all_history), all_history[0].shape[0]))
    for i in range(len(all_history)):
        tensor[i] = all_history[i] 
        n = tensor.shape[1]

        print("banana" , i)
        plt.plot(np.array(range(n)), tensor[i], color='C'+str(i%10))
    #plt.plot(np.transpose(tensor))
    plt.legend(all_lrs)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.title('LR Finding')
    if show:
        plt.show()

def plot_grad(param_history, show=False):
    return
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

def plot_hist_loss(optimizer_history, color='dodgerblue', show=False):
    n = len(optimizer_history)
    plt.plot(range(n), optimizer_history, color)
    plt.xlim(0, n)
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate')
    if show:
        plt.show()

def plot_hist_lr(training_history, color='fuchsia', show=False):
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