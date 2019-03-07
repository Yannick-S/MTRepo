import matplotlib.pyplot as plt

def plot_hist(training_history):
    n = len(training_history['nll'])
    plt.plot(range(n), training_history['nll'], 'dodgerblue', label='nll')
    plt.plot(range(n), training_history['acc'], 'orange', label='acc')
    plt.xlim(0, n)
    plt.xlabel('Epoch')
    plt.ylabel('BCE')
    plt.title('Binary Cross Entropy on Training/Validation Set')
    plt.legend()
    plt.show()