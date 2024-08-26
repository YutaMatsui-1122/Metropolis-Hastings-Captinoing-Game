import numpy as np
import matplotlib.pyplot as plt
import os

def load_losses(directory, prefix, start=0):
    losses = []
    index = start
    while True:
        filename = os.path.join(directory, f"{prefix}{index}_loss.npy")
        if not os.path.exists(filename):
            break
        loss = np.load(filename)
        losses.extend(loss)
        index += 1
    return losses

def average_losses(losses, interval=157):
    avg_losses = []
    for i in range(0, len(losses), interval):
        avg_losses.append(np.mean(losses[i:i+interval]))
    return avg_losses

def plot_losses(avg_losses, directory):
    plt.plot(avg_losses)
    plt.xlabel('Iteration (x157)')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time (Averaged)')
    plt.savefig(os.path.join(directory, 'loss_plot.png'))

if __name__ == "__main__":
    # ディレクトリ名とファイルのプレフィックスを指定
    directory = 'exp/GEMCG_first_1/B'
    prefix = 'clipcap_B_'
    
    losses = load_losses(directory, prefix)
    if losses:
        avg_losses = average_losses(losses,)
        plot_losses(avg_losses, directory)
    else:
        print("No loss files found.")
