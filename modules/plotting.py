import numpy as np
import matplotlib.pyplot as plt

def plot (before,after):
    counts_before, bins_before = np.histogram(before,bins=np.arange(0,200,1))
    plt.hist(bins_before[:-1], bins_before, weights=counts_before, label='Before')

    counts_after, bins_after = np.histogram(after,bins=np.arange(0,200,1))
    plt.hist(bins_after[:-1], bins_after, weights=counts_after, label='After',histtype='step')
    plt.legend(loc="best")
    plt.show()
