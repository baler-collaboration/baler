import numpy as np
import matplotlib.pyplot as plt
import pickle
import modules.data_processing as data_processing
from matplotlib.backends.backend_pdf import PdfPages

def plot (before_path,after_path):
    with open(before_path, 'rb') as handle:
        before = pickle.load(handle)
    with open(after_path, 'rb') as handle:
        after = pickle.load(handle)

    columns = data_processing.get_columns(before)
    number_of_columns = len(columns)

    figure, ax1 = plt.subplots(figsize=(18.3*(1/2.54)*1.7, 13.875*(1/2.54)*1.32))
    with PdfPages(after_path.split("after.pickle")[0]+"comparison.pdf") as pdf:
        for index, column in enumerate(columns):
            print(f'{index} of {number_of_columns}')
            counts_before, bins_before = np.histogram(before[column],bins=np.arange(0,200,1))
            ax1.hist(bins_before[:-1], bins_before, weights=counts_before, label='Before')
        
            counts_after, bins_after = np.histogram(after[column],bins=np.arange(0,200,1))
            ax1.hist(bins_after[:-1], bins_after, weights=counts_after, label='After',histtype='step')

            ax1.set_title(f"{column}")
            ax1.set_xlabel(column, ha='right', x=1.0)
            ax1.set_ylabel("Counts", ha='right', y=1.0)

            ax1.legend(loc="best")
            pdf.savefig()
            ax1.clear()
            #if index==10: break
