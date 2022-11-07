import numpy as np
import matplotlib.pyplot as plt
import pickle
import modules.data_processing as data_processing
from matplotlib.backends.backend_pdf import PdfPages

import sys

def plot (before_path,after_path):
    with open(before_path, 'rb') as handle:
        before = pickle.load(handle)
    with open(after_path, 'rb') as handle:
        after = pickle.load(handle)

    response = (after-before)/before

    columns = data_processing.get_columns(before)
    number_of_columns = len(columns)

    with PdfPages(after_path.split("after.pickle")[0]+"comparison.pdf") as pdf:
        figure1, ax1 = plt.subplots(figsize=(18.3*(1/2.54)*1.7, 13.875*(1/2.54)*1.32))
        for index, column in enumerate(columns):
            print(f'{index} of {number_of_columns}')

            minimum = int(min(before[column]+after[column]))
            maximum = int(max(before[column]+after[column]))
            diff = maximum - minimum
            step = int(diff/500)

            #counts_before, bins_before = np.histogram(before[column],bins=np.arange(minimum,maximum,step))
            counts_before, bins_before = np.histogram(before[column],bins=np.arange(0,200,1))
            ax1.hist(bins_before[:-1], bins_before, weights=counts_before, label='Before')
        
            #counts_after, bins_after = np.histogram(after[column],bins=np.arange(minimum,maximum,step))
            counts_after, bins_after = np.histogram(after[column],bins=np.arange(0,200,1))
            ax1.hist(bins_after[:-1], bins_after, weights=counts_after, label='After',histtype='step')

            ax1.set_title(f"{column}")
            ax1.set_xlabel(column, ha='right', x=1.0)
            ax1.set_ylabel("Counts", ha='right', y=1.0)

            ax1.legend(loc="best")
            pdf.savefig()
            ax1.clear()
            #if index==10: break

    with PdfPages(after_path.split("after.pickle")[0]+"response.pdf") as pdf:
        figure2, ax2 = plt.subplots(figsize=(18.3*(1/2.54)*1.7, 13.875*(1/2.54)*1.32))
        for index, column in enumerate(columns):
            print(f'{index} of {number_of_columns}')

            minimum = min(response[column])
            maximum = max(response[column])
            diff = maximum - minimum
            if diff == np.inf: continue
            if diff == 0: continue
            step = int(diff/100)

            #counts_response, bins_response = np.histogram(response[column],bins=np.arange(minimum,maximum,step))
            counts_response, bins_response = np.histogram(response[column],bins=np.arange(-20,20,1))
            ax2.hist(bins_response[:-1], bins_response, weights=counts_response, label='Response')

            ax2.set_title(f"{column} Response")
            ax2.set_xlabel(f'{column} Response', ha='right', x=1.0)
            ax2.set_ylabel("Counts", ha='right', y=1.0)

            pdf.savefig()
            ax2.clear()
            #if index==10: break
