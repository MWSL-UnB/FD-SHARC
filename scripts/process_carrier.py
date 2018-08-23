# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:29:25 2017

@author: edgar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def process(simulation_cases: list, 
            segment_factor: np.array, 
            num_snapshots: int, 
            dl_tdd_factor: float) -> np.array:
    
    inr = np.zeros((len(simulation_cases), num_snapshots))
    
    for c in range(len(simulation_cases)):
        file_name = "\\[SYS] INR samples.txt"
        #freq = 'SHARC43'
        freq = 'SHARC48'

        file_dl = open(freq + "DL" + simulation_cases[c] + file_name, "r") 
        samples_dl = list()
        for line in file_dl:
            if not line.startswith("#"):
                #print(float(line[-10:-1]))
                samples_dl.append(float(line[-10:-1]))
        file_dl.close()

        file_ul = open(freq + "UL" + simulation_cases[c] + file_name, "r") 
        samples_ul = list()
        for line in file_ul:
            if not line.startswith("#"):
                #print(float(line[-10:-1]))
                samples_ul.append(float(line[-10:-1]))
        file_ul.close()
        
        # 1) This is the case where we select I/N samples according to TDD factor
        #    and apply a single processing on top of them
#        for n in range(num_snapshots):
#            num_dl = np.count_nonzero(np.random.random(segment_factor[c]) < dl_tdd_factor)
#            dl = np.random.choice(samples_dl, num_dl)
#            ul = np.random.choice(samples_ul, segment_factor[c] - num_dl)
#            inr[c,n] = 10*np.log10(np.sum(np.power(10, 0.1*np.concatenate((dl, ul)))))

        # 2) In this case, we process DL and UL separatelly and apply the 
        #    TDD factor at the end
        for n in range(num_snapshots):
            dl = np.random.choice(samples_dl, segment_factor[c])
            dlm = 10*np.log10(np.sum(np.power(10, 0.1*dl)))
            ul = np.random.choice(samples_ul, segment_factor[c])
            ulm = 10*np.log10(np.sum(np.power(10, 0.1*ul)))            
            inr[c,n] = 10*np.log10(dl_tdd_factor*np.power(10, 0.1*dlm) + (1-dl_tdd_factor)*np.power(10, 0.1*ulm))        
        
    return inr
    
    
def create_cdf(inr: np.array, n_bins: int) -> tuple:
    num_simulation_cases = len(inr)
    x = np.empty((num_simulation_cases, n_bins))
    y = np.empty((num_simulation_cases, n_bins))
    for c in range(num_simulation_cases):
        values, base = np.histogram(inr[c], bins = n_bins)
        cumulative = np.cumsum(values)
        x[c,:] = base[:-1]
        y[c,:] = cumulative / cumulative[-1]
    return x, y

    
def plot_inr(x_i: np.array, y_i: np.array, label_i: np.array):
    #file_name = "INR_plot_43"
    file_name = "INR_plot_48"
    file_extension = ".png"
    transparent_figure = False
    
    color = list(("-b", "-r", "-k", "-g"))
    
    plt.figure(figsize=(16,6), facecolor='w', edgecolor='k')
    plt.subplot(121)
    for x, y, label, c in zip(x_i, y_i, label_i, color):
        plt.plot(x, y, c, label = label, linewidth=1)        
    #plt.title("Carrier #06")
    plt.xlabel("$I/N$ [dB]")
    plt.ylabel("Probability of $I/N$ < $X$")
    #plt.xlim((-45, -20))
    plt.ylim((0, 1))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.grid()
    plt.savefig(file_name + file_extension, transparent = transparent_figure)        

    #plt.show()
    
    
        
if __name__ == "__main__":
    dl_tdd_factor = 0.8
    num_snapshots = 5000
    n_bins = 200

    #segment_factor = np.array([126, 197, 324, 454])
    segment_factor = np.array([126, 197, 454])
    simulation_cases = list(("90",
                             "45",
                             #"30",
                             "20"))

    label = np.array(("Elevation: $90^o$",
                      "Elevation: $45^o$",
                      #"Elevation: $30^o$",
                      "Elevation: $20^o$"))      
    
    inr = process(simulation_cases, segment_factor, num_snapshots, dl_tdd_factor)
    x, y = create_cdf(inr, n_bins)
    plot_inr(x, y, label)