import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Search all files with txt extension in folder
files = glob('*.txt')

for file in files:
    print(file)
    # Collect and plot_scripts data
    data = np.loadtxt(file,skiprows=1)
    plt.figure()
    if 'CDF' in file:
        x = data[:,0]
        y = data[:,1]
    else:
        values, base = np.histogram(data[:,1], bins=200)
        cumulative = np.cumsum(values)
        x = base[:-1]
        y = cumulative / cumulative[-1]

    plt.plot(x,y)
    # Save figure
    plot_name = file[:-4]
    plt.title(plot_name)
    plt.savefig(plot_name + ".png")
    plt.close()
