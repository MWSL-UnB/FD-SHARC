import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Search all files with txt extension in folder
files = glob('*.txt')

for file in files:
	# Collect and plot data
	data = np.loadtxt(file,skiprows=1)
	plt.figure()
	plt.plot(data[:,0],data[:,1])
	# Save figure
	plot_name = file[:-4]
	plt.title(plot_name)
	plt.savefig(plot_name + ".png")