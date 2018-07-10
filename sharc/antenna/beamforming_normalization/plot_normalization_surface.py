# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:26:10 2018

@author: Calil
"""

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

# Load file
file_name = 'ue_norm_4x4.npz'
data = np.load(file_name)
data_dict = {key:data[key] for key in data}
data.close()

# Print max and min
norm_max = np.amax(data_dict['correction_factor_co_channel'])
norm_min = np.amin(data_dict['correction_factor_co_channel'])

# Create plot data
phi_min = -180
phi_max = +180
theta_min = 0
theta_max = 180
x,y = np.meshgrid(np.arange(theta_min,theta_max,data_dict['resolution']),
                            np.arange(phi_min,phi_max,data_dict['resolution']))
z = data_dict['correction_factor_co_channel']

# Plot
pl.figure()
ax = pl.subplot(111, projection='3d')
ax.plot_surface(x,y,z)
pl.show()