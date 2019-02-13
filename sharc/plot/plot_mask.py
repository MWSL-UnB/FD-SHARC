# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:15:39 2017

@author: Calil
"""

from sharc.spectral_mask_imt import SpectralMaskImt
from sharc.support.enumerations import StationType
import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
sta_type = StationType.IMT_BS
p_tx = 24.30
freq = 43000
band = 200

# Create mask
msk = SpectralMaskImt(sta_type, freq, band)
msk.set_mask(power=p_tx)

# Frequencies
freqs = np.linspace(-3000, 3000, num=5000) + freq
freqs_ghz = freqs / 1000

# Mask values
mask_val = np.ones_like(freqs) * msk.mask_dbm[0]
for k in range(len(msk.freq_lim) - 1, -1, -1):
    mask_val[np.where(freqs < msk.freq_lim[k])] = msk.mask_dbm[k]

# RAS Band
freq_ras = np.linspace(42.5, 43.5, num=1000)
ras_val = np.zeros_like(freq_ras)
ras_val[0] = np.min(mask_val) - 10
ras_val[-1] = np.min(mask_val) - 10

# Plot
plt.figure(figsize=(10, 5))
plt.plot(freqs_ghz, mask_val, 'k', label="Máscara espectral IMT")
plt.plot(freq_ras, ras_val, 'r--', label="Banda da RAS")
plt.legend(loc=2)
plt.xlim([42.3, 43.7])
plt.ylim([-22, 7])
plt.xlabel("Frequência [GHz]")
plt.ylabel("Densidade espectral de potência [dBm/MHz]")
plt.grid()
plt.show()
