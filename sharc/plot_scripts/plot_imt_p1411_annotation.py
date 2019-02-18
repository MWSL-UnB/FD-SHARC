# Imports
import matplotlib.pyplot as plt
import numpy as np

from sharc.propagation.propagation_imt_p1411 import PropagationImtP1411

# Create propagation object
rnd = np.random.RandomState(1536)
propag = PropagationImtP1411(rnd)

d_min = 1
d_max = 1e3
data_num = 1000

# Input parameters
dist = np.linspace(d_min, d_max, num=data_num)
freq = 27e6 * np.ones_like(dist)

# No shadowing
shad = False
# Calculate loss
loss = propag.get_loss(distance_3D=dist, frequency=freq, shadow=shad)
# Plot with shadowing loss
plt.semilogx(dist, loss, label='Modelo UE-UE', linewidth=2.0)

loss_nlos = propag.calculate_loss(propag.nlos_alpha, propag.nlos_beta, propag.nlos_gamma, 0, dist, freq/1e3)
plt.semilogx(dist, loss_nlos, 'g-.', label="P.1411 NLOS", linewidth=1.0)
loss_los = propag.calculate_loss(propag.los_alpha, propag.los_beta, propag.los_gamma, 0, dist, freq/1e3)
plt.semilogx(dist, loss_los, 'g--', label="P.1411 LOS", linewidth=1.0)
loss_fspl = propag.calculate_loss(propag.fspl_alpha, propag.fspl_beta, propag.fspl_gamma, 0, dist, freq/1e3)
plt.semilogx(dist, loss_fspl, 'g:', label="Espaço livre", linewidth=1.0)

# plt.annotate('FSPL', (1.5, 140))
# plt.annotate('LOS', (7.5, 160))
# plt.annotate('NLOS', (80, 210))

# line_format = 'r:'
# line_width = 1.0
# line_x = propag.fspl_to_los_dist * np.ones(100)
# line_y = np.linspace(np.min(loss), np.max(loss), num=100)
# plt.semilogx(line_x, line_y, line_format, linewidth=line_width)
# line_x = propag.los_dist * np.ones(100)
# line_y = np.linspace(np.min(loss), np.max(loss), num=100)
# plt.semilogx(line_x, line_y, line_format, linewidth=line_width)
# line_x = propag.los_to_nlos_dist * np.ones(100)
# line_y = np.linspace(np.min(loss), np.max(loss), num=100)
# plt.semilogx(line_x, line_y, line_format, linewidth=line_width)
# line_x = propag.nlos_dist * np.ones(100)
# line_y = np.linspace(np.min(loss), np.max(loss), num=100)
# plt.semilogx(line_x, line_y, line_format, linewidth=line_width)

plt.xlabel("Distância [m]")
plt.ylabel("Perda de propagação [dB]")
plt.legend()
plt.grid()
plt.xlim((d_min, d_max))
plt.ylim((np.min(loss), np.max(loss)))
plt.show()
