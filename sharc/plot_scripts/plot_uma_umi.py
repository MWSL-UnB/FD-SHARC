
import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler

from sharc.propagation.propagation_uma import PropagationUMa
from sharc.propagation.propagation_umi import PropagationUMi
from sharc.propagation.propagation_free_space import PropagationFreeSpace

rnd = np.random.RandomState(1)
###########################################################################
# Print LOS probability
distance_2D = np.column_stack((np.linspace(1, 10000, num=10000)[:, np.newaxis],
                               np.linspace(1, 10000, num=10000)[:, np.newaxis],
                               np.linspace(1, 10000, num=10000)[:, np.newaxis]))
h_ue = np.array([1.5, 17, 23])
uma = PropagationUMa(rnd)
umi = PropagationUMi(rnd)

los_probability_uma = np.empty(distance_2D.shape)
name_uma = list()
los_probability_umi = np.empty(distance_2D.shape)
name_umi = list()

los_probability_uma = uma.get_los_probability(distance_2D, h_ue)
los_probability_umi = umi.get_los_probability(distance_2D[:, 0])

fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
ax = fig.gca()
ax.set_prop_cycle(cycler('color', ['g', 'g', 'g', 'b']) +
                  cycler('linestyle', ['-', '--', '-.', '-']))

for h in range(len(h_ue)):
    name_uma.append("UMa, $h_u = {:4.1f}$ $m$".format(h_ue[h]))
    ax.loglog(distance_2D[:, h], los_probability_uma[:, h], label=name_uma[h])
ax.loglog(distance_2D[:, 0], los_probability_umi, label="UMi")

plt.xlabel("Distância [m]")
plt.ylabel("Probabilidade")
plt.xlim((1, distance_2D[-1, 0]))
plt.ylim((0, 1.1))
plt.legend(loc="lower left")
plt.tight_layout()
plt.grid()

# Plot loss
shadowing_std = 0
distance_2D = np.linspace(1, 10000, num=10000)[:, np.newaxis]
freq = 27000 * np.ones(distance_2D.shape)
h_bs = 25 * np.ones(len(distance_2D[:, 0]))
h_ue = 1.5 * np.ones(len(distance_2D[0, :]))
h_e = np.zeros(distance_2D.shape)
distance_3D = np.sqrt(distance_2D ** 2 + (h_bs[:, np.newaxis] - h_ue) ** 2)

fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
ax = fig.gca()
ax.set_prop_cycle(cycler('color', ['g', 'g', 'b', 'b', 'r']) +
                  cycler('linestyle', ['-', '--', '-', '--', ':']))

loss_los = uma.get_loss_los(distance_2D, distance_3D, freq, h_bs, h_ue, h_e, shadowing_std)
loss_nlos = uma.get_loss_nlos(distance_2D, distance_3D, freq, h_bs, h_ue, h_e, shadowing_std)
ax.semilogx(distance_2D, loss_los, label="UMa LOS")
ax.semilogx(distance_2D, loss_nlos, label="UMa NLOS")

loss_los = umi.get_loss_los(distance_2D, distance_3D, freq, h_bs, h_ue, h_e, shadowing_std)
loss_nlos = umi.get_loss_nlos(distance_2D, distance_3D, freq, h_bs, h_ue, h_e, shadowing_std)
ax.semilogx(distance_2D, loss_los, label="UMi LOS")
ax.semilogx(distance_2D, loss_nlos, label="UMi NLOS")

loss_fs = PropagationFreeSpace(rnd).get_loss(distance_2D=distance_2D, frequency=freq)
ax.semilogx(distance_2D, loss_fs, label="Espaço livre")

plt.xlabel("Distância [m]")
plt.ylabel("Perda de propagação [dB]")
plt.xlim((1, distance_2D[-1, 0]))
# plt.ylim((0, 1.1))
plt.legend(loc="upper left")
plt.tight_layout()
plt.grid()

plt.show()
