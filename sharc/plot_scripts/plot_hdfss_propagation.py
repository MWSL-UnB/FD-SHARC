
import matplotlib.pyplot as plt
import numpy as np

from sharc.support.enumerations import StationType
from sharc.parameters.parameters_fss_es import ParametersFssEs
from sharc.propagation.propagation_p1411 import PropagationP1411
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_hdfss_roof_top import PropagationHDFSSRoofTop

rnd = np.random.RandomState(55)

d_min = 1
d_max = 1e5
data_num = 1000

# Generate

par = ParametersFssEs()
par.building_loss_enabled = False
par.shadow_enabled = False
par.same_building_enabled = False
par.diffraction_enabled = False
par.bs_building_entry_loss_type = 'FIXED_VALUE'
par.bs_building_entry_loss_prob = 0.5
par.bs_building_entry_loss_value = 50

dist = np.array([np.logspace(np.log10(d_min), np.log10(d_max), num=data_num)])
freq = 40e3 * np.ones_like(dist)

prop = PropagationHDFSSRoofTop(par, rnd)
loss = prop.get_loss(distance_3D=dist,
                     frequency=freq,
                     elevation=0.0,
                     imt_sta_type=StationType.IMT_BS,
                     imt_x=1000.0*np.ones(dist.shape[1]),
                     imt_y=1000.0*np.ones(dist.shape[1]),
                     imt_z=6.0*np.ones(dist.shape[1]),
                     es_x=np.array([0.0]),
                     es_y=np.array([0.0]),
                     es_z=np.array([10.0]))
plt.semilogx(dist[0], loss[0][0, :], 'b-',label='Modelo HDFSS', linewidth=2.0)

prop = PropagationP1411(rnd)
loss = prop.get_loss(distance_3D=dist[0],
                     frequency=freq[0],
                     shadow=False,
                     los=False)
plt.semilogx(dist[0], loss, 'g-.',label='P.1411 NLOS', linewidth=1.0)

loss = prop.get_loss(distance_3D=dist[0],
                     frequency=freq[0],
                     shadow=False,
                     los=True)
plt.semilogx(dist[0], loss, 'g--',label='P.1411 LOS', linewidth=1.0)

prop = PropagationFreeSpace(rnd)
loss = prop.get_loss(distance_3D=dist[0],
                     frequency=freq[0])
plt.semilogx(dist[0], loss, 'g:',label='Espaço livre', linewidth=1.0)

plt.xlabel("Distância [m]")
plt.ylabel("Perda de propagação [dB]")
plt.xlim((d_min, d_max))
plt.legend(loc="upper left")
plt.tight_layout()
plt.grid()
plt.show()




