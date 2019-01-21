from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_clear_air_452 import PropagationClearAir

import numpy as np

class PropagationImtP452(Propagation):

    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)

        self.p452 = PropagationClearAir(random_number_gen)

    def get_loss(self, *args, **kwargs):

        distance = np.asarray(kwargs["distance_3D"])
        f = np.asarray(kwargs["frequency"])
        num_sec = kwargs.pop("number_of_sectors", 1)
        indoor_stas= kwargs.pop("indoor_stations", 1)
        ele = kwargs["elevation"]
        bel = kwargs.pop("bel_enabled", False)
        es_par = kwargs["es_params"]
        tx_g = kwargs["tx_gain"]
        rx_g = kwargs["rx_gain"]

        loss_mtx = np.zeros_like(distance)

        distance[np.isnan(distance)] = 100.0
        tx_g[np.isnan(tx_g)] = 0.0
        rx_g[np.isnan(rx_g)] = 0.0

        for k, dist in enumerate(distance):
            k_range = [m for m in range(3 * (k // 3), 3 * (k // 3) + 3)]
            loss = self.p452.get_loss(distance_3D=np.array([dist]),
                                      frequency=np.array([f[k]]),
                                      number_of_sectors=num_sec,
                                      indoor_stations=np.array([indoor_stas[k]]),
                                      elevation=ele,
                                      bel_enabled=bel,
                                      es_params=es_par,
                                      tx_gain=tx_g[k,:],
                                      rx_gain=np.reshape(rx_g[:,k_range], (171)))

            loss_mtx[k,:] = loss

        for k, ls in enumerate(loss):
            idx_range = [3 * (k // 3) + m for m in range(3)]
            loss_mtx[k, idx_range] = es_par.co_site_bs_loss
            loss_mtx[k, k] = np.nan

        return loss_mtx
