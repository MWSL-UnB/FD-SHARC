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
        bel = kwargs.pop("bel_enabled", True)
        es_par = kwargs["es_params"]
        tx_g = np.ravel(np.asarray(kwargs["tx_gain"]))
        rx_g = np.ravel(np.asarray(kwargs["rx_gain"]))

        loss = self.p452.get_loss(distance_3D=distance,
                                  frequency=f,
                                  number_of_sectors=num_sec,
                                  indoor_stations=indoor_stas,
                                  elevation=ele,
                                  bel_enabled=bel,
                                  es_params=es_par,
                                  tx_gain=tx_g,
                                  rx_gain=rx_g)

        loss[np.isnan(loss)] = es_par.co_site_bs_loss

        np.fill_diagonal(loss, np.nan)

        return loss
