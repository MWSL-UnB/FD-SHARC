# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:03:24 2017

@author: edgar
"""

import sys
import numpy.random as rnd

from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_p619 import PropagationP619
from sharc.propagation.propagation_sat_simple import PropagationSatSimple
from sharc.propagation.propagation_ter_simple import PropagationTerSimple
from sharc.propagation.propagation_uma import PropagationUMa
from sharc.propagation.propagation_umi import PropagationUMi
from sharc.propagation.propagation_abg import PropagationABG
from sharc.propagation.propagation_clear_air_452 import PropagationClearAir
from sharc.propagation.propagation_tvro import PropagationTvro
from sharc.propagation.propagation_indoor import PropagationIndoor
from sharc.propagation.propagation_fd_indoor import PropagationFDIndoor
from sharc.propagation.propagation_hdfss import PropagationHDFSS
from sharc.propagation.propagation_imt_p1411 import PropagationImtP1411

class PropagationFactory(object):

    @staticmethod
    def create_propagation(channel_model: str, param: Parameters, random_number_gen: rnd.RandomState) -> Propagation:
        if channel_model == "FSPL":
            return PropagationFreeSpace(random_number_gen)
        elif channel_model == "ABG":
            return PropagationABG(random_number_gen)
        elif channel_model == "UMa":
            return PropagationUMa(random_number_gen)
        elif channel_model == "UMi":
            return PropagationUMi(random_number_gen)
        elif channel_model == "CI":
            return PropagationCloseIn(random_number_gen)
        elif channel_model == "SatelliteSimple":
            return PropagationSatSimple(random_number_gen)
        elif channel_model == "TerrestrialSimple":
            return PropagationTerSimple(random_number_gen)
        elif channel_model == "P619":
            return PropagationP619(random_number_gen)
        elif channel_model == "P452":
            return PropagationClearAir(random_number_gen)
        elif channel_model == "TVRO":
            return PropagationTvro(random_number_gen)
        elif channel_model == "HDFSS":
            return PropagationHDFSS(random_number_gen)
        elif channel_model == "INDOOR":
            return PropagationIndoor(random_number_gen, param.indoor, param.imt.ue_k * param.imt.ue_k_m)
        elif channel_model == "FD_INDOOR":
            return PropagationFDIndoor(random_number_gen, param.indoor, param.imt.ue_k * param.imt.ue_k_m)
        elif channel_model == "IMT_P1411":
            return PropagationImtP1411(random_number_gen)
        elif channel_model == "NONE":
            return None
        else:
            sys.stderr.write("ERROR\nInvalid channel_model: " + channel_model)
            sys.exit(1)
