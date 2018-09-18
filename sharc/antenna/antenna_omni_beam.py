# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:06:55 2018

@author: Calil
"""

import numpy as np

from sharc.antenna.antenna import Antenna


class AntennaOmniBeam(Antenna):
    def __init__(self,):
        super().__init__()

    def add_beam(self, phi: float, theta: float):
        self.beams_list.append((phi,theta-90))

    def calculate_gain(self, *args, **kwargs) -> np.array:
        phi_vec = np.asarray(kwargs["phi_vec"])
        theta_vec = np.asarray(kwargs["theta_vec"])
        if("beams_l" in kwargs.keys()): 
            beams_l = np.asarray(kwargs["beams_l"],dtype=int)
        else: 
            beams_l = -1*np.ones_like(phi_vec)
           
        n_direct = len(theta_vec)

        gains = np.zeros(n_direct)
        
        if not len(theta_vec) == len(phi_vec) == len(beams_l):
            raise ValueError

        for g in range(n_direct):
            self.beams_list[beams_l[g]]
            gains[g] = beams_l[g] + 1 if beams_l[g] != -1 else beams_l[g]

        return gains

    def reset_beams(self):
        self.beams_list = []

