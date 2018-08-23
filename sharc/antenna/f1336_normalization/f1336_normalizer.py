# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 08:47:17 2018

@author: Calil
"""

import numpy as np
from scipy.integrate import dblquad

from sharc.antenna.antenna_sector_f1336 import AntennaSectorF1336
from sharc.support.named_tuples import AntennaPar
from sys import stdout


class AntennaSectorF1336Normalizer(object):
    """
    
    """
    def __init__(self, res_deg: float, tol: float):
        """
        Class constructor

        Parameters:
            res_deg (float): correction factor matrix resolution in degrees
            tol (float): absolute tolerance for integration
        """
        # Initialize attributes
        self.resolution_deg = res_deg
        self.tolerance = tol
        
        self.phi_min_deg = -180
        self.phi_max_deg = 180
        self.theta_min_deg = 0
        self.theta_max_deg = 180

        self.phi_min_rad = np.deg2rad(self.phi_min_deg)
        self.phi_max_rad = np.deg2rad(self.phi_max_deg)
        self.theta_min_rad = np.deg2rad(self.theta_min_deg)
        self.theta_max_rad = np.deg2rad(self.theta_max_deg)

        self.phi_vals_deg = np.arange(self.phi_min_deg,
                                      self.phi_max_deg,res_deg)
        self.theta_vals_deg = np.arange(self.theta_min_deg,
                                        self.theta_max_deg,res_deg)
        
        self.antenna = None

    def generate_correction_matrix(self, par: AntennaPar, file_name: str):
        """
        Generates the correction factor matrix and saves it in a file

        Parameters:
            par (AntennaPar): set of antenna parameters to which calculate the
                correction factor
            file_name (str): name of file to which save the correction matrix
        """
        # Create antenna object
        azi = 0 # Antenna azimuth: 0 degrees for simplicity
        ele = 0 # Antenna elevation: 0 degrees as well

        # Loop throug all the possible beams
        dtilt = par.downtilt_deg
        s = '\ttilt = ' + str(dtilt)
        print(s)
        stdout.flush()
        self.antenna = AntennaSectorF1336(par,dtilt,azi,ele)
        correction_factor, error = self.calculate_correction_factor()

        # Save in file
        self._save_files(correction_factor,
                         error,
                         par,
                         file_name)

    def calculate_correction_factor(self):
        """
        Calculates single correction factor

        Returns:
            correction_factor (float): correction factor value [dB]
            error (tuple): upper and lower error bounds [dB]
        """
        int_f = lambda t,p: \
        np.power(10,self.antenna.calculate_gain(phi_vec=np.rad2deg(p),
                                                theta_vec=np.rad2deg(t))/10)*np.sin(t)

        integral_val, err = dblquad(int_f,self.phi_min_rad,self.phi_max_rad,
                          lambda p: self.theta_min_rad,
                          lambda p: self.theta_max_rad,
                          epsabs=self.tolerance,
                          epsrel=0.0)

        correction_factor = -10*np.log10(integral_val/(4*np.pi))

        hig_bound = -10*np.log10((integral_val - err)/(4*np.pi))
        low_bound = -10*np.log10((integral_val + err)/(4*np.pi))

        return correction_factor, (low_bound,hig_bound)

    def _save_files(self, cf, err, par, file_name):
        """
        Saves input correction factor and error values to npz file.
        Data is saved in an .npz file in a dict like data structure with the
        following keys:
            resolution (float): antenna array correction factor matrix angle
                resolution [deg]
            correction_factor (1D np.array): correction factor [dB]
                for each of down tilt values in down_tilt_range.
            error (1D np.array of tuples): lower and upper bounds of
                calculated correction factors [dB], considering integration
                error
            parameters (AntennaPar): antenna parameters used in the
                normalization

        Parameters:
            cf (1D np.array): co-channel correction factor [dB]
            err (2D np.array): co-channel correction factor lower and upper
                bounds considering integration errors [dB]
            par (AtennaPar): antenna parameters used in normalization
            file_name (str): name of file to which save normalization data
        """
        np.savez(file_name,
                 resolution = self.resolution_deg,
                 correction_factor = cf,
                 error = err,
                 parameters = par)

if __name__ == '__main__':
    """
    Plots correction factor for horizontal and vertical planes.
    """
    import matplotlib.pyplot as plt
    import os

    # Create normalizer object
    resolution = 10
    tolerance = 1e-1
    norm = AntennaSectorF1336Normalizer(resolution,tolerance)

    # Antenna parameters
    normalization = False
    norm_data = None
    element_pattern = "M2101"
    element_max_g = 15
    element_phi_deg_3db = 65
    element_theta_deg_3db = 0
    element_am = 30
    element_sla_v = 30
    n_rows = 8
    n_columns = 8
    horiz_spacing = 0.5
    vert_spacing = 0.5
    down_tilt = -10
    par = AntennaPar(normalization,
                     norm_data,
                     element_pattern,
                     element_max_g,
                     element_phi_deg_3db,
                     element_theta_deg_3db,
                     element_am,
                     element_sla_v,
                     n_rows,
                     n_columns,
                     horiz_spacing,
                     vert_spacing,
                     down_tilt)

    # Set range of values & calculate correction factor
    file_name = 'main_test.npz'
    norm.generate_correction_matrix(par,file_name)
    data = np.load(file_name)
    print(data['correction_factor'])
    print(data['error'])
    data.close()
#    os.remove(file_name)
