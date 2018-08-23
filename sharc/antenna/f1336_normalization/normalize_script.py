# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:49:29 2018

@author: Calil

This script generates the correction factors for the F.1336 Antennas,
and saves them in files with the given names.
This script must be ran with the appropriate parameters prior to using any
F.1336 normalization in the SHARC simulator, since the simulator merely reads the
correction factor values from the saved files.

Variables:
    resolution (float): resolution of the azimuth and elevation angles in the
        antenna array correction integral [deg].
    tolerance (float): absolute tolerance of the correction factor integral, in
        linear scale.
    norm (AntennaSectorF1336Normalizer): object that calculates the normalization.
    param_list (list): list of antenna parameters to which calculate the
        correction factors. New parameters are added as:
            AntennaPar(normalization,
                       norm_data,
                       element_pattern,
                       element_max_g,
                       element_phi_deg_3db,
                       element_theta_deg_3db,
                       element_am,
                       element_sla_v,
                       n_rows,
                       n_columns,
                       element_horiz_spacing,
                       element_vert_spacing,
                       downtilt_deg)
            normalization parameter must be set to False, otherwise script will
            try to normalize an already normalized antenna.
    file_names (list): list of file names to which save the normalization data.
        Files are paired with AntennaPar objects in param_list, so that the
        normalization data of the first element of param_list is saved in a
        file with the name specified in the first element of file_names and so
        on.

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

"""
from sys import stdout
from sharc.support.named_tuples import AntennaPar
from sharc.antenna.f1336_normalization.f1336_normalizer import AntennaSectorF1336Normalizer

###############################################################################
## List of antenna parameters to which calculate the normalization factors.
param_list = [AntennaPar(False,None,"",15,65,0,0,0,0,0,0,0,-10)]
file_names = ['bs_norm_f1336_15_65_0.npz']
###############################################################################
## Setup
# General parameters
resolution = 2
tolerance = 1e-2

# Create object
norm = AntennaSectorF1336Normalizer(resolution,tolerance)
###############################################################################
## Normalize and save
for par, file in zip(param_list,file_names):
    s = 'Generating ' + file
    print(s)
    stdout.flush()
    norm.generate_correction_matrix(par,file)
