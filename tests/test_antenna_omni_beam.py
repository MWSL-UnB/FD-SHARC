# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:22:17 2018

@author: Calil
"""

import numpy as np

import unittest
import numpy.testing as npt

from sharc.antenna.antenna_omni_beam import AntennaOmniBeam

class AntennaOmniBeamTest(unittest.TestCase):
    
    def setUp(self):
        self.antenna = AntennaOmniBeam()
        
    def test_calculate_gain(self):
        # Add a beam
        self.antenna.add_beam( 35.0, 45.0)
        self.antenna.add_beam(-35.0, 25.0)
        self.antenna.add_beam( 90.0,-40.0)
        
        # Create angles
        num_angles = 5
        phi_vec = np.ones(num_angles)
        theta_vec = np.ones(num_angles)
        beams_idx = np.array([0,1,2,1,0])
        
        # Calculate gains
        gains = self.antenna.calculate_gain(phi_vec=phi_vec,
                                            theta_vec=theta_vec,
                                            beams_l=beams_idx)
        
        # Assert
        npt.assert_equal(gains,beams_idx + 1)
        
        # Check error raise
        beams_idx = np.array([0,1,3,1,0])
        
        # Calculate gains
        with self.assertRaises(IndexError):
            gains = self.antenna.calculate_gain(phi_vec=phi_vec,
                                                theta_vec=theta_vec,
                                                beams_l=beams_idx)
        
        
if __name__ == '__main__':
    unittest.main()
