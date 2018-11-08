# -*- coding: utf-8 -*-
"""
Created on Mon Mar  13 15:14:34 2017

@author: edgar
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.propagation.propagation_imt_p1411 import PropagationImtP1411


class PropagationImtP1411Test(unittest.TestCase):

    def setUp(self):
        self.propagation = PropagationImtP1411(np.random.RandomState())

    def test_get_loss(self):
        # Test 1
        dist = np.array([[4.1, 4.5],
                         [4.7, 4.9]])
        freq = 1e8*np.ones_like(dist)
        loss = self.propagation.get_loss(distance_3D=dist,
                                         frequency=freq,
                                         shadow=False)
        npt.assert_allclose(loss, np.array([[144.97, 146.95], [147.94, 148.92]]), atol=1e-1)

        # Test 2
        dist = np.array([1.0, 4.1])
        freq = 1e8 * np.ones_like(dist)
        loss = self.propagation.get_loss(distance_3D=dist,
                                         frequency=freq,
                                         shadow=False)
        npt.assert_allclose(loss, np.array([132.4, 144.98]), atol=1e-1)

    def test_get_loss_range(self):
        # Test 1
        dist = np.array([0.5, 3.9, 4.5, 15, 27, 155.2])
        fspl, fspl_los, los, los_nlos, nlos = self.propagation.get_loss_range(dist)

        npt.assert_equal(fspl, [True, True, False, False, False, False])
        npt.assert_equal(fspl_los, [False, False, True, False, False, False])
        npt.assert_equal(los, [False, False, False, True, False, False])
        npt.assert_equal(los_nlos, [False, False, False, False, True, False])
        npt.assert_equal(nlos, [False, False, False, False, False, True])

        # Test 2
        dist = np.array([[0.5, 3.9, 4.5, 15, 27, 155.2],
                         [0.5, 3.9, 4.5, 15, 27, 155.2]])
        fspl, fspl_los, los, los_nlos, nlos = self.propagation.get_loss_range(dist)

        npt.assert_equal(fspl, [[True, True, False, False, False, False],
                                [True, True, False, False, False, False]])
        npt.assert_equal(fspl_los, [[False, False, True, False, False, False],
                                    [False, False, True, False, False, False]])
        npt.assert_equal(los, [[False, False, False, True, False, False],
                               [False, False, False, True, False, False]])
        npt.assert_equal(los_nlos, [[False, False, False, False, True, False],
                                    [False, False, False, False, True, False]])
        npt.assert_equal(nlos, [[False, False, False, False, False, True],
                                [False, False, False, False, False, True]])

    def test_calculate_loss(self):
        # Test 1
        self.propagation.shadow = False
        alpha = 2.0
        beta = 25.0
        gamma = 1.5
        sigma = 0.0
        dist = np.array([[1.3, 500],
                         [22.4, 33]])
        f = 1e5*np.ones_like(dist)
        loss = self.propagation.calculate_loss(alpha, beta, gamma, sigma, dist, f)
        npt.assert_allclose(loss, np.array([[102.28, 153.98], [127.01, 130.37]]), atol=1e-2)

    def test_interpolate(self):
        # Test 1
        dist = np.array([[4.1, 4.5],
                         [4.7, 4.9]])
        freq = 1e5*np.ones_like(dist)
        self.propagation.shadow = False
        loss_range = 'FSPL_TO_LOS'
        loss = self.propagation.interpolate(dist, freq, loss_range)
        npt.assert_allclose(loss, np.array([[144.97, 146.95], [147.94, 148.92]]), atol=1e-1)


if __name__ == '__main__':
    unittest.main()
