# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:28:11 2018

@author: Calil
"""

import unittest
import numpy as np
import numpy.testing as npt
import math

from sharc.simulation_tn_full_duplex import SimulationTNFullDuplex
from sharc.parameters.parameters import Parameters
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.antenna.antenna_omni_beam import AntennaOmniBeam
from sharc.station_factory import StationFactory
from sharc.propagation.propagation_factory import PropagationFactory
from sharc.propagation.propagation_for_test import PropagationForTest


class SimulationTNFullDuplexTest(unittest.TestCase):

    def setUp(self):
        self.param = Parameters()

        self.param.general.imt_link = "TN-FULLDUPLEX"
        self.param.general.enable_cochannel = True
        self.param.general.enable_adjacent_channel = False
        self.param.general.overwrite_output = True
        self.param.general.seed = "101"
        self.param.general.results_format = "CDF"
        self.param.general.save_snapshot = 10
        self.param.general.suppress_large_results = False

        self.param.imt.topology = "SINGLE_BS"
        self.param.imt.wrap_around = False
        self.param.imt.num_macrocell_sites = 19
        self.param.imt.num_clusters = 2
        self.param.imt.intersite_distance = 150
        self.param.imt.minimum_separation_distance_bs_ue = 10
        self.param.imt.interfered_with = False
        self.param.imt.frequency = 10000
        self.param.imt.bandwidth = 100
        self.param.imt.rb_bandwidth = 0.180
        self.param.imt.spectral_mask = "ITU 265-E"
        self.param.imt.guard_band_ratio = 0.1
        self.param.imt.ho_margin = 3
        self.param.imt.bs_load_probability = 1
        self.param.imt.dl_load_imbalance = 1
        self.param.imt.ul_load_imbalance = 1
        self.param.imt.num_resource_blocks = 10
        self.param.imt.bs_conducted_power = 10
        self.param.imt.bs_height = 6
        self.param.imt.bs_aclr = 40
        self.param.imt.bs_acs = 30
        self.param.imt.bs_noise_figure = 7
        self.param.imt.bs_noise_temperature = 290
        self.param.imt.bs_ohmic_loss = 3
        self.param.imt.bs_sic = 100
        self.param.imt.ul_attenuation_factor = 0.4
        self.param.imt.ul_sinr_min = -10
        self.param.imt.ul_sinr_max = 22
        self.param.imt.ue_k = 1
        self.param.imt.ue_k_m = 2
        self.param.imt.ue_indoor_percent = 0
        self.param.imt.ue_distribution_distance = "RAYLEIGH"
        self.param.imt.ue_distribution_azimuth = "UNIFORM"
        self.param.imt.ue_distribution_type = "ANGLE_AND_DISTANCE"
        self.param.imt.ue_tx_power_control = "OFF"
        self.param.imt.ue_p_o_pusch = -95
        self.param.imt.ue_alfa = 0.8
        self.param.imt.ue_p_cmax = 20
        self.param.imt.ue_conducted_power = 10
        self.param.imt.ue_height = 1.5
        self.param.imt.ue_aclr = 35
        self.param.imt.ue_acs = 25
        self.param.imt.ue_noise_figure = 9
        self.param.imt.ue_ohmic_loss = 3
        self.param.imt.ue_body_loss = 4
        self.param.imt.ue_sic = 100
        self.param.imt.dl_attenuation_factor = 0.6
        self.param.imt.dl_sinr_min = -10
        self.param.imt.dl_sinr_max = 30
        self.param.imt.channel_model = "FSPL"
        self.param.imt.bs_bs_channel_model = "FSPL"
        self.param.imt.ue_ue_channel_model = "FSPL"
        self.param.imt.line_of_sight_prob = 0.75  # probability of line-of-sight (not for FSPL)
        self.param.imt.shadowing = False
        self.param.imt.noise_temperature = 290
        self.param.imt.BOLTZMANN_CONSTANT = 1.38064852e-23

        self.param.antenna_imt.bs_antenna_type = "BEAMFORMING"
        self.param.antenna_imt.normalization = False
        self.param.antenna_imt.bs_element_pattern = "M2101"
        self.param.antenna_imt.bs_normalization_file = None
        self.param.antenna_imt.bs_tx_element_max_g = 10
        self.param.antenna_imt.bs_tx_element_phi_deg_3db = 80
        self.param.antenna_imt.bs_tx_element_theta_deg_3db = 80
        self.param.antenna_imt.bs_tx_element_am = 25
        self.param.antenna_imt.bs_tx_element_sla_v = 25
        self.param.antenna_imt.bs_tx_n_rows = 16
        self.param.antenna_imt.bs_tx_n_columns = 16
        self.param.antenna_imt.bs_tx_element_horiz_spacing = 1
        self.param.antenna_imt.bs_tx_element_vert_spacing = 1
        self.param.antenna_imt.bs_rx_element_max_g = 5
        self.param.antenna_imt.bs_rx_element_phi_deg_3db = 65
        self.param.antenna_imt.bs_rx_element_theta_deg_3db = 65
        self.param.antenna_imt.bs_rx_element_am = 30
        self.param.antenna_imt.bs_rx_element_sla_v = 30
        self.param.antenna_imt.bs_rx_n_rows = 2
        self.param.antenna_imt.bs_rx_n_columns = 2
        self.param.antenna_imt.bs_downtilt_deg = 10
        self.param.antenna_imt.bs_rx_element_horiz_spacing = 0.5
        self.param.antenna_imt.bs_rx_element_vert_spacing = 0.5
        self.param.antenna_imt.bs_element_pattern = "M2101"
        self.param.antenna_imt.ue_antenna_type = "BEAMFORMING"
        self.param.antenna_imt.ue_tx_element_max_g = 5
        self.param.antenna_imt.ue_tx_element_phi_deg_3db = 65
        self.param.antenna_imt.ue_tx_element_theta_deg_3db = 65
        self.param.antenna_imt.ue_tx_element_am = 30
        self.param.antenna_imt.ue_tx_element_sla_v = 30
        self.param.antenna_imt.ue_tx_n_rows = 2
        self.param.antenna_imt.ue_tx_n_columns = 1
        self.param.antenna_imt.ue_tx_element_horiz_spacing = 0.5
        self.param.antenna_imt.ue_tx_element_vert_spacing = 0.5
        self.param.antenna_imt.ue_rx_element_max_g = 10
        self.param.antenna_imt.ue_rx_element_phi_deg_3db = 90
        self.param.antenna_imt.ue_rx_element_theta_deg_3db = 90
        self.param.antenna_imt.ue_rx_element_am = 25
        self.param.antenna_imt.ue_rx_element_sla_v = 25
        self.param.antenna_imt.ue_rx_n_rows = 16
        self.param.antenna_imt.ue_rx_n_columns = 16
        self.param.antenna_imt.ue_rx_element_horiz_spacing = 1
        self.param.antenna_imt.ue_rx_element_vert_spacing = 1
        self.param.antenna_imt.ue_element_pattern = "M2101"

        self.param.fss_ss.frequency = 10000
        self.param.fss_ss.bandwidth = 100
        self.param.fss_ss.altitude = 35786000
        self.param.fss_ss.lat_deg = 0
        self.param.fss_ss.azimuth = 0
        self.param.fss_ss.elevation = 270
        self.param.fss_ss.tx_power_density = -30
        self.param.fss_ss.noise_temperature = 950
        self.param.fss_ss.antenna_gain = 51
        self.param.fss_ss.antenna_pattern = "OMNI"
        self.param.fss_ss.imt_altitude = 1000
        self.param.fss_ss.imt_lat_deg = -23.5629739
        self.param.fss_ss.imt_long_diff_deg = (-46.6555132 - 75)
        self.param.fss_ss.channel_model = "FSPL"
        self.param.fss_ss.line_of_sight_prob = 0.01
        self.param.fss_ss.surf_water_vapour_density = 7.5
        self.param.fss_ss.specific_gaseous_att = 0.1
        self.param.fss_ss.time_ratio = 0.5
        self.param.fss_ss.antenna_l_s = -20
        self.param.fss_ss.BOLTZMANN_CONSTANT = 1.38064852e-23
        self.param.fss_ss.EARTH_RADIUS = 6371000

        self.param.fss_es.location = "FIXED"
        self.param.fss_es.x = -5000
        self.param.fss_es.y = 0
        self.param.fss_es.height = 10
        self.param.fss_es.elevation_min = 20
        self.param.fss_es.elevation_max = 20
        self.param.fss_es.azimuth = "0"
        self.param.fss_es.frequency = 10000
        self.param.fss_es.bandwidth = 100
        self.param.fss_es.noise_temperature = 100
        self.param.fss_es.tx_power_density = -60
        self.param.fss_es.antenna_gain = 50
        self.param.fss_es.antenna_pattern = "OMNI"
        self.param.fss_es.channel_model = "FSPL"
        self.param.fss_es.line_of_sight_prob = 1
        self.param.fss_es.BOLTZMANN_CONSTANT = 1.38064852e-23
        self.param.fss_es.EARTH_RADIUS = 6371000

    def test_simulation_2bs_4ue_fss_ss(self):
        self.param.general.system = "FSS_SS"

        self.simulation = SimulationTNFullDuplex(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState(1)

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.bs.active = np.ones(2, dtype=bool)

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([20, 70, 110, 170])
        self.simulation.ue.y = np.array([0, 0, 0, 0])
        self.simulation.ue.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.ue.active = np.zeros(4, dtype=bool)

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.assertEqual(self.simulation.link, {0: [0, 1], 1: [2, 3]})

        # test selection method
        self.simulation.select_ue(random_number_gen)
        npt.assert_equal(self.simulation.link, {0: [0, 1], 1: [2, 3]})
        npt.assert_equal(self.simulation.link_dl, {0: [0], 1: [2]})
        npt.assert_equal(self.simulation.link_ul, {0: [1], 1: [3]})
        npt.assert_equal(self.simulation.ue.active,
                         np.array([True, True, True, True]))
        npt.assert_equal(self.simulation.ue_beam_rbs, np.array([0, 0, 0, 0]))
        npt.assert_equal(self.simulation.bs_to_ue_beam_idx, np.array([0, 1, 0, 1]))
        npt.assert_equal(self.simulation.bs_beam_rbs, {0: [('DL', 0), ('UL', 0)], 1: [('DL', 0), ('UL', 0)]})

        # Test gains
        bs_ue_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.ue)
        npt.assert_equal(bs_ue_gain, np.array([[1, 2, 1, 2],
                                               [1, 2, 1, 2]]))

        ue_bs_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.bs)
        npt.assert_equal(ue_bs_gain, np.array([[1, 1],
                                               [1, 1],
                                               [1, 1],
                                               [1, 1]]))

        ue_ue_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.ue)
        npt.assert_equal(ue_ue_gain, np.array([[1, 1, 1, 1],
                                               [1, 1, 1, 1],
                                               [1, 1, 1, 1],
                                               [1, 1, 1, 1]]))

        bs_bs_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.bs)
        npt.assert_equal(bs_bs_gain, np.array([[1, 2, 1, 2],
                                               [1, 2, 1, 2]]))

        # Create propagation
        self.simulation.propagation_imt = PropagationFactory.create_propagation(self.param.imt.channel_model,
                                                                                self.param, random_number_gen)
        self.simulation.propagation_imt_bs_bs = PropagationFactory.create_propagation(
            self.param.imt.bs_bs_channel_model,
            self.param, random_number_gen)
        self.simulation.propagation_imt_ue_ue = PropagationFactory.create_propagation(
            self.param.imt.ue_ue_channel_model,
            self.param, random_number_gen)
        self.simulation.propagation_system = PropagationFactory.create_propagation(self.param.fss_ss.channel_model,
                                                                                   self.param, random_number_gen)

        # test coupling loss method
        self.simulation.coupling_loss_imt = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                        self.simulation.ue,
                                                                                        self.simulation.propagation_imt)
        expected_coupling_loss_imt = np.array([[78.47 - 1 - 1, 89.35 - 2 - 1, 93.27 - 1 - 1, 97.05 - 2 - 1],
                                               [97.55 - 1 - 1, 94.72 - 2 - 1, 91.53 - 1 - 1, 81.99 - 2 - 1]])
        npt.assert_allclose(self.simulation.coupling_loss_imt, expected_coupling_loss_imt, atol=1e-2)

        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt_ue_ue)
        expected_coupling_loss_ue_ue = np.array([[np.nan, 86.43 - 1 - 1, 91.53 - 1 - 1, 95.97 - 1 - 1],
                                                 [86.43 - 1 - 1, np.nan, 84.49 - 1 - 1, 92.46 - 1 - 1],
                                                 [91.53 - 1 - 1, 84.49 - 1 - 1, np.nan, 88.01 - 1 - 1],
                                                 [95.97 - 1 - 1, 92.46 - 1 - 1, 88.01 - 1 - 1, np.nan]])
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue, expected_coupling_loss_ue_ue, atol=1e-2)

        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt_bs_bs)
        expected_coupling_loss_bs_bs = np.array([[np.nan, np.nan, 98.47 - 1 - 2, 98.47 - 2 - 1],
                                                 [98.47 - 1 - 2, 98.47 - 2 - 1, np.nan, np.nan]])
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs, expected_coupling_loss_bs_bs, atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 1)
        npt.assert_allclose(self.simulation.ue.bandwidth, bandwidth_per_ue * np.ones(4), atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx_bs = 10 + 0 - 10 * math.log10(1)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx_bs]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx_bs]), atol=1e-2)
        p_tx_ue = 20
        npt.assert_allclose(self.simulation.ue.tx_power, p_tx_ue * np.ones(4))

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        # check UE received power
        expected_ue_rx_power = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                               self.param.imt.ue_body_loss - expected_coupling_loss_imt[[0, 1], [0, 2]]
        npt.assert_allclose(self.simulation.ue.rx_power[[0, 2]],
                            expected_ue_rx_power,
                            atol=1e-2)

        # check UE received interference
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[[1, 0], [0, 2]]
        interference_ue_same_cell = p_tx_ue - 2*self.param.imt.ue_ohmic_loss - 2*self.param.imt.ue_body_loss - \
                                    expected_coupling_loss_ue_ue[[1, 3], [0, 2]]
        interference_ue_other_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                     expected_coupling_loss_ue_ue[[3, 1], [0, 2]]
        expected_ue_rx_interference = 10*np.log10(np.power(10, 0.1*interference_bs) +
                                                  np.power(10, 0.1 * interference_ue_same_cell) +
                                                  np.power(10, 0.1*interference_ue_other_cell))
        npt.assert_allclose(self.simulation.ue.rx_interference[[0, 2]], expected_ue_rx_interference, atol=1e-2)

        # check UE thermal noise
        expected_ue_thermal_noise = 10*np.log10(self.param.imt.noise_temperature*self.param.imt.BOLTZMANN_CONSTANT*1e3) + \
                                 10*np.log10(bandwidth_per_ue * np.ones(4) * 1e6) + self.param.imt.ue_noise_figure
        npt.assert_allclose(self.simulation.ue.thermal_noise, expected_ue_thermal_noise, atol=1e-2)

        # check self-interference
        npt.assert_allclose(self.simulation.ue.self_interference[[0, 2]], -np.inf*np.ones(2), atol=1e-2)

        # check UE thermal noise + interference + self interference
        expected_total_interference = 10*np.log10(np.power(10, 0.1*expected_ue_rx_interference) +
                                                  np.power(10, 0.1*expected_ue_thermal_noise[[0, 2]]))
        npt.assert_allclose(self.simulation.ue.total_interference[[0, 2]], expected_total_interference, atol=1e-2)

        # check SNR
        expected_ue_snr = expected_ue_rx_power - expected_ue_thermal_noise[[0, 2]]
        npt.assert_allclose(self.simulation.ue.snr[[0, 2]], expected_ue_snr, atol=1e-2)

        # check SINR
        expected_ue_sinr = expected_ue_rx_power - expected_total_interference
        npt.assert_allclose(self.simulation.ue.sinr[[0, 2]], expected_ue_sinr, atol=5e-2)

        # check BS received power
        expected_bs_0_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[0,1]
        npt.assert_allclose(self.simulation.bs.rx_power[0], np.array([expected_bs_0_rx_power]), atol=1e-2)
        expected_bs_1_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[1, 3]
        npt.assert_allclose(self.simulation.bs.rx_power[1], np.array([expected_bs_1_rx_power]), atol=1e-2)

        # check BS received interference
        interference_bs = p_tx_bs - 2*self.param.imt.bs_ohmic_loss - expected_coupling_loss_bs_bs[0, 3]
        interference_ue = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss \
                          - self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[0,3]
        expected_bs_0_rx_interference = 10*np.log10(np.power(10, 0.1*interference_ue) +
                                                    np.power(10, 0.1*interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[0], expected_bs_0_rx_interference, atol=1e-2)

        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_bs_bs[1, 1]
        interference_ue = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss \
                          - self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[1, 1]
        expected_bs_1_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[1], expected_bs_1_rx_interference, atol=1e-2)

        # check BS thermal noise
        expected_bs_thermal_noise = 10 * np.log10(self.param.imt.bs_noise_temperature *
                                                  self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                    10 * np.log10(bandwidth_per_ue * np.ones(2) * 1e6) + self.param.imt.bs_noise_figure
        npt.assert_allclose(self.simulation.bs.thermal_noise, expected_bs_thermal_noise, atol=1e-2)

        # check self-interference
        expected_bs_self_interference = p_tx_bs - self.param.imt.bs_sic
        npt.assert_allclose(self.simulation.bs.self_interference[0], expected_bs_self_interference, atol=1e-2)
        npt.assert_allclose(self.simulation.bs.self_interference[1], expected_bs_self_interference, atol=1e-2)

        # check BS thermal noise + interference
        expected_bs_0_total_interference = 10*np.log10(np.power(10, 0.1 * expected_bs_0_rx_interference) +
                                                       np.power(10, 0.1 * expected_bs_thermal_noise[0]) +
                                                       np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[0], expected_bs_0_total_interference, atol=1e-2)

        expected_bs_1_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_1_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise[1]) +
                                                         np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[1], expected_bs_1_total_interference, atol=1e-2)

        # check SNR
        expected_bs_0_snr = expected_bs_0_rx_power - expected_bs_thermal_noise[0]
        npt.assert_allclose(self.simulation.bs.snr[0], expected_bs_0_snr,  atol=1e-2)
        expected_bs_1_snr = expected_bs_1_rx_power - expected_bs_thermal_noise[1]
        npt.assert_allclose(self.simulation.bs.snr[1], expected_bs_1_snr, atol=1e-2)

        # check SINR
        expected_bs_0_sinr = expected_bs_0_rx_power - expected_bs_0_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[0], expected_bs_0_sinr, atol=1e-2)
        expected_bs_1_sinr = expected_bs_1_rx_power - expected_bs_1_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[1], expected_bs_1_sinr, atol=1e-2)

        # Create system
        self.simulation.system = StationFactory.generate_fss_space_station(self.param.fss_ss)
        self.simulation.system.x = np.array([0])
        self.simulation.system.y = np.array([0])
        self.simulation.system.height = np.array([self.param.fss_ss.altitude])

        # test the method that calculates interference from IMT UE to FSS space station
        self.simulation.calculate_external_interference()

        # check coupling loss
        expected_cl_bs_sys = np.array([203.52 - 51 - 1, 203.52 - 51 - 1])
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_system, expected_cl_bs_sys, atol=1e-2)
        expected_cl_ue_sys = np.array([203.52-51-1, 203.52-51-1, 203.52-51-1, 203.52-51-1])
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_system, expected_cl_ue_sys, atol=1e-2)

        # check interference generated by IMT to FSS space station
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - expected_cl_bs_sys
        interference_ue = p_tx_ue - self.param.imt.ue_body_loss - self.param.imt.ue_ohmic_loss - \
                          expected_cl_ue_sys[[1,3]]
        expected_rx_interference = 10 * math.log10(np.sum(np.power(10, 0.1 * interference_bs)) + \
                                          np.sum(np.power(10, 0.1 * interference_ue)))
        self.assertAlmostEqual(self.simulation.system.rx_interference, expected_rx_interference, delta=1e-2)

        # check FSS space station thermal noise
        thermal_noise = 10 * np.log10(self.param.fss_ss.BOLTZMANN_CONSTANT * self.param.fss_ss.noise_temperature * 1e3
                                      * self.param.fss_ss.bandwidth * 1e6)
        self.assertAlmostEqual(self.simulation.system.thermal_noise, thermal_noise, delta=1e-2)

        # check INR at FSS space station
        expected_inr = expected_rx_interference - thermal_noise
        self.assertAlmostEqual(self.simulation.system.inr, expected_inr, delta=1e-2)

    def test_simulation_2bs_4ue_fss_ss_imbalance(self):
        self.param.imt.dl_load_imbalance = 2.0
        self.param.imt.ul_load_imbalance = 1.0 / self.param.imt.dl_load_imbalance

        self.param.general.system = "FSS_SS"

        self.simulation = SimulationTNFullDuplex(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState(1)

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.bs.active = np.ones(2, dtype=bool)

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([20, 70, 110, 170])
        self.simulation.ue.y = np.array([0, 0, 0, 0])
        self.simulation.ue.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.ue.active = np.zeros(4, dtype=bool)

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.assertEqual(self.simulation.link, {0: [0, 1], 1: [2, 3]})

        # Test selection method
        self.simulation.select_ue(random_number_gen)
        self.assertEqual(self.simulation.link_dl, {0: [0], 1: [2]})
        npt.assert_equal(self.simulation.link_ul, {0: [1], 1: []})
        self.assertEqual(self.simulation.link, {0: [0, 1], 1: [2]})
        npt.assert_equal(self.simulation.ue.active,
                         np.array([True, True, True, False]))
        npt.assert_equal(self.simulation.ue_beam_rbs, np.array([0, 0, 0, -1]))
        npt.assert_equal(self.simulation.bs_to_ue_beam_idx, np.array([0, 1, 0, -1]))
        npt.assert_equal(self.simulation.bs_beam_rbs, {0: [('DL', 0), ('UL', 0)], 1: [('DL', 0)]})

        # Test gains
        bs_ue_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.ue)
        npt.assert_equal(bs_ue_gain, np.array([[1, 2, 1, 0],
                                               [1, 0, 1, 0]]))

        ue_bs_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.bs)
        npt.assert_equal(ue_bs_gain, np.array([[1, 1],
                                               [1, 1],
                                               [1, 1],
                                               [0, 0]]))

        ue_ue_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.ue)
        npt.assert_equal(ue_ue_gain, np.array([[1, 1, 1, 0],
                                               [1, 1, 1, 0],
                                               [1, 1, 1, 0],
                                               [0, 0, 0, 0]]))

        bs_bs_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.bs)
        npt.assert_equal(bs_bs_gain, np.array([[1, 2, 1, 2],
                                               [1, 0, 1, 0]]))

        # Create propagation
        self.simulation.propagation_imt = PropagationFactory.create_propagation(self.param.imt.channel_model,
                                                                                self.param, random_number_gen)
        self.simulation.propagation_imt_bs_bs = PropagationFactory.create_propagation(
            self.param.imt.bs_bs_channel_model,
            self.param, random_number_gen)
        self.simulation.propagation_imt_ue_ue = PropagationFactory.create_propagation(
            self.param.imt.ue_ue_channel_model,
            self.param, random_number_gen)
        self.simulation.propagation_system = PropagationFactory.create_propagation(self.param.fss_ss.channel_model,
                                                                                   self.param, random_number_gen)

        # test coupling loss method
        self.simulation.coupling_loss_imt = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                        self.simulation.ue,
                                                                                        self.simulation.propagation_imt)
        expected_coupling_loss_imt = np.array([[78.47 - 1 - 1, 89.35 - 2 - 1, 93.27 - 1 - 1, 97.05 - 0 - 0],
                                               [97.55 - 1 - 1, 94.72 - 0 - 1, 91.53 - 1 - 1, 81.99 - 0 - 0]])
        npt.assert_allclose(self.simulation.coupling_loss_imt, expected_coupling_loss_imt, atol=1e-2)

        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt_ue_ue)
        expected_coupling_loss_ue_ue = np.array([[np.nan, 86.43 - 1 - 1, 91.53 - 1 - 1, 95.97 - 0 - 0],
                                                 [86.43 - 1 - 1, np.nan, 84.49 - 1 - 1, 92.46 - 0 - 0],
                                                 [91.53 - 1 - 1, 84.49 - 1 - 1, np.nan, 88.01 - 0 - 0],
                                                 [95.97 - 0 - 0, 92.46 - 0 - 0, 88.01 - 0 - 0, np.nan]])
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue, expected_coupling_loss_ue_ue, atol=1e-2)

        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt_bs_bs)
        expected_coupling_loss_bs_bs = np.array([[np.nan, np.nan, 98.47 - 1 - 0, 98.47 - 2 - 1],
                                                 [98.47 - 1 - 2, 98.47 - 0 - 0, np.nan, np.nan]])
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs, expected_coupling_loss_bs_bs, atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 1)
        npt.assert_allclose(self.simulation.ue.bandwidth[0:3], bandwidth_per_ue * np.ones(3), atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx_bs = 10 + 0 - 10 * math.log10(1)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx_bs]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx_bs]), atol=1e-2)
        p_tx_ue = 20
        npt.assert_allclose(self.simulation.ue.tx_power[0:3], p_tx_ue * np.ones(3))

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        ## check UE received power
        expected_ue_rx_power = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                               self.param.imt.ue_body_loss - expected_coupling_loss_imt[[0, 1], [0, 2]]
        npt.assert_allclose(self.simulation.ue.rx_power[[0, 2]],
                            expected_ue_rx_power,
                            atol=1e-2)

        # check UE received interference
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[[1, 0], [0, 2]]
        interference_ue_same_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                    expected_coupling_loss_ue_ue[[1, 3], [0, 2]]
        interference_ue_same_cell[1] = -np.inf
        interference_ue_other_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                     expected_coupling_loss_ue_ue[[3, 1], [0, 2]]
        interference_ue_other_cell[0] = -np.inf
        expected_ue_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_bs) +
                                                    np.power(10, 0.1 * interference_ue_same_cell) +
                                                    np.power(10, 0.1 * interference_ue_other_cell))
        npt.assert_allclose(self.simulation.ue.rx_interference[[0, 2]], expected_ue_rx_interference, atol=1e-2)

        # check UE thermal noise
        expected_ue_thermal_noise = 10 * np.log10(
            self.param.imt.noise_temperature * self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                 10 * np.log10(bandwidth_per_ue * np.ones(3) * 1e6) + self.param.imt.ue_noise_figure
        npt.assert_allclose(self.simulation.ue.thermal_noise[0:3], expected_ue_thermal_noise, atol=1e-2)

        # check self-interference
        npt.assert_allclose(self.simulation.ue.self_interference[[0, 2]], -np.inf * np.ones(2), atol=1e-2)

        # check UE thermal noise + interference + self interference
        expected_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_ue_rx_interference) +
                                                    np.power(10, 0.1 * expected_ue_thermal_noise[[0, 2]]))
        npt.assert_allclose(self.simulation.ue.total_interference[[0, 2]], expected_total_interference, atol=1e-2)

        # check SNR
        expected_ue_snr = expected_ue_rx_power - expected_ue_thermal_noise[[0, 2]]
        npt.assert_allclose(self.simulation.ue.snr[[0, 2]], expected_ue_snr, atol=1e-2)

        # check SINR
        expected_ue_sinr = expected_ue_rx_power - expected_total_interference
        npt.assert_allclose(self.simulation.ue.sinr[[0, 2]], expected_ue_sinr, atol=5e-2)

        # check BS received power
        expected_bs_0_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[0, 1]
        npt.assert_allclose(self.simulation.bs.rx_power[0], np.array([expected_bs_0_rx_power]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.rx_power[1], np.array([]), atol=1e-2)

        # check BS received interference
        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_bs_bs[0, 3]
        interference_ue = -np.inf
        expected_bs_0_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[0], expected_bs_0_rx_interference, atol=1e-2)
        expected_bs_1_rx_interference = -500.00
        npt.assert_allclose(self.simulation.bs.rx_interference[1], expected_bs_1_rx_interference, atol=1e-2)

        # check BS thermal noise
        expected_bs_thermal_noise = 10 * np.log10(self.param.imt.bs_noise_temperature *
                                                  self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                    10 * np.log10(bandwidth_per_ue * np.ones(2) * 1e6) + self.param.imt.bs_noise_figure
        npt.assert_allclose(self.simulation.bs.thermal_noise, expected_bs_thermal_noise, atol=1e-2)

        # check self-interference
        expected_bs_0_self_interference = p_tx_bs - self.param.imt.bs_sic
        npt.assert_allclose(self.simulation.bs.self_interference[0], expected_bs_0_self_interference, atol=1e-2)
        expected_bs_1_self_interference = -500.00
        npt.assert_allclose(self.simulation.bs.self_interference[1], expected_bs_1_self_interference, atol=1e-2)

        # check BS thermal noise + interference
        expected_bs_0_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_0_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise[0]) +
                                                         np.power(10, 0.1 * expected_bs_0_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[0], expected_bs_0_total_interference, atol=1e-2)

        expected_bs_1_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_1_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise[1]) +
                                                         np.power(10, 0.1 * expected_bs_1_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[1], expected_bs_1_total_interference, atol=1e-2)

        # check SNR
        expected_bs_0_snr = expected_bs_0_rx_power - expected_bs_thermal_noise[0]
        npt.assert_allclose(self.simulation.bs.snr[0], expected_bs_0_snr, atol=1e-2)
        npt.assert_allclose(self.simulation.bs.snr[1], [], atol=1e-2)

        # check SINR
        expected_bs_0_sinr = expected_bs_0_rx_power - expected_bs_0_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[0], expected_bs_0_sinr, atol=1e-2)
        npt.assert_allclose(self.simulation.bs.sinr[1], [], atol=1e-2)

        # Create system
        self.simulation.system = StationFactory.generate_fss_space_station(self.param.fss_ss)
        self.simulation.system.x = np.array([0])
        self.simulation.system.y = np.array([0])
        self.simulation.system.height = np.array([self.param.fss_ss.altitude])

        # test the method that calculates interference from IMT UE to FSS space station
        self.simulation.calculate_external_interference()

        # check coupling loss
        expected_cl_bs_sys = np.array([203.52 - 51 - 1, 203.52 - 51 - 1])
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_system, expected_cl_bs_sys, atol=1e-2)
        expected_cl_ue_sys = np.array([203.52 - 51 - 1, 203.52 - 51 - 1, 203.52 - 51 - 1])
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_system[0:3], expected_cl_ue_sys, atol=1e-2)

        # check interference generated by IMT to FSS space station
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - expected_cl_bs_sys
        interference_ue = p_tx_ue - self.param.imt.ue_body_loss - self.param.imt.ue_ohmic_loss - \
                          expected_cl_ue_sys[1]
        expected_rx_interference = 10 * math.log10(np.sum(np.power(10, 0.1 * interference_bs)) + \
                                                   np.sum(np.power(10, 0.1 * interference_ue)))
        self.assertAlmostEqual(self.simulation.system.rx_interference, expected_rx_interference, delta=1e-2)

        # check FSS space station thermal noise
        thermal_noise = 10 * np.log10(self.param.fss_ss.BOLTZMANN_CONSTANT * self.param.fss_ss.noise_temperature * 1e3
                                      * self.param.fss_ss.bandwidth * 1e6)
        self.assertAlmostEqual(self.simulation.system.thermal_noise, thermal_noise, delta=1e-2)

        # check INR at FSS space station
        expected_inr = expected_rx_interference - thermal_noise
        self.assertAlmostEqual(self.simulation.system.inr, expected_inr, delta=1e-2)

    def test_simulation_2bs_8ue_fss_es_balance(self):
        self.param.imt.ue_k = 2
        self.param.imt.ue_k_m = 2

        self.param.imt.dl_load_imbalance = 1.0
        self.param.imt.ul_load_imbalance = 1.0 / self.param.imt.dl_load_imbalance

        self.param.general.system = "FSS_ES"

        self.simulation = SimulationTNFullDuplex(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState(1326)
        #
        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.bs.active = np.ones(2, dtype=bool)
        #
        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([10, 20, 70, 100, 110, 120, 170, 190])
        self.simulation.ue.y = np.zeros(8)
        self.simulation.ue.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam()])

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.assertEqual(self.simulation.link, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})

        # # Test selection method
        self.simulation.select_ue(random_number_gen)
        self.assertEqual(self.simulation.link_dl, {0: [0, 1], 1: [5, 7]})
        npt.assert_equal(self.simulation.link_ul, {0: [2, 3], 1: [6, 4]})
        self.assertEqual(self.simulation.link, {0: [0, 1, 2, 3], 1: [5, 7, 6, 4]})
        npt.assert_equal(self.simulation.ue.active,
                         np.array([True, True, True, True, True, True, True, True]))
        npt.assert_equal(self.simulation.ue_beam_rbs, np.array([0., 1., 0., 1., 1., 0., 0., 1.]))
        npt.assert_equal(self.simulation.bs_to_ue_beam_idx, np.array([0, 1, 2, 3, 3, 0, 2, 1]))
        npt.assert_equal(self.simulation.bs_beam_rbs, {0: [('DL', 0), ('DL', 1), ('UL', 0), ('UL', 1)],
                                                       1: [('DL', 0), ('DL', 1), ('UL', 0), ('UL', 1)]})

        # Test gains
        expected_bs_ue_gain = np.array([[1., 2., 3., 4., 4., 1., 3., 2.],
                                        [1., 2., 3., 4., 4., 1., 3., 2.]])
        bs_ue_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.ue)
        npt.assert_equal(bs_ue_gain, expected_bs_ue_gain)

        expected_ue_bs_gain = np.ones((8, 2))
        ue_bs_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.bs)
        npt.assert_equal(ue_bs_gain, expected_ue_bs_gain)

        expected_ue_ue_gain = np.ones((8, 8))
        ue_ue_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.ue)
        npt.assert_equal(ue_ue_gain, expected_ue_ue_gain)

        expected_bs_bs_gain = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                                        [1, 2, 3, 4, 1, 2, 3, 4]])
        bs_bs_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.bs)
        npt.assert_equal(bs_bs_gain, expected_bs_bs_gain)

        # Create propagation
        self.simulation.propagation_imt = PropagationForTest(random_number_gen)
        self.simulation.propagation_imt_bs_bs = PropagationForTest(random_number_gen)
        self.simulation.propagation_imt_ue_ue = PropagationForTest(random_number_gen)
        self.simulation.propagation_system = PropagationForTest(random_number_gen)
        #
        # # test coupling loss method
        expected_path_loss_imt = np.array([[10, 20, 70, 100, 110, 120, 170, 190],
                                           [190, 180, 130, 100, 90, 80, 30, 10]])
        expected_coupling_loss_imt = expected_path_loss_imt - expected_bs_ue_gain - np.transpose(expected_ue_bs_gain)
        self.simulation.coupling_loss_imt = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                        self.simulation.ue,
                                                                                        self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt, expected_coupling_loss_imt, atol=1e-2)

        expected_path_loss_imt_ue_ue = np.array([[np.nan, 10, 60, 90, 100, 110, 160, 180],
                                                 [10, np.nan, 50, 80, 90, 100, 150, 170],
                                                 [60, 50, np.nan, 30, 40, 50, 100, 120],
                                                 [90, 80, 30, np.nan, 10, 20, 70, 90],
                                                 [100, 90, 40, 10, np.nan, 10, 60, 80],
                                                 [110, 100, 50, 20, 10, np.nan, 50, 70],
                                                 [160, 150, 100, 70, 60, 50, np.nan, 20],
                                                 [180, 170, 120, 90, 80, 70, 20, np.nan]])
        expected_coupling_loss_imt_ue_ue = expected_path_loss_imt_ue_ue - expected_ue_ue_gain - \
                                           np.transpose(expected_ue_ue_gain)
        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue, expected_coupling_loss_imt_ue_ue, atol=1e-2)

        expected_bs_bs_path_loss = np.array([[np.nan, np.nan, np.nan, np.nan, 200, 200, 200, 200],
                                             [200, 200, 200, 200, np.nan, np.nan, np.nan, np.nan]])
        expected_coupling_loss_imt_bs_bs = expected_bs_bs_path_loss - expected_bs_bs_gain  - \
                                           np.array([expected_bs_bs_gain[1, [0, 1, 2, 3, 2, 3, 0, 1]],
                                                     expected_bs_bs_gain[0, [2, 3, 0, 1, 0, 1, 2, 3]]])
        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs, expected_coupling_loss_imt_bs_bs, atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 2)
        npt.assert_allclose(self.simulation.ue.bandwidth, bandwidth_per_ue, atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx_bs = 10 + 0 - 10 * math.log10(2)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx_bs, p_tx_bs]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx_bs, p_tx_bs]), atol=1e-2)
        p_tx_ue = 20
        npt.assert_allclose(self.simulation.ue.tx_power, p_tx_ue)

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        ## check UE received power
        expected_ue_rx_power = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                               self.param.imt.ue_body_loss - expected_coupling_loss_imt[[0, 0, 1, 1], [0, 1, 5, 7]]
        npt.assert_allclose(self.simulation.ue.rx_power[[0, 1, 5, 7]],
                            expected_ue_rx_power,
                            atol=1e-2)

        # check UE received interference
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[[1, 1, 0, 0], [0, 1, 5, 7]]
        interference_ue_same_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                    expected_coupling_loss_imt_ue_ue[[0, 1, 5, 7], [2, 3, 6, 4]]
        interference_ue_other_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                     expected_coupling_loss_imt_ue_ue[[0, 1, 5, 7], [6, 4, 2, 3]]
        expected_ue_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_bs) +
                                                    np.power(10, 0.1 * interference_ue_same_cell) +
                                                    np.power(10, 0.1 * interference_ue_other_cell))
        npt.assert_allclose(self.simulation.ue.rx_interference[[0, 1, 5, 7]], expected_ue_rx_interference, atol=1e-2)

        # check UE thermal noise
        expected_ue_thermal_noise = 10 * np.log10(
            self.param.imt.noise_temperature * self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                 10 * np.log10(bandwidth_per_ue * 1e6) + self.param.imt.ue_noise_figure
        npt.assert_allclose(self.simulation.ue.thermal_noise, expected_ue_thermal_noise, atol=1e-2)

        # check self-interference
        npt.assert_allclose(self.simulation.ue.self_interference, -np.inf, atol=1e-2)

        # check UE thermal noise + interference + self interference
        expected_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_ue_rx_interference) +
                                                    np.power(10, 0.1 * expected_ue_thermal_noise))
        npt.assert_allclose(self.simulation.ue.total_interference[[0, 1, 5, 7]], expected_total_interference, atol=1e-2)

        # check SNR
        expected_ue_snr = expected_ue_rx_power - expected_ue_thermal_noise
        npt.assert_allclose(self.simulation.ue.snr[[0, 1, 5, 7]], expected_ue_snr, atol=1e-2)

        # check SINR
        expected_ue_sinr = expected_ue_rx_power - expected_total_interference
        npt.assert_allclose(self.simulation.ue.sinr[[0, 1, 5, 7]], expected_ue_sinr, atol=5e-2)

        # check BS received power
        expected_bs_0_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[0, [2, 3]]
        npt.assert_allclose(self.simulation.bs.rx_power[0], expected_bs_0_rx_power, atol=1e-2)
        expected_bs_1_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[1, [6, 4]]
        npt.assert_allclose(self.simulation.bs.rx_power[1], expected_bs_1_rx_power, atol=1e-2)

        # check BS received interference
        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt_bs_bs[[1, 1], [0, 1]]
        npt.assert_allclose(self.simulation.bs.interference_from_bs[0], interference_bs, atol=1e-2)
        interference_ue = p_tx_ue - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[0, [6, 4]]
        npt.assert_allclose(self.simulation.bs.interference_from_ue[0], interference_ue, atol=1e-2)
        expected_bs_0_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[0], expected_bs_0_rx_interference, atol=1e-2)

        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt_bs_bs[[0, 0], [4, 5]]
        npt.assert_allclose(self.simulation.bs.interference_from_bs[1], interference_bs, atol=1e-2)
        interference_ue = p_tx_ue - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[1, [2, 3]]
        npt.assert_allclose(self.simulation.bs.interference_from_ue[1], interference_ue, atol=1e-2)
        expected_bs_1_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[1], expected_bs_1_rx_interference, atol=1e-2)

        # check BS thermal noise
        expected_bs_thermal_noise = 10 * np.log10(self.param.imt.bs_noise_temperature *
                                                  self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                    10 * np.log10(bandwidth_per_ue * 1e6) + self.param.imt.bs_noise_figure
        npt.assert_allclose(self.simulation.bs.thermal_noise, expected_bs_thermal_noise, atol=1e-2)

        # check self-interference
        expected_bs_self_interference = p_tx_bs - self.param.imt.bs_sic
        npt.assert_allclose(self.simulation.bs.self_interference[0], expected_bs_self_interference, atol=1e-2)
        npt.assert_allclose(self.simulation.bs.self_interference[1], expected_bs_self_interference, atol=1e-2)

        # check BS thermal noise + interference
        expected_bs_0_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_0_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise) +
                                                         np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[0], expected_bs_0_total_interference, atol=1e-2)

        expected_bs_1_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_1_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise) +
                                                         np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[1], expected_bs_1_total_interference, atol=1e-2)

        # check SNR
        expected_bs_0_snr = expected_bs_0_rx_power - expected_bs_thermal_noise
        npt.assert_allclose(self.simulation.bs.snr[0], expected_bs_0_snr, atol=1e-2)
        expected_bs_1_snr = expected_bs_1_rx_power - expected_bs_thermal_noise
        npt.assert_allclose(self.simulation.bs.snr[1], expected_bs_1_snr, atol=1e-2)

        # check SINR
        expected_bs_0_sinr = expected_bs_0_rx_power - expected_bs_0_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[0], expected_bs_0_sinr, atol=1e-2)
        expected_bs_1_sinr = expected_bs_1_rx_power - expected_bs_1_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[1], expected_bs_1_sinr, atol=1e-2)

        # Create system
        self.simulation.system = StationFactory.generate_fss_earth_station(self.param.fss_es, random_number_gen)
        self.simulation.system.x = np.array([-10])
        self.simulation.system.y = np.array([0])
        self.simulation.system.height = np.array([1.5])

        # test the method that calculates interference from IMT to FSS space station
        self.simulation.calculate_external_interference()

        # check coupling loss
        expected_path_loss_imt_ue_system = np.array([20, 30, 80, 110, 120, 130, 180, 200])
        expected_ue_system_gain = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        expected_coupling_loss_imt_ue_system = expected_path_loss_imt_ue_system - expected_ue_system_gain \
                                               - self.param.fss_es.antenna_gain
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_system, expected_coupling_loss_imt_ue_system,
                            atol=1e-2)

        expected_path_loss_imt_bs_system = np.array([10.9658, 10.9658, 210.0482, 210.0482])
        expected_bs_system_gain = np.array([1, 2, 1, 2])
        expected_coupling_loss_imt_bs_system = expected_path_loss_imt_bs_system - expected_bs_system_gain \
                                               - self.param.fss_es.antenna_gain
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_system, expected_coupling_loss_imt_bs_system,
                            atol=1e-2)

        # check interference generated by IMT to FSS earth station
        interference_bs = p_tx_bs - expected_coupling_loss_imt_bs_system - self.param.imt.bs_ohmic_loss
        interference_ue = p_tx_ue - expected_coupling_loss_imt_ue_system[[2, 3, 4, 6]] - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss
        expected_es_rx_interference = 10 * math.log10(np.sum(np.power(10, 0.1 * interference_bs)) + \
                                                      np.sum(np.power(10, 0.1 * interference_ue)))
        self.assertAlmostEqual(self.simulation.system.rx_interference, expected_es_rx_interference, delta=1e-2)

        # check FSS earth station thermal noise
        expected_es_thermal_noise = 10 * np.log10(self.param.fss_es.BOLTZMANN_CONSTANT *
                                                  self.param.fss_es.noise_temperature * self.param.fss_es.bandwidth *
                                                  1e6 * 1e3)
        self.assertAlmostEqual(self.simulation.system.thermal_noise, expected_es_thermal_noise, delta=1e-2)

        # check INR at FSS space station
        expected_inr = expected_es_rx_interference - expected_es_thermal_noise
        self.assertAlmostEqual(self.simulation.system.inr, expected_inr, delta=1e-2)

    def test_simulation_2bs_8ue_fss_es_imbalance(self):
        self.param.imt.ue_k = 2
        self.param.imt.ue_k_m = 2

        self.param.imt.dl_load_imbalance = 2.0
        self.param.imt.ul_load_imbalance = 1.0 / self.param.imt.dl_load_imbalance

        self.param.general.system = "FSS_ES"

        self.simulation = SimulationTNFullDuplex(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState(133)
        #
        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam()])
        self.simulation.bs.active = np.ones(2, dtype=bool)
        #
        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([10, 20, 70, 100, 110, 120, 170, 190])
        self.simulation.ue.y = np.zeros(8)
        self.simulation.ue.antenna = np.array([AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam(),
                                               AntennaOmniBeam(), AntennaOmniBeam()])

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.assertEqual(self.simulation.link, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})

        # Test selection method
        self.simulation.select_ue(random_number_gen)
        self.assertEqual(self.simulation.link_dl, {0: [3, 0], 1: [5, 6]})
        npt.assert_equal(self.simulation.link_ul, {0: [2], 1: [7, 4]})
        self.assertEqual(self.simulation.link, {0: [3, 0, 2], 1: [5, 6, 7, 4]})
        expected_active_ues = np.array([True, False, True, True, True, True, True, True])
        npt.assert_equal(self.simulation.ue.active, expected_active_ues)
        npt.assert_equal(self.simulation.ue_beam_rbs, np.array([1., -1., 1., 0., 1., 0., 1., 0.]))
        npt.assert_equal(self.simulation.ue_directions, ['DL', '', 'UL', 'DL', 'UL', 'DL', 'DL', 'UL'])
        npt.assert_equal(self.simulation.bs_to_ue_beam_idx, np.array([1, -1, 2, 0, 3, 0, 1, 2]))
        npt.assert_equal(self.simulation.bs_beam_rbs, {0: [('DL', 0), ('DL', 1), ('UL', 1)],
                                                       1: [('DL', 0), ('DL', 1), ('UL', 0), ('UL', 1)]})

        # Test gains
        expected_bs_ue_gain = np.array([[2., 0., 3., 1., 3., 1., 2., 0.],
                                        [2., 0., 4., 1., 4., 1., 2., 3.]])
        bs_ue_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.ue)
        npt.assert_equal(bs_ue_gain, expected_bs_ue_gain)

        expected_ue_bs_gain = np.ones((8, 2))
        expected_ue_bs_gain[1,:] = 0
        ue_bs_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.bs)
        npt.assert_equal(ue_bs_gain, expected_ue_bs_gain)

        expected_ue_ue_gain = np.ones((8, 8))
        expected_ue_ue_gain[1, :] = 0
        expected_ue_ue_gain[:, 1] = 0
        ue_ue_gain = self.simulation.calculate_imt_gains(self.simulation.ue,
                                                         self.simulation.ue)
        npt.assert_equal(ue_ue_gain, expected_ue_ue_gain)

        expected_bs_bs_gain = np.array([[1, 2, 3, 0, 1, 2, 3, 0],
                                        [1, 2, 3, 4, 1, 2, 3, 4]])
        bs_bs_gain = self.simulation.calculate_imt_gains(self.simulation.bs,
                                                         self.simulation.bs)
        npt.assert_equal(bs_bs_gain, expected_bs_bs_gain)

        # Create propagation
        self.simulation.propagation_imt = PropagationForTest(random_number_gen)
        self.simulation.propagation_imt_bs_bs = PropagationForTest(random_number_gen)
        self.simulation.propagation_imt_ue_ue = PropagationForTest(random_number_gen)
        self.simulation.propagation_system = PropagationForTest(random_number_gen)

        # test coupling loss method
        expected_path_loss_imt = np.array([[10, 20, 70, 100, 110, 120, 170, 190],
                                           [190, 180, 130, 100, 90, 80, 30, 10]])
        expected_coupling_loss_imt = expected_path_loss_imt - expected_bs_ue_gain - np.transpose(expected_ue_bs_gain)
        self.simulation.coupling_loss_imt = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                        self.simulation.ue,
                                                                                        self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt, expected_coupling_loss_imt, atol=1e-2)

        expected_path_loss_imt_ue_ue = np.array([[np.nan, 10, 60, 90, 100, 110, 160, 180],
                                                 [10, np.nan, 50, 80, 90, 100, 150, 170],
                                                 [60, 50, np.nan, 30, 40, 50, 100, 120],
                                                 [90, 80, 30, np.nan, 10, 20, 70, 90],
                                                 [100, 90, 40, 10, np.nan, 10, 60, 80],
                                                 [110, 100, 50, 20, 10, np.nan, 50, 70],
                                                 [160, 150, 100, 70, 60, 50, np.nan, 20],
                                                 [180, 170, 120, 90, 80, 70, 20, np.nan]])
        expected_coupling_loss_imt_ue_ue = expected_path_loss_imt_ue_ue - expected_ue_ue_gain - \
                                           np.transpose(expected_ue_ue_gain)
        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue, expected_coupling_loss_imt_ue_ue, atol=1e-2)

        expected_bs_bs_path_loss = np.array([[np.nan, np.nan, np.nan, np.nan, 200, 200, 200, 200],
                                             [200, 200, 200, 200, np.nan, np.nan, np.nan, np.nan]])
        expected_coupling_loss_imt_bs_bs = expected_bs_bs_path_loss - expected_bs_bs_gain  - \
                                           np.array([expected_bs_bs_gain[1, [0, 1, 2, 3, 2, 3, 1, 0]],
                                                     expected_bs_bs_gain[0, [3, 2, 0, 1, 0, 1, 2, 3]]])
        expected_coupling_loss_imt_bs_bs[0, 7] = 200
        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs, expected_coupling_loss_imt_bs_bs, atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 2)
        npt.assert_allclose(self.simulation.ue.bandwidth[expected_active_ues], bandwidth_per_ue, atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx_bs = 10 + 0 - 10 * math.log10(2)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx_bs, p_tx_bs]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx_bs, p_tx_bs]), atol=1e-2)
        p_tx_ue = 20
        npt.assert_allclose(self.simulation.ue.tx_power[expected_active_ues], p_tx_ue)

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        ## check UE received power
        expected_ue_rx_power = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                               self.param.imt.ue_body_loss - expected_coupling_loss_imt[[0, 0, 1, 1], [0, 3, 5, 6]]
        npt.assert_allclose(self.simulation.ue.rx_power[[0, 3, 5, 6]],
                            expected_ue_rx_power,
                            atol=1e-2)

        # check UE received interference
        interference_bs = p_tx_bs - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[[1, 1, 0, 0], [0, 3, 5, 6]]
        npt.assert_allclose(self.simulation.ue.interference_from_bs[[0, 3, 5, 6]], interference_bs, atol=1e-2)
        interference_ue_same_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                    expected_coupling_loss_imt_ue_ue[[0, 3, 5, 6], [2, 1, 7, 4]]
        interference_ue_same_cell[1] = -np.inf
        interference_ue_other_cell = p_tx_ue - 2 * self.param.imt.ue_ohmic_loss - 2 * self.param.imt.ue_body_loss - \
                                     expected_coupling_loss_imt_ue_ue[[0, 3, 5, 6], [4, 7, 1, 2]]
        interference_ue_other_cell[2] = -np.inf
        interference_ue = 10 * np.log10(np.power(10, 0.1 * interference_ue_same_cell) +
                                        np.power(10, 0.1 * interference_ue_other_cell))
        npt.assert_allclose(self.simulation.ue.interference_from_ue[[0, 3, 5, 6]], interference_ue, atol=1e-2)
        expected_ue_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_bs) +
                                                    np.power(10, 0.1 * interference_ue))
        npt.assert_allclose(self.simulation.ue.rx_interference[[0, 3, 5, 6]], expected_ue_rx_interference, atol=1e-2)

        # check UE thermal noise
        expected_ue_thermal_noise = 10 * np.log10(
            self.param.imt.noise_temperature * self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                 10 * np.log10(bandwidth_per_ue * 1e6) + self.param.imt.ue_noise_figure
        npt.assert_allclose(self.simulation.ue.thermal_noise[expected_active_ues], expected_ue_thermal_noise, atol=1e-2)

        # check self-interference
        npt.assert_allclose(self.simulation.ue.self_interference, -np.inf, atol=1e-2)

        # check UE thermal noise + interference + self interference
        expected_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_ue_rx_interference) +
                                                    np.power(10, 0.1 * expected_ue_thermal_noise))
        npt.assert_allclose(self.simulation.ue.total_interference[[0, 3, 5, 6]], expected_total_interference, atol=1e-2)

        # check SNR
        expected_ue_snr = expected_ue_rx_power - expected_ue_thermal_noise
        npt.assert_allclose(self.simulation.ue.snr[[0, 3, 5, 6]], expected_ue_snr, atol=1e-2)

        # check SINR
        expected_ue_sinr = expected_ue_rx_power - expected_total_interference
        npt.assert_allclose(self.simulation.ue.sinr[[0, 3, 5, 6]], expected_ue_sinr, atol=5e-2)

        # check BS received power
        expected_bs_0_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[0, 2]
        npt.assert_allclose(self.simulation.bs.rx_power[0], expected_bs_0_rx_power, atol=1e-2)
        expected_bs_1_rx_power = p_tx_ue - self.param.imt.ue_ohmic_loss - self.param.imt.ue_body_loss - \
                                 self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt[1, [7, 4]]
        npt.assert_allclose(self.simulation.bs.rx_power[1], expected_bs_1_rx_power, atol=1e-2)

        # check BS received interference
        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt_bs_bs[1, 1]
        npt.assert_allclose(self.simulation.bs.interference_from_bs[0], interference_bs, atol=1e-2)
        interference_ue = p_tx_ue - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[0, 4]
        npt.assert_allclose(self.simulation.bs.interference_from_ue[0], interference_ue, atol=1e-2)
        expected_bs_0_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[0], expected_bs_0_rx_interference, atol=1e-2)

        interference_bs = p_tx_bs - 2 * self.param.imt.bs_ohmic_loss - expected_coupling_loss_imt_bs_bs[[0, 0], [4, 5]]
        npt.assert_allclose(self.simulation.bs.interference_from_bs[1], interference_bs, atol=1e-2)
        interference_ue = p_tx_ue - self.param.imt.bs_ohmic_loss - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss - expected_coupling_loss_imt[1, [3, 2]]
        interference_ue[0] = -np.inf
        npt.assert_allclose(self.simulation.bs.interference_from_ue[1], interference_ue, atol=1e-2)
        expected_bs_1_rx_interference = 10 * np.log10(np.power(10, 0.1 * interference_ue) +
                                                      np.power(10, 0.1 * interference_bs))
        npt.assert_allclose(self.simulation.bs.rx_interference[1], expected_bs_1_rx_interference, atol=1e-2)

        # check BS thermal noise
        expected_bs_thermal_noise = 10 * np.log10(self.param.imt.bs_noise_temperature *
                                                  self.param.imt.BOLTZMANN_CONSTANT * 1e3) + \
                                    10 * np.log10(bandwidth_per_ue * 1e6) + self.param.imt.bs_noise_figure
        npt.assert_allclose(self.simulation.bs.thermal_noise, expected_bs_thermal_noise, atol=1e-2)

        # check self-interference
        expected_bs_self_interference = p_tx_bs - self.param.imt.bs_sic
        npt.assert_allclose(self.simulation.bs.self_interference[0], expected_bs_self_interference, atol=1e-2)
        npt.assert_allclose(self.simulation.bs.self_interference[1], expected_bs_self_interference, atol=1e-2)

        # check BS thermal noise + interference
        expected_bs_0_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_0_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise) +
                                                         np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[0], expected_bs_0_total_interference, atol=1e-2)

        expected_bs_1_total_interference = 10 * np.log10(np.power(10, 0.1 * expected_bs_1_rx_interference) +
                                                         np.power(10, 0.1 * expected_bs_thermal_noise) +
                                                         np.power(10, 0.1 * expected_bs_self_interference))
        npt.assert_allclose(self.simulation.bs.total_interference[1], expected_bs_1_total_interference, atol=1e-2)

        # check SNR
        expected_bs_0_snr = expected_bs_0_rx_power - expected_bs_thermal_noise
        npt.assert_allclose(self.simulation.bs.snr[0], expected_bs_0_snr, atol=1e-2)
        expected_bs_1_snr = expected_bs_1_rx_power - expected_bs_thermal_noise
        npt.assert_allclose(self.simulation.bs.snr[1], expected_bs_1_snr, atol=1e-2)

        # check SINR
        expected_bs_0_sinr = expected_bs_0_rx_power - expected_bs_0_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[0], expected_bs_0_sinr, atol=1e-2)
        expected_bs_1_sinr = expected_bs_1_rx_power - expected_bs_1_total_interference
        npt.assert_allclose(self.simulation.bs.sinr[1], expected_bs_1_sinr, atol=1e-2)

        # Create system
        self.simulation.system = StationFactory.generate_fss_earth_station(self.param.fss_es, random_number_gen)
        self.simulation.system.x = np.array([-10])
        self.simulation.system.y = np.array([0])
        self.simulation.system.height = np.array([1.5])

        # test the method that calculates interference from IMT to FSS space station
        self.simulation.calculate_external_interference()
        #
        # check coupling loss
        expected_path_loss_imt_ue_system = np.array([20, 30, 80, 110, 120, 130, 180, 200])
        expected_ue_system_gain = np.array([1, 0, 1, 1, 1, 1, 1, 1])
        expected_coupling_loss_imt_ue_system = expected_path_loss_imt_ue_system - expected_ue_system_gain \
                                               - self.param.fss_es.antenna_gain
        expected_coupling_loss_imt_ue_system[1] = 30.0
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_system, expected_coupling_loss_imt_ue_system,
                            atol=1e-2)

        expected_path_loss_imt_bs_system = np.array([10.9658, 10.9658, 210.0482, 210.0482])
        expected_bs_system_gain = np.array([1, 2, 1, 2])
        expected_coupling_loss_imt_bs_system = expected_path_loss_imt_bs_system - expected_bs_system_gain \
                                               - self.param.fss_es.antenna_gain
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_system, expected_coupling_loss_imt_bs_system,
                            atol=1e-2)

        # check interference generated by IMT to FSS earth station
        interference_bs = p_tx_bs - expected_coupling_loss_imt_bs_system - self.param.imt.bs_ohmic_loss
        interference_ue = p_tx_ue - expected_coupling_loss_imt_ue_system[[2, 4, 7]] - self.param.imt.ue_ohmic_loss - \
                          self.param.imt.ue_body_loss
        expected_es_rx_interference = 10 * math.log10(np.sum(np.power(10, 0.1 * interference_bs)) + \
                                                      np.sum(np.power(10, 0.1 * interference_ue)))
        self.assertAlmostEqual(self.simulation.system.rx_interference, expected_es_rx_interference, delta=1e-2)

        # check FSS earth station thermal noise
        expected_es_thermal_noise = 10 * np.log10(self.param.fss_es.BOLTZMANN_CONSTANT *
                                                  self.param.fss_es.noise_temperature * self.param.fss_es.bandwidth *
                                                  1e6 * 1e3)
        self.assertAlmostEqual(self.simulation.system.thermal_noise, expected_es_thermal_noise, delta=1e-2)

        # check INR at FSS space station
        expected_inr = expected_es_rx_interference - expected_es_thermal_noise
        self.assertAlmostEqual(self.simulation.system.inr, expected_inr, delta=1e-2)

if __name__ == '__main__':
    unittest.main()
    # Run single test
#    suite = unittest.TestSuite()
#    suite.addTest(SimulationTNFullDuplexTest("test_simulation_2bs_4ue_fss_ss_imbalance"))
#    runner = unittest.TextTestRunner()
#    runner.run(suite)
