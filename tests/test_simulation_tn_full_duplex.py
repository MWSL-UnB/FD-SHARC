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

        self.param.fss_es.x = -5000
        self.param.fss_es.y = 0
        self.param.fss_es.height = 10
        self.param.fss_es.elevation = 20
        self.param.fss_es.azimuth = 0
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
        self.simulation.propagation_imt_bs_bs = PropagationFactory.create_propagation(self.param.imt.bs_bs_channel_model,
                                                                                      self.param, random_number_gen)
        self.simulation.propagation_imt_ue_ue = PropagationFactory.create_propagation(self.param.imt.ue_ue_channel_model,
                                                                                      self.param, random_number_gen)
        self.simulation.propagation_system = PropagationFactory.create_propagation(self.param.fss_ss.channel_model,
                                                                                   self.param, random_number_gen)

        # test coupling loss method
        self.simulation.coupling_loss_imt = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                        self.simulation.ue,
                                                                                        self.simulation.propagation_imt)
        npt.assert_allclose(self.simulation.coupling_loss_imt,
                            np.array([[78.47 - 1 - 1, 89.35 - 2 - 1, 93.27 - 1 - 1, 97.05 - 2 - 1],
                                      [97.55 - 1 - 1, 94.72 - 2 - 1, 91.53 - 1 - 1, 81.99 - 2 - 1]]),
                            atol=1e-2)

        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt_ue_ue)
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue,
                            np.array([[np.nan, 86.43 - 1 - 1, 91.53 - 1 - 1, 95.97 - 1 - 1],
                                      [86.43 - 1 - 1, np.nan, 84.49 - 1 - 1, 92.46 - 1 - 1],
                                      [91.53 - 1 - 1, 84.49 - 1 - 1, np.nan, 88.01 - 1 - 1],
                                      [95.97 - 1 - 1, 92.46 - 1 - 1, 88.01 - 1 - 1, np.nan]]),
                            atol=1e-2)

        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt_bs_bs)
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs,
                            np.array([[np.nan, np.nan, 98.47 - 1 - 2, 98.47 - 2 - 1],
                                      [98.47 - 1 - 2, 98.47 - 2 - 1, np.nan, np.nan]]),
                            atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 1)
        npt.assert_allclose(self.simulation.ue.bandwidth, bandwidth_per_ue * np.ones(4), atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx = 10 + 0 - 10 * math.log10(1)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx]), atol=1e-2)
        npt.assert_allclose(self.simulation.ue.tx_power, 20 * np.ones(4))

        # test method that calculates SINR
        self.simulation.calculate_sinr()

        # # check UE received power
        # npt.assert_allclose(self.simulation.ue.rx_power,
        #                     np.array([-70.48, -80.36, -70.55, -60.00]),
        #                     atol=1e-2)
        #
        # # check UE received interference
        # npt.assert_allclose(self.simulation.ue.rx_interference,
        #                     np.array([-53.53, -52.45, -53.48, -52.44]),
        #                     atol=1e-2)
        #
        # # check UE thermal noise
        # npt.assert_allclose(self.simulation.ue.thermal_noise,
        #                     np.array([-88.44, -88.44, -88.44, -88.44]),
        #                     atol=1e-2)
        #
        # # check tx power
        # npt.assert_allclose(self.simulation.ue.tx_power,
        #                     np.array([20.0, 20.0, 20.0, 20.0]),
        #                     atol=1e-2)
        #
        # # check self-interference
        # npt.assert_allclose(self.simulation.ue.self_interference,
        #                     np.array([-80, -80, -80, -80]),
        #                     atol=1e-2)
        #
        # # check UE thermal noise + interference + self interference
        # npt.assert_allclose(self.simulation.ue.total_interference,
        #                     np.array([-53.52, -52.44, -53.48, -52.42]),
        #                     atol=1e-2)
        #
        # # check SNR
        # npt.assert_allclose(self.simulation.ue.snr,
        #                     np.array([-70.48 - (-88.44), -80.36 - (-88.44), -70.55 - (-88.44), -60.00 - (-88.44)]),
        #                     atol=1e-2)
        #
        # # check SINR
        # npt.assert_allclose(self.simulation.ue.sinr,
        #                     np.array([-70.48 - (-53.52), -80.36 - (-52.44), -70.54 - (-53.48), -60.00 - (-52.42)]),
        #                     atol=5e-2)
        #
        # # check BS received power
        # npt.assert_allclose(self.simulation.bs.rx_power[0],
        #                     np.array([-57.47, -67.35]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.rx_power[1],
        #                     np.array([-57.54, -46.99]),
        #                     atol=1e-2)
        #
        # # check BS received interference
        # npt.assert_allclose(self.simulation.bs.rx_interference[0],
        #                     np.array([-60.27, -63.05]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.rx_interference[1],
        #                     np.array([-75.50, -71.70]),
        #                     atol=1e-2)
        #
        # # check BS thermal noise
        # npt.assert_allclose(self.simulation.bs.thermal_noise,
        #                     np.array([-90.44, -90.44]),
        #                     atol=1e-2)
        #
        # # check BS thermal noise + interference
        # npt.assert_allclose(self.simulation.bs.total_interference[0],
        #                     np.array([-60.27, -63.05]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.total_interference[1],
        #                     np.array([-75.29, -71.62]),
        #                     atol=5e-2)
        #
        # # check self-interference
        # npt.assert_allclose(self.simulation.bs.self_interference[0],
        #                     np.array([-93.01, -93.01]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.self_interference[1],
        #                     np.array([-93.01, -93.01]),
        #                     atol=1e-2)
        #
        # # check SNR
        # npt.assert_allclose(self.simulation.bs.snr[0],
        #                     np.array([32.97, 23.09]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.snr[1],
        #                     np.array([32.91, 43.45]),
        #                     atol=1e-2)
        #
        # # check SINR
        # npt.assert_allclose(self.simulation.bs.sinr[0],
        #                     np.array([2.80, -4.30]),
        #                     atol=1e-2)
        # npt.assert_allclose(self.simulation.bs.sinr[1],
        #                     np.array([17.75, 24.62]),
        #                     atol=1e-2)

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
        npt.assert_equal(self.simulation.link_ul, {0: [1], 1: [-1]})
        self.assertEqual(self.simulation.link, {0: [0, 1], 1: [2]})
        npt.assert_equal(self.simulation.ue.active,
                         np.array([True, True, True, False]))
        npt.assert_equal(self.simulation.ue_beam_rbs, np.array([0, 0, 0, -1]))
        npt.assert_equal(self.simulation.bs_to_ue_beam_idx, np.array([0, 1, 0, -1]))
        npt.assert_equal(self.simulation.bs_beam_rbs, {0: [('DL', 0), ('UL', 0)], 1: [('DL', 0), ('', -1)]})

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
        npt.assert_allclose(self.simulation.coupling_loss_imt,
                            np.array([[78.47 - 1 - 1, 89.35 - 2 - 1, 93.27 - 1 - 1, 97.05 - 0 - 0],
                                      [97.55 - 1 - 1, 94.72 - 0 - 1, 91.53 - 1 - 1, 81.99 - 0 - 0]]),
                            atol=1e-2)

        self.simulation.coupling_loss_imt_ue_ue = self.simulation.calculate_imt_coupling_loss(self.simulation.ue,
                                                                                              self.simulation.ue,
                                                                                              self.simulation.propagation_imt_ue_ue)
        npt.assert_allclose(self.simulation.coupling_loss_imt_ue_ue,
                            np.array([[np.nan, 86.43 - 1 - 1, 91.53 - 1 - 1, 95.97 - 0 - 0],
                                      [86.43 - 1 - 1, np.nan, 84.49 - 1 - 1, 92.46 - 0 - 0],
                                      [91.53 - 1 - 1, 84.49 - 1 - 1, np.nan, 88.01 - 0 - 0],
                                      [95.97 - 0 - 0, 92.46 - 0 - 0, 88.01 - 0 - 0, np.nan]]),
                            atol=1e-2)

        self.simulation.coupling_loss_imt_bs_bs = self.simulation.calculate_imt_coupling_loss(self.simulation.bs,
                                                                                              self.simulation.bs,
                                                                                              self.simulation.propagation_imt_bs_bs)
        npt.assert_allclose(self.simulation.coupling_loss_imt_bs_bs,
                            np.array([[np.nan, np.nan, 98.47 - 1 - 0, 98.47 - 2 - 1],
                                      [98.47 - 1 - 2, 98.47 - 0 - 0, np.nan, np.nan]]),
                            atol=1e-2)

        # test scheduler and bandwidth allocation
        self.simulation.scheduler()
        bandwidth_per_ue = math.trunc((1 - 0.1) * 100 / 1)
        npt.assert_allclose(self.simulation.ue.bandwidth[0:3], bandwidth_per_ue * np.ones(3), atol=1e-2)

        # test power control
        # there is no power control, so BSs and UEs will transmit at maximum
        # power
        self.simulation.power_control()
        p_tx = 10 + 0 - 10 * math.log10(1)
        npt.assert_allclose(self.simulation.bs.tx_power[0], np.array([p_tx]), atol=1e-2)
        npt.assert_allclose(self.simulation.bs.tx_power[1], np.array([p_tx]), atol=1e-2)
        npt.assert_allclose(self.simulation.ue.tx_power[0:3], 20 * np.ones(3))


if __name__ == '__main__':
    unittest.main()
    # Run single test
#    suite = unittest.TestSuite()
#    suite.addTest(SimulationTNFullDuplexTest("test_simulation_2bs_4ue_fss_ss_imbalance"))
#    runner = unittest.TextTestRunner()
#    runner.run(suite)
