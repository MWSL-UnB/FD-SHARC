# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:27:04 2018

@author: Calil
"""

from itertools import compress
import numpy as np
import math
from warnings import filterwarnings

from sharc.support.enumerations import StationType
from sharc.station_manager import StationManager
from sharc.simulation import Simulation
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_factory import PropagationFactory

class SimulationTNFullDuplex(Simulation):
    """
    Implements the full duplex simulation
    """

    def __init__(self, parameters: Parameters, parameter_file: str):
        super().__init__(parameters,parameter_file)
        self.coupling_loss_imt_bs_bs = np.empty(0)
        self.coupling_loss_imt_ue_ue = np.empty(0)
        self.coupling_loss_imt_bs_system = np.empty(0)
        self.coupling_loss_imt_ue_system = np.empty(0)
        self.system_ul_inr = np.empty(0)
        self.system_dl_inr = np.empty(0)
        self.bs_to_ue_beam_idx = np.empty(0)
        self.ue_beam_rbs = np.empty(0)
        self.bs_beam_rbs = dict()

        filterwarnings("ignore", "invalid value encountered in", RuntimeWarning)

    def initialize(self, *args, **kwargs):
        super().initialize(*args,**kwargs)
        num_bs = self.topology.num_base_stations
        num_ue = num_bs * self.parameters.imt.ue_k * self.parameters.imt.ue_k_m
        self.bs_to_ue_beam_idx = -1.0 * np.ones(num_ue, dtype=int)
        self.ue_beam_rbs = -1.0 * np.ones(num_ue, dtype=int)
        self.bs_beam_rbs = dict()

    def snapshot(self, *args, **kwargs):
        write_to_file = kwargs["write_to_file"]
        snapshot_number = kwargs["snapshot_number"]
        seed = kwargs["seed"]
        
        random_number_gen = np.random.RandomState(seed)

        self.propagation_imt = PropagationFactory.create_propagation(self.parameters.imt.channel_model, self.parameters,
                                                                     random_number_gen)
        self.propagation_imt_bs_bs = PropagationFactory.create_propagation(self.parameters.imt.bs_bs_channel_model,
                                                                           self.parameters, random_number_gen)
        self.propagation_imt_ue_ue = PropagationFactory.create_propagation(self.parameters.imt.ue_ue_channel_model,
                                                                           self.parameters, random_number_gen)
        self.propagation_system = PropagationFactory.create_propagation(self.param_system.channel_model,
                                                                        self.parameters,
                                                                        random_number_gen)
        
        # In case of hotspots, base stations coordinates have to be calculated
        # on every snapshot. Anyway, let topology decide whether to calculate
        # or not
        self.topology.calculate_coordinates(random_number_gen)
        
        # Create the base stations (remember that it takes into account the
        # network load factor)
        self.bs = StationFactory.generate_imt_base_stations(self.parameters.imt,
                                                            self.parameters.antenna_imt,
                                                            self.topology,
                                                            random_number_gen)

        # Create IMT user equipments
        self.ue = StationFactory.generate_imt_ue(self.parameters.imt,
                                                 self.parameters.antenna_imt,
                                                 self.topology,
                                                 random_number_gen)
        
        # Create the other system (FSS, HAPS, etc...)
        self.system = StationFactory.generate_system(self.parameters, self.topology, random_number_gen)
        
        #self.plot_scenario()
        
        self.connect_ue_to_bs()
        self.select_ue(random_number_gen)
        
        # Calculate coupling loss after beams are created
        self.coupling_loss_imt = self.calculate_imt_coupling_loss(self.bs, 
                                                                  self.ue,
                                                                  self.propagation_imt)
        
        # UE to UE coupling loss
        self.coupling_loss_imt_ue_ue = self.calculate_imt_coupling_loss(self.ue,
                                                                        self.ue,
                                                                        self.propagation_imt_ue_ue)
        
        # BS to BS coupling loss
        self.coupling_loss_imt_bs_bs = self.calculate_imt_coupling_loss(self.bs,
                                                                        self.bs,
                                                                        self.propagation_imt_bs_bs)
        
        
        # Scheduler which divides the band equally among BSs and UEs
        self.scheduler()
        
        # Stations power control
        self.power_control()
        
        if self.parameters.imt.interfered_with:
            # Execute this piece of code if the other system generates 
            # interference into IMT
            self.calculate_sinr()
            if self.parameters.general.system != "NONE":
                self.calculate_sinr_ext()
            #self.recalculate_sinr()
            #self.calculate_imt_degradation()
            pass
        else:
            # Execute this piece of code if IMT generates interference into
            # the other system
            self.calculate_sinr()
            if self.parameters.general.system != "NONE":
                self.calculate_external_interference()
            #self.calculate_external_degradation()
            pass
        
        self.collect_results(write_to_file, snapshot_number)

    def select_ue(self, random_number_gen: np.random.RandomState):
        # Calculate angles using wrap around technique
        if self.wrap_around_enabled:
            self.bs_to_ue_d_2D, self.bs_to_ue_d_3D, self.bs_to_ue_phi, self.bs_to_ue_theta = \
                self.bs.get_dist_angles_wrap_around(self.ue)
        else:
            self.bs_to_ue_d_2D = self.bs.get_distance_to(self.ue)
            self.bs_to_ue_d_3D = self.bs.get_3d_distance_to(self.ue)
            self.bs_to_ue_phi, self.bs_to_ue_theta = self.bs.get_pointing_vector_to(self.ue)
            
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            # create BS beam RB lists
            self.bs_beam_rbs[bs] = list()

            # select K UE's among the ones that are in DL
            random_number_gen.shuffle(self.link[bs])
            K = self.parameters.imt.ue_k
            self.link_dl[bs] = self.link[bs][:K]
            
            # select up to K UL UEs
            self.link_ul[bs] = np.repeat(-1, len(self.link_dl[bs]))
            ul_ues = random_number_gen.rand(len(self.link[bs][K:])) < self.parameters.imt.ul_load_imbalance
            self.link_ul[bs][ul_ues] = np.array(self.link[bs][K:], dtype=int)[ul_ues]
            
            # delete UEs that are not active
            active_ues = []
            for k, ue_num in enumerate(self.link[bs]):
                if (ue_num in self.link_dl[bs]) or (ue_num in self.link_ul[bs]):
                    active_ues.append(ue_num)
            self.link[bs] = active_ues
            #
            #
            # num_ul_ues = np.count_nonzero(self.link_ul[bs] != -1)
            # num_ues = (len(self.link_dl[bs]) + num_ul_ues)


            # define UE RB group
            for k, ue in enumerate(self.link_dl[bs]):
                self.ue_beam_rbs[ue] = k
                self.bs_beam_rbs[bs].append(('DL', k))
            for k, ue in enumerate(self.link_ul[bs]):
                if ue != -1:
                    self.ue_beam_rbs[ue] = k
                    self.bs_beam_rbs[bs].append(('UL', k))
                else:
                    self.ue_beam_rbs[ue] = -1
                    self.bs_beam_rbs[bs].append(('', -1))

            # redefine link_ul
            self.link_ul[bs] = self.link_ul[bs][ul_ues]
            
            # Activate the selected UE's and create beams
            self.ue.active[self.link[bs]] = True
            for ue in self.link[bs]:
                # add beam to BS antennas
                self.bs.antenna[bs].add_beam(self.bs_to_ue_phi[bs,ue],
                                             self.bs_to_ue_theta[bs,ue])
                # add beam to UE antennas
                self.ue.antenna[ue].add_beam(self.bs_to_ue_phi[bs,ue] - 180,
                                             180 - self.bs_to_ue_theta[bs,ue])
                # set beam indexes
                self.bs_to_ue_beam_idx[ue] = len(self.bs.antenna[bs].beams_list) - 1
                
        
    def calculate_imt_gains(self,
                            station_1: StationManager,
                            station_2: StationManager,
                            c_channel = True) -> np.array:
        """
        Calculates the gains of antennas in station_1 in the direction of
        station_2
        """
        station_1_active = np.where(station_1.active)[0]
        station_2_active = np.where(station_2.active)[0]

        # Initialize variables (phi, theta, beams_idx)
        if(station_1.station_type is StationType.IMT_BS):
            # Define BS variables
            if(station_2.station_type is StationType.IMT_UE):
                phi = self.bs_to_ue_phi
                theta = self.bs_to_ue_theta
                idx_range = 1
                beams_idx = self.bs_to_ue_beam_idx
            elif(station_2.station_type is StationType.IMT_BS):
                if self.wrap_around_enabled:
                    d_2D, d_3D, phi, theta = station_1.get_dist_angles_wrap_around(station_2)
                else:
                    phi, theta = station_1.get_pointing_vector_to(station_2)
                idx_range = self.parameters.imt.ue_k*self.parameters.imt.ue_k_m
                phi = np.repeat(phi,idx_range,1)
                theta = np.repeat(theta,idx_range,1)
                beams_idx = np.tile(np.arange(idx_range),self.bs.num_stations)
            # Calculate gains
            gains = np.zeros(phi.shape)
            for k in station_1_active:
                station_2_mask = np.logical_and(np.repeat(station_2.active,idx_range,0),
                                                beams_idx < len(station_1.antenna[k].beams_list))
                gains[k,station_2_mask] = station_1.antenna[k].calculate_gain(phi_vec=phi[k,station_2_mask],
                                                                              theta_vec=theta[k,station_2_mask],
                                                                              beams_l=beams_idx[station_2_mask])
            return gains

        elif(station_1.station_type is StationType.IMT_UE):
            if self.wrap_around_enabled:
                d_2D, d_3D, phi, theta = station_1.get_dist_angles_wrap_around(station_2)
            else:
                phi, theta = station_1.get_pointing_vector_to(station_2)
            beams_idx = np.zeros(len(station_2_active),dtype=int)

        # Calculate gains
        gains = np.zeros(phi.shape)
        for k in station_1_active:
            gains[k,station_2_active] = station_1.antenna[k].calculate_gain(phi_vec=phi[k,station_2_active],
                                                                            theta_vec=theta[k,station_2_active],
                                                                            beams_l=beams_idx)
        return gains
    
    def calculate_imt_coupling_loss(self,
                                    station_a: StationManager,
                                    station_b: StationManager,
                                    propagation: Propagation,
                                    c_channel = True) -> np.array:
        
        if station_a.station_type is StationType.IMT_BS and \
           station_b.station_type is StationType.IMT_UE and \
           self.parameters.imt.topology == "INDOOR":
            elevation_angles = np.transpose(station_b.get_elevation(station_a))
        else:
            elevation_angles = None
            
        if self.wrap_around_enabled:
            d_2D, d_3D = station_a.get_dist_angles_wrap_around(station_b, return_dist=True)
        else:
            d_2D = station_a.get_distance_to(station_b)
            d_3D = station_a.get_3d_distance_to(station_b)
        freq = self.parameters.imt.frequency

        # define antenna gains
        if station_a.station_type is StationType.IMT_BS and station_b.station_type is StationType.IMT_BS:
            # Path loss repeat
            idx_range = self.parameters.imt.ue_k*self.parameters.imt.ue_k_m
            # Calculate and manipulate gains
            all_gains = self.calculate_imt_gains(station_a, station_b)
            gain_a = all_gains
            gain_b = np.zeros_like(all_gains)

            station_a_active = np.where(station_a.active)[0]
            station_b_active = np.where(station_b.active)[0]
            # loop in the current BS
            idx_range = self.parameters.imt.ue_k*self.parameters.imt.ue_k_m
            for k in station_b_active:
                # loop in the other BS
                for m in station_a_active:
                    station_b_beams = [n for n in range(m * idx_range, (m+1) * idx_range)]

                    station_a_gains = list()
                    # current BS beams
                    for beam_b in self.bs_beam_rbs[k]:
                        associated = False
                        # other BS beams
                        for b_num, beam_a in enumerate(self.bs_beam_rbs[m]):
                            # if the beams are associated
                            if beam_a[1] == beam_b[1] and beam_a[0] != beam_b[0]:
                                # use its gain
                                station_a_gains.append(all_gains[m, k*idx_range + b_num])
                                associated = True
                        # if no beams are associated, use token gain 0.0 instead
                        if not associated:
                            station_a_gains.append(0.0)

                    gain_b[k, station_b_beams] = station_a_gains
        else:
            gain_a = self.calculate_imt_gains(station_a, station_b)
            gain_b = np.transpose(self.calculate_imt_gains(station_b, station_a))

        path_loss = propagation.get_loss(distance_3D=d_3D,
                                         distance_2D=d_2D,
                                         frequency=freq * np.ones(d_2D.shape),
                                         indoor_stations=np.tile(station_b.indoor, (station_a.num_stations, 1)),
                                         bs_height=station_a.height,
                                         ue_height=station_b.height,
                                         elevation=elevation_angles,
                                         shadowing=self.parameters.imt.shadowing,
                                         line_of_sight_prob=self.parameters.imt.line_of_sight_prob,
                                         a_type=station_a.station_type,
                                         b_type=station_b.station_type,
                                         es_params=self.parameters.imt,
                                         tx_gain=gain_a,
                                         rx_gain=gain_b,
                                         imt_site=station_a.site)

        if station_a.station_type is StationType.IMT_BS and station_b.station_type is StationType.IMT_BS:
            path_loss = np.repeat(path_loss, idx_range, 1)

        # collect IMT BS and UE antenna gain and path loss samples
        if station_a.station_type is StationType.IMT_BS and station_b.station_type is StationType.IMT_UE:
            self.path_loss_imt = path_loss
            self.imt_bs_antenna_gain = gain_a
            self.imt_ue_antenna_gain = gain_b
        elif station_a.station_type is StationType.IMT_BS and station_b.station_type is StationType.IMT_BS:
            self.path_loss_imt_bs_bs = path_loss
            self.imt_bs_bs_antenna_gain = gain_a
        elif station_a.station_type is StationType.IMT_UE and station_b.station_type is StationType.IMT_UE:
            self.path_loss_imt_ue_ue = path_loss
            self.imt_ue_ue_antenna_gain = gain_a
            
        # calculate coupling loss
        coupling_loss = np.squeeze(path_loss - gain_a - gain_b)

        return coupling_loss

    def power_control(self):
        """
        Apply downling and uplink power control algorithm
        """
        # Downlink Power Control:
        # Currently, the maximum transmit power of the base station is equaly
        # divided among the selected UEs
        tx_power = self.parameters.imt.bs_conducted_power + self.bs_power_gain \
                   - 10 * math.log10(self.parameters.imt.ue_k)
        # calculate tansmit powers to have a structure such as
        # {bs_1: [pwr_1, pwr_2,...], ...}, where bs_1 is the base station id,
        # pwr_1 is the transmit power from bs_1 to ue_1, pwr_2 is the transmit
        # power from bs_1 to ue_2, etc
        bs_active = np.where(self.bs.active)[0]
        self.bs.tx_power = dict([(bs, tx_power * np.ones(self.parameters.imt.ue_k)) for bs in bs_active])

        # Uplink power control:
        if self.parameters.imt.ue_tx_power_control == "OFF":
            ue_active = np.where(self.ue.active)[0]
            self.ue.tx_power[ue_active] = self.parameters.imt.ue_p_cmax * np.ones(len(ue_active))
        else:
            bs_active = np.where(self.bs.active)[0]
            for bs in bs_active:
                ue = self.link_ul[bs]
                p_cmax = self.parameters.imt.ue_p_cmax
                m_pusch = self.num_rb_per_ue
                p_o_pusch = self.parameters.imt.ue_p_o_pusch
                alpha = self.parameters.imt.ue_alpha
                cl = self.coupling_loss_imt[bs, ue] + self.parameters.imt.bs_ohmic_loss \
                     + self.parameters.imt.ue_ohmic_loss + self.parameters.imt.ue_body_loss
                self.ue.tx_power[ue] = np.minimum(p_cmax, 10 * np.log10(m_pusch) + p_o_pusch + alpha * cl)
    
        
    def calculate_sinr(self):
        """
        Calculates the downlink and uplink SINR for each UE and BS.
        Self-interference is considered
        """    
        ### Downlink SINR
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            ue = self.link_dl[bs]

            self.ue.rx_power[ue] = self.bs.tx_power[bs] - self.parameters.imt.bs_ohmic_loss \
                                   - self.coupling_loss_imt[bs, ue] \
                                   - self.parameters.imt.ue_body_loss \
                                   - self.parameters.imt.ue_ohmic_loss

            # create a list with base stations that generate interference in ue_list
            bs_interf = bs_active

            #  Internal interference
            for bi in bs_interf:
                #  Interference from BSs
                if bi != bs:
                    interference_bs = self.bs.tx_power[bi] \
                                      - self.parameters.imt.bs_ohmic_loss \
                                      - self.coupling_loss_imt[bi, ue] \
                                      - self.parameters.imt.ue_body_loss \
                                      - self.parameters.imt.ue_ohmic_loss
                else:
                    interference_bs = -np.inf
                           
                # Interference from UEs
                interference_ue = -np.inf*np.ones_like(ue)
                ul_ues = self.link_ul[bi]
                interferer_ue = []
                interfered_ue = []
                for ed in ue:
                    for er in ul_ues:
                        if self.ue_beam_rbs[ed] == self.ue_beam_rbs[er]:
                            interfered_ue.append(ed)
                            interferer_ue.append(er)
                interfered_ue_idx = [k in interfered_ue for k in ue]
                interference_ue[interfered_ue_idx] = self.ue.tx_power[interferer_ue] \
                                                     - self.coupling_loss_imt_ue_ue[interferer_ue, interfered_ue] \
                                                     - 2*self.parameters.imt.ue_body_loss \
                                                     - 2*self.parameters.imt.ue_ohmic_loss
           
                self.ue.rx_interference[ue] = 10*np.log10( \
                    np.power(10, 0.1*self.ue.rx_interference[ue]) + \
                    np.power(10, 0.1*interference_bs) + \
                    np.power(10, 0.1*interference_ue))

            # in TN-FD only BSs will have self-interference
            self.ue.self_interference[ue] = -np.inf

        self.ue.thermal_noise = \
            10*math.log10(self.parameters.imt.BOLTZMANN_CONSTANT*self.parameters.imt.noise_temperature*1e3) + \
            10*np.log10(self.ue.bandwidth * 1e6) + \
            self.ue.noise_figure
            
        self.ue.total_interference = \
            10*np.log10(np.power(10, 0.1*self.ue.rx_interference) + \
                        np.power(10, 0.1*self.ue.thermal_noise))
            
        self.ue.sinr = self.ue.rx_power - self.ue.total_interference
        self.ue.snr = self.ue.rx_power - self.ue.thermal_noise
        
        ### Uplink SNIR
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            ue = self.link_ul[bs]
            self_interference_idx = [k < len(ue) for k in range(len(self.link_dl[bs]))]
            self.bs.rx_power[bs] = self.ue.tx_power[ue]  \
                                   - self.parameters.imt.ue_ohmic_loss - self.parameters.imt.ue_body_loss \
                                   - self.coupling_loss_imt[bs,ue] - self.parameters.imt.bs_ohmic_loss
            # create a list of BSs that serve the interfering UEs
            bs_interf = [b for b in bs_active if b not in [bs]]

            # calculate intra system interference
            for bi in bs_interf:
                # interference from UEs
                interference_ue = -np.inf * np.ones_like(ue)
                ul_ues = self.link_ul[bi]
                interferer_ue = []
                interfered_ul_beam = []
                for k, ed in enumerate(ue):
                    for er in ul_ues:
                        if self.ue_beam_rbs[ed] == self.ue_beam_rbs[er]:
                            interfered_ul_beam.append(k)
                            interferer_ue.append(er)
                interference_ue[interfered_ul_beam] = self.ue.tx_power[interferer_ue]\
                                                      - self.parameters.imt.ue_ohmic_loss\
                                                      - self.parameters.imt.ue_body_loss\
                                                      - self.coupling_loss_imt[bs, interferer_ue]\
                                                      - self.parameters.imt.bs_ohmic_loss

                # interference from BSs
                bs_interfered_beam = []
                bi_interferer_beam = []
                bs_ul_interfered_beam = []
                bi_dl_interferer_beam = []
                ul_count = 0
                # current BS beams
                for bs_num, beam_bs in enumerate(self.bs_beam_rbs[bs]):
                    # other BS beams
                    dl_count = 0
                    for bi_num, beam_bi in enumerate(self.bs_beam_rbs[bi]):
                        # if the beams are associated
                        if beam_bs[0] == 'UL' and beam_bi[0] == 'DL' and beam_bs[1] == beam_bi[1]:
                            # use its index
                            bs_interfered_beam.append(bs_num)
                            bi_interferer_beam.append(bs*self.parameters.imt.ue_k*self.parameters.imt.ue_k_m + bi_num)
                            bi_dl_interferer_beam.append(dl_count)
                            bs_ul_interfered_beam.append(ul_count)
                        if beam_bi[0] == 'DL':
                            dl_count += 1
                    if beam_bs[0] == 'UL':
                        ul_count += 1

                interference_bs = self.bs.tx_power[bi][bi_dl_interferer_beam] \
                                  - 2*self.parameters.imt.bs_ohmic_loss \
                                  - self.coupling_loss_imt_bs_bs[bi, bi_interferer_beam]
                                
                self.bs.rx_interference[bs][bs_ul_interfered_beam] = 10*np.log10( \
                    np.power(10, 0.1*self.bs.rx_interference[bs][bs_ul_interfered_beam])
                    + np.power(10, 0.1*interference_ue) \
                    + np.power(10, 0.1*interference_bs))
                
            # calculate self interference
            self.bs.self_interference[bs][self_interference_idx] = self.bs.tx_power[bs][self_interference_idx] - \
                                                                   self.bs.sic[bs]
            
            # calculate N
            self.bs.thermal_noise[bs] = \
                10*np.log10(self.parameters.imt.BOLTZMANN_CONSTANT*self.parameters.imt.noise_temperature*1e3) + \
                10*np.log10(self.bs.bandwidth[bs] * 1e6) + \
                self.bs.noise_figure[bs]
    
            # calculate I+N+SI
            self.bs.total_interference[bs] = \
                10*np.log10(np.power(10, 0.1*self.bs.rx_interference[bs]) + \
                            np.power(10, 0.1*self.bs.thermal_noise[bs])   + \
                            np.power(10, 0.1*self.bs.self_interference[bs]))
                
            # calculate SNR and SINR
            self.bs.sinr[bs] = self.bs.rx_power[bs] - \
                               self.bs.total_interference[bs][np.array(self.ue_beam_rbs[ue],dtype=int)]
            self.bs.snr[bs] = self.bs.rx_power[bs] - self.bs.thermal_noise[bs]

        
    def calculate_sinr_ext(self):
        """
        Calculates the SINR and INR for each UE and BS taking into
        account the interference that is generated by the other system into 
        IMT system.
        """
        ### Downlink
        self.coupling_loss_imt_ue_system = self.calculate_coupling_loss(self.system, 
                                                                        self.ue,
                                                                        self.propagation_system)       
        
        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        # calculate interference only to active UE's
        ue = np.where(self.ue.active)[0]
        tx_power = self.param_system.tx_power_density + 10*np.log10(self.ue.bandwidth[ue]*1e6) + 30
        self.ue.ext_interference[ue] = tx_power - self.coupling_loss_imt_ue_system[ue] \
                            - self.parameters.imt.ue_body_loss - self.parameters.imt.ue_ohmic_loss

        self.ue.sinr_ext[ue] = self.ue.rx_power[ue] \
            - (10*np.log10(np.power(10, 0.1*self.ue.total_interference[ue]) + np.power(10, 0.1*self.ue.ext_interference[ue])))
        self.ue.inr[ue] = self.ue.ext_interference[ue] - self.ue.thermal_noise[ue]
        
        ### Uplink
        self.coupling_loss_imt_bs_system = self.calculate_coupling_loss(self.system, 
                                                                     self.bs,
                                                                     self.propagation_system)       
        
        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        # calculate interference only to active UE's
        bs_active = np.where(self.bs.active)[0]
        tx_power = self.param_system.tx_power_density + 10*np.log10(self.bs.bandwidth*1e6) + 30
        for bs in bs_active:
            active_beams = [i for i in range(bs*self.parameters.imt.ue_k, (bs+1)*self.parameters.imt.ue_k)]
            self.bs.ext_interference[bs] = tx_power[bs] - self.coupling_loss_imt_bs_system[active_beams] \
                                            - self.parameters.imt.bs_ohmic_loss

            self.bs.sinr_ext[bs] = self.bs.rx_power[bs] \
                - (10*np.log10(np.power(10, 0.1*self.bs.total_interference[bs]) + np.power(10, 0.1*self.bs.ext_interference[bs])))
            self.bs.inr[bs] = self.bs.ext_interference[bs] - self.bs.thermal_noise[bs]
        
        
    def calculate_external_interference(self):
        """
        Calculates interference that IMT system generates on other system
        """
        ### Downlink & Uplink
        self.coupling_loss_imt_bs_system = self.calculate_coupling_loss(self.system, 
                                                                        self.bs,
                                                                        self.propagation_system)
        self.coupling_loss_imt_ue_system = self.calculate_coupling_loss(self.system, 
                                                                        self.ue,
                                                                        self.propagation_system)
        
        # calculate N
        self.system.thermal_noise = \
            10*math.log10(self.param_system.BOLTZMANN_CONSTANT* \
                          self.param_system.noise_temperature*1e3) + \
                          10*math.log10(self.param_system.bandwidth * 1e6)
                          
        # Overlapping bandwidth weights
        weights = self.calculate_bw_weights(self.parameters.imt.bandwidth,
                                            self.param_system.bandwidth,
                                            self.parameters.imt.ue_k)

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        # calculate interference only from active UE's
        bs_active = np.where(self.bs.active)[0]
        for bs in bs_active:
            active_beams = [i for i in range(bs*self.parameters.imt.ue_k, (bs+1)*self.parameters.imt.ue_k)]
            
            interference_bs = self.bs.tx_power[bs] - self.coupling_loss_imt_bs_system[active_beams] - \
                              self.parameters.imt.bs_ohmic_loss
                              
            total_interference_bs = np.sum(weights*np.power(10, 0.1*interference_bs))
            
                                
            self.system.rx_interference = 10*math.log10( \
                    math.pow(10, 0.1*self.system.rx_interference) + \
                    total_interference_bs)
        
        self.system_dl_inr = np.array([self.system.rx_interference - self.system.thermal_noise])
        
        # UE interference
        accumulated_interference_ue = -np.inf
        for bs in bs_active:
            ue = self.link_ul[bs]
            ue_interf_mask = [k < len(ue) for k in range(len(self.link_dl[bs]))]

            interference_ue = self.ue.tx_power[ue] - self.parameters.imt.ue_ohmic_loss \
                              - self.parameters.imt.ue_body_loss \
                              - self.coupling_loss_imt_ue_system[ue]
                              
            total_interference_ue = np.sum(weights[ue_interf_mask]*np.power(10, 0.1*interference_ue))
                     
            accumulated_interference_ue = 10*np.log10(np.power(10, 0.1*accumulated_interference_ue) + \
                                          total_interference_ue)
            
            self.system.rx_interference = 10*np.log10(np.power(10, 0.1*self.system.rx_interference) + \
                                          total_interference_ue)
        
        self.system_ul_inr = np.array([accumulated_interference_ue - self.system.thermal_noise])

        # calculate INR at the system
        self.system.inr = np.array([self.system.rx_interference - self.system.thermal_noise])
        
        
    def collect_results(self, write_to_file: bool, snapshot_number: int):
        if (not self.parameters.imt.interfered_with) and self.parameters.general.system != "NONE":
            self.results.system_inr.extend(self.system.inr.tolist())
            self.results.system_inr_scaled.extend((self.system.inr + 10*math.log10(self.param_system.inr_scaling)).tolist())
            self.results.system_ul_inr_scaled.extend((self.system_ul_inr + 10*math.log10(self.param_system.inr_scaling)).tolist())
            self.results.system_dl_inr_scaled.extend((self.system_dl_inr + 10*math.log10(self.param_system.inr_scaling)).tolist())
        
        bs_active = np.where(self.bs.active)[0]
        total_ue_tput = 0
        total_bs_tput = 0
        for bs in bs_active:
            ue = self.link[bs]
            ue_dl = self.link_dl[bs]
            ue_ul = self.link_ul[bs]
            self.results.imt_path_loss.extend(self.path_loss_imt[bs,ue])
            self.results.imt_coupling_loss.extend(self.coupling_loss_imt[bs,ue])
            
            if not self.parameters.general.suppress_large_results:
                self.results.imt_coupling_loss_all.extend(self.coupling_loss_imt[bs,:])
            
                bs_bs_pl = self.path_loss_imt_bs_bs[bs, ~np.isnan(self.coupling_loss_imt_bs_bs[bs,:])]
                self.results.imt_bs_bs_path_loss.extend(bs_bs_pl)
            
                ue_ue_pl = np.ravel(self.path_loss_imt_ue_ue[ue])
                ue_ue_pl = ue_ue_pl[~np.isnan(ue_ue_pl)]
                self.results.imt_ue_ue_path_loss.extend(ue_ue_pl)
            
                bs_bs_cl = self.coupling_loss_imt_bs_bs[bs, ~np.isnan(self.coupling_loss_imt_bs_bs[bs,:])]
                self.results.imt_coupling_loss_bs_bs.extend(bs_bs_cl)
            
                ue_ue_cl = np.ravel(self.coupling_loss_imt_ue_ue[ue])
                ue_ue_cl = ue_ue_cl[~np.isnan(ue_ue_cl)]
                self.results.imt_coupling_loss_ue_ue.extend(ue_ue_cl)
            
                bs_bs_ag = self.imt_bs_bs_antenna_gain[bs, ~np.isnan(self.imt_bs_bs_antenna_gain[bs,:])]
                self.results.imt_bs_bs_antenna_gain.extend(bs_bs_ag)
            
                ue_ue_ag = np.ravel(self.imt_ue_ue_antenna_gain[ue])
                ue_ue_ag = ue_ue_ag[~np.isnan(ue_ue_ag)]
                self.results.imt_ue_ue_antenna_gain.extend(ue_ue_ag)
            
            self.results.imt_bs_antenna_gain.extend(self.imt_bs_antenna_gain[bs,ue])
            self.results.imt_ue_antenna_gain.extend(self.imt_ue_antenna_gain[bs,ue])
            
            
            tput = self.calculate_imt_tput(self.ue.sinr[ue_dl],
                                           self.parameters.imt.dl_sinr_min,
                                           self.parameters.imt.dl_sinr_max,
                                           self.parameters.imt.dl_attenuation_factor)
            tput = tput*self.ue.bandwidth[ue_dl]
            self.results.imt_dl_tput.extend(tput.tolist())
            
            total_ue_tput += np.sum(tput)
            
            tput = self.calculate_imt_tput(self.bs.sinr[bs],
                                           self.parameters.imt.ul_sinr_min,
                                           self.parameters.imt.ul_sinr_max,
                                           self.parameters.imt.ul_attenuation_factor)
            tput = tput*self.bs.bandwidth[bs]
            self.results.imt_ul_tput.extend(tput.tolist())
            
            total_bs_tput += np.sum(tput)

            if self.parameters.imt.interfered_with:
                tput_ext = self.calculate_imt_tput(self.ue.sinr_ext[ue],
                                                   self.parameters.imt.dl_sinr_min,
                                                   self.parameters.imt.dl_sinr_max,
                                                   self.parameters.imt.dl_attenuation_factor)
                tput_ext = tput*self.ue.bandwidth[ue]
                self.results.imt_dl_tput_ext.extend(tput_ext.tolist()) 
                self.results.imt_dl_sinr_ext.extend(self.ue.sinr_ext[ue].tolist())
                self.results.imt_dl_inr.extend(self.ue.inr[ue].tolist())
                
                tput_ext = self.calculate_imt_tput(self.bs.sinr_ext[bs],
                                                      self.parameters.imt.ul_sinr_min,
                                                      self.parameters.imt.ul_sinr_max,
                                                      self.parameters.imt.ul_attenuation_factor)
                tput_ext = tput*self.bs.bandwidth[bs]
                self.results.imt_ul_tput_ext.extend([tput_ext])  
                self.results.imt_ul_sinr_ext.extend(self.bs.sinr_ext[bs].tolist())
                self.results.imt_ul_inr.extend(self.bs.inr[bs].tolist())

            if self.parameters.general.system != "NONE":
                self.results.system_imt_ue_antenna_gain.extend(self.system_imt_ue_antenna_gain[0,ue_ul])
                self.results.imt_ue_system_antenna_gain.extend(self.imt_ue_system_antenna_gain[0,ue_ul])
                
                active_beams = [i for i in range(bs*self.parameters.imt.ue_k, (bs+1)*self.parameters.imt.ue_k)]
                self.results.system_imt_bs_antenna_gain.extend(self.system_imt_bs_antenna_gain[0,active_beams])
                self.results.imt_bs_system_antenna_gain.extend(self.imt_bs_system_antenna_gain[0,active_beams])
            
                self.results.system_ul_coupling_loss.extend(self.coupling_loss_imt_ue_system[ue_ul])
                self.results.system_dl_coupling_loss.extend([self.coupling_loss_imt_bs_system[active_beams]])

            self.results.imt_dl_tx_power.extend(self.bs.tx_power[bs].tolist())
            self.results.imt_dl_rx_power.extend(self.ue.rx_power[ue].tolist())
            self.results.imt_dl_sinr.extend(self.ue.sinr[ue_dl].tolist())
            self.results.imt_dl_snr.extend(self.ue.snr[ue_dl].tolist())
            self.results.imt_dl_ue_total_interf.extend(self.ue.total_interference[ue_dl].tolist())
            self.results.imt_dl_ue_self_interf.extend(self.ue.self_interference[ue_dl].tolist())
            self.results.imt_ue_interf_from_ue.extend(self.ue.interference_from_ue[ue_dl].tolist())
            self.results.imt_ue_interf_from_bs.extend(self.ue.interference_from_bs[ue_dl].tolist())
            self.results.imt_ue_thermal_noise.extend(self.ue.thermal_noise[ue_dl].tolist())

            self.results.imt_ul_tx_power.extend(self.ue.tx_power[ue_ul].tolist())
            self.results.imt_ul_rx_power.extend(self.bs.rx_power[bs].tolist())
            self.results.imt_ul_sinr.extend(self.bs.sinr[bs].tolist())
            self.results.imt_ul_snr.extend(self.bs.snr[bs].tolist())
            if len(ue_ul):
                self.results.imt_ul_bs_total_interf.extend(self.bs.total_interference[bs].tolist())
                self.results.imt_ul_bs_self_interf.extend(self.bs.self_interference[bs].tolist())
                self.results.imt_bs_interf_from_ue.extend(self.bs.interference_from_ue[bs].tolist())
                self.results.imt_bs_interf_from_bs.extend(self.bs.interference_from_bs[bs].tolist())
                self.results.imt_bs_thermal_noise.extend([self.bs.thermal_noise[bs]])
            
        self.results.imt_total_tput.extend([total_ue_tput + total_bs_tput])
        self.results.imt_dl_total_tput.extend([total_ue_tput])
        self.results.imt_ul_total_tput.extend([total_bs_tput])
            
        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)

