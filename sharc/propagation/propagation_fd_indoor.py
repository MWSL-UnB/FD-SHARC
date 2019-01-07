from sharc.parameters.parameters_indoor import ParametersIndoor
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_inh_office import PropagationInhOffice
from sharc.propagation.propagation_building_entry_loss import PropagationBuildingEntryLoss
from sharc.support.enumerations import StationType

import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

class PropagationFDIndoor(Propagation):
    """
    This is a wrapper class which can be used for indoor simulations. It
    calculates the basic BS-BS or UE-UE path loss of the same building.
    It also includes an additional building entry loss for the outdoor UE's
    that are served by indoor BS's.
    """
    # For stations that are not in the same building, this value is assigned
    # so this kind of inter-building interference will not be effectivelly
    # taken into account during SINR calculations. This is assumption
    # simplifies the implementation and it is reasonable: intra-building
    # interference is much higher than inter-building interference
    HIGH_PATH_LOSS = 400

    def __init__(self, random_number_gen: np.random.RandomState, param: ParametersIndoor, ue_per_cell):
        super().__init__(random_number_gen)

        if param.basic_path_loss == "FSPL":
            self.bpl = PropagationFreeSpace(random_number_gen)
        elif param.basic_path_loss == "INH_OFFICE":
            self.bpl = PropagationInhOffice(random_number_gen)
        else:
            sys.stderr.write("ERROR\nInvalid indoor basic path loss model: " + param.basic_path_loss)
            sys.exit(1)

        self.bel = PropagationBuildingEntryLoss(random_number_gen)
        self.building_class = param.building_class
        self.bs_per_building = param.num_cells
        self.ue_per_building = ue_per_cell*param.num_cells
        self.num_builds = param.n_rows * param.n_colums

    def get_loss(self, *args, **kwargs) -> np.array:
        """
        Calculates path loss for LOS and NLOS cases with respective shadowing
        (if shadowing has to be added)
        Parameters
        ----------
            distance_3D (np.array) : 3D distances between stations
            distance_2D (np.array) : 2D distances between stations
            elevation (np.array) : elevation angles from UE's to BS's
            frequency (np.array) : center frequencie [MHz]
            indoor (np.array) : indicates whether UE is indoor
            shadowing (bool) : if shadowing should be added or not
        Returns
        -------
            array with path loss values with dimensions of distance_2D
        """
        distance_3D = kwargs["distance_3D"]
        distance_2D = kwargs["distance_2D"]
        elevation = kwargs["elevation"]
        frequency = kwargs["frequency"]
        indoor = kwargs["indoor_stations"]
        shadowing = kwargs["shadowing"]
        a_type = kwargs["a_type"]
        b_type = kwargs["b_type"]

        if a_type is StationType.IMT_BS and b_type is StationType.IMT_BS:
            leap_i = self.bs_per_building
            all_los = True
            indoor_stas = np.ones_like(distance_2D, dtype=bool)
        elif a_type is StationType.IMT_UE and b_type is StationType.IMT_UE:
            leap_i = self.ue_per_building
            all_los = False
            indoor_stas = np.logical_and(indoor.transpose(),indoor)
        else:
            sys.stderr.write("ERROR\nChannel model FD_INDOOR being used for non FD propagation.")
            sys.exit(1)

        loss = PropagationFDIndoor.HIGH_PATH_LOSS*np.ones(frequency.shape)
        for i in range(self.num_builds):
            ui = int(leap_i*i)
            uf = int(leap_i*(i+1))

            # calculate basic path loss
            loss[ui:uf,ui:uf] = self.bpl.get_loss(distance_3D = distance_3D[ui:uf, ui:uf],
                                                  distance_2D = distance_2D[ui:uf, ui:uf],
                                                  frequency = frequency[ui:uf, ui:uf],
                                                  indoor = indoor[0, ui:uf],
                                                  shadowing = shadowing,
                                                  all_los=all_los)

            # calculates the additional building entry loss for outdoor UEs to indoor UEs
            bel = (~ indoor_stas[ui:uf, ui:uf]) * self.bel.get_loss(frequency[ui:uf, ui:uf], elevation[ui:uf, ui:uf],
                                                                    "RANDOM", self.building_class)

            loss[ui:uf, ui:uf] = loss[ui:uf, ui:uf] + bel

        return loss


if __name__ == '__main__':
    params = ParametersIndoor()
    params.basic_path_loss = "INH_OFFICE"
    params.n_rows = 3
    params.n_colums = 1
    params.ue_indoor_percent = .95
    params.num_cells = 3
    params.building_class = "TRADITIONAL"

    ue_per_bs = 3

    num_bs = params.num_cells*params.n_rows*params.n_colums
    num_ue = num_bs*ue_per_bs

    # UE-UE path loss
    distance_2D = 150*np.random.random((num_ue, num_ue))
    frequency = 27000*np.ones(distance_2D.shape)
    indoor = np.array([np.random.rand(num_ue) < params.ue_indoor_percent])
    h_bs = 3*np.ones(num_bs)
    h_ue = 1.5*np.ones(num_ue)
    distance_3D = np.sqrt(distance_2D**2 + (h_ue[:,np.newaxis] - h_ue)**2)
    height_diff = np.tile(h_ue, (num_ue, 1)) - np.tile(h_ue, (num_ue, 1))
    elevation = np.degrees(np.arctan(height_diff/distance_2D))

    propagation_indoor = PropagationFDIndoor(np.random.RandomState(),params,ue_per_bs)
    loss_indoor = propagation_indoor.get_loss(distance_3D = distance_3D,
                                              distance_2D = distance_2D,
                                              elevation = elevation,
                                              frequency = frequency,
                                              indoor_stations = indoor,
                                              shadowing = False,
                                              a_type=StationType.IMT_UE,
                                              b_type=StationType.IMT_UE)

