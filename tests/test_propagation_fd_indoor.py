import unittest
import numpy as np
import numpy.testing as npt

from sharc.propagation.propagation_fd_indoor import PropagationFDIndoor
from sharc.parameters.parameters_indoor import ParametersIndoor
from sharc.support.enumerations import StationType

class PropagationFDIndoorTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_loss(self):
        params = ParametersIndoor()
        params.basic_path_loss = "INH_OFFICE"
        params.n_rows = 2
        params.n_colums = 1
        params.ue_indoor_percent = .95
        params.num_cells = 3
        params.building_class = "TRADITIONAL"

        ue_per_bs = 1
        num_bs = params.num_cells * params.n_rows * params.n_colums
        num_ue = num_bs * ue_per_bs

        h_bs = 3 * np.ones(num_bs)
        h_ue = 1.5 * np.ones(num_ue)

        # UE-UE path loss
        distance_2D = 150 * np.random.random((num_ue, num_ue))
        frequency = 27000 * np.ones(distance_2D.shape)
        indoor = np.array([np.random.rand(num_ue) < params.ue_indoor_percent])
        distance_3D = np.sqrt(distance_2D ** 2 + (h_ue[:, np.newaxis] - h_ue) ** 2)
        height_diff = np.tile(h_ue, (num_ue, 1)) - np.tile(h_ue, (num_ue, 1))
        elevation = np.degrees(np.arctan(height_diff / distance_2D))

        propagation_indoor = PropagationFDIndoor(np.random.RandomState(), params, ue_per_bs)
        loss_indoor = propagation_indoor.get_loss(distance_3D=distance_3D,
                                                  distance_2D=distance_2D,
                                                  elevation=elevation,
                                                  frequency=frequency,
                                                  indoor_stations=indoor,
                                                  shadowing=False,
                                                  a_type=StationType.IMT_UE,
                                                  b_type=StationType.IMT_UE)

        self.assertEqual(loss_indoor.shape, (6, 6))

        # BS-BS path loss
        distance_2D = 150 * np.random.random((num_bs, num_bs))
        frequency = 27000 * np.ones(distance_2D.shape)
        indoor = np.ones((1, num_bs), dtype=bool)
        distance_3D = np.sqrt(distance_2D ** 2 + (h_ue[:, np.newaxis] - h_ue) ** 2)
        height_diff = np.tile(h_ue, (num_ue, 1)) - np.tile(h_ue, (num_ue, 1))
        elevation = np.degrees(np.arctan(height_diff / distance_2D))

        propagation_indoor = PropagationFDIndoor(np.random.RandomState(), params, ue_per_bs)
        loss_indoor = propagation_indoor.get_loss(distance_3D=distance_3D,
                                                  distance_2D=distance_2D,
                                                  elevation=elevation,
                                                  frequency=frequency,
                                                  indoor_stations=indoor,
                                                  shadowing=False,
                                                  a_type=StationType.IMT_BS,
                                                  b_type=StationType.IMT_BS)

        self.assertEqual(loss_indoor.shape, (6, 6))

if __name__ == '__main__':
    unittest.main()
