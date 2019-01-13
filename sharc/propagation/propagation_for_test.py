from sharc.propagation.propagation import Propagation

import numpy as np

class PropagationForTest(Propagation):
    """
    Implements a propagation model for testing, which yields a loss equal to the distance.
    """

    def get_loss(self, *args, **kwargs) -> np.array:
        if "distance_2D" in kwargs:
            d = kwargs["distance_2D"]
        else:
            d = kwargs["distance_3D"]

        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        loss = d

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
