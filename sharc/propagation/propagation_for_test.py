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

        f = kwargs["frequency"]
        loss = d

        return loss
