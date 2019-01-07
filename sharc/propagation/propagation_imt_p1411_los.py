from sharc.propagation.propagation import Propagation

import numpy as np
from warnings import filterwarnings, catch_warnings

class PropagationImtP1411Los(Propagation):
    """

    """
    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)

        self.los_alpha = 2.12
        self.los_beta = 29.20
        self.los_gamma = 2.11
        self.los_sigma = 5.06

        self.nlos_alpha = 4.00
        self.nlos_beta = 10.20
        self.nlos_gamma = 2.36
        self.nlos_sigma = 7.60

    def get_loss(self, *args, **kwargs) -> np.array:
        """

        """
        d_3D = kwargs["distance_3D"]
        d_2D = kwargs["distance_2D"]

        f = kwargs["frequency"] / 1e3
        self.shadow = kwargs.pop("shadow", True)
        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        loss = np.zeros_like(d_2D)

        los_probability = self.get_los_probability(d_2D)
        los_condition = self.get_los_condition(los_probability)

        i_los = np.where(los_condition == True)[:2]
        i_nlos = np.where(los_condition == False)[:2]

        loss = np.empty(d_2D.shape)

        if len(i_los[0]):
            loss_los = self.calculate_loss(self.los_alpha, self.los_beta, self.los_gamma, self.los_sigma, d_3D, f)
            loss[i_los] = loss_los[i_los]

        if len(i_nlos[0]):
            loss_nlos = self.calculate_loss(self.nlos_alpha, self.nlos_beta, self.nlos_gamma, self.nlos_sigma, d_3D, f)
            loss[i_nlos] = loss_nlos[i_nlos]

        return loss

    def calculate_loss(self, alpha, beta, gamma, sigma, d, f):
        """
        Calculates the alpha beta gamma sigma loss.
        :param alpha:
        :param beta:
        :param gamma:
        :param sigma:
        :param d:
        :param f:
        :return loss:
        """
        if self.shadow:
            shadow_loss = self.random_number_gen.normal(0.0, sigma, np.shape(d))
        else:
            shadow_loss = 0.0

        with catch_warnings(record=False) as w:
            filterwarnings("ignore", "divide by zero encountered in log10", RuntimeWarning)
            loss = 10 * alpha * np.log10(d) + 10 * gamma * np.log10(f) + beta + shadow_loss

        return loss

    def get_los_condition(self, p_los: np.array) -> np.array:
        """
        Evaluates if links are LOS (True) of NLOS (False).

        Parameters
        ----------
            p_los : array with LOS probabilities for each user link.

        Returns
        -------
            An array with True or False if links are in LOS of NLOS
            condition, respectively.
        """
        los_condition = self.random_number_gen.random_sample(p_los.shape) < p_los
        return los_condition


    def get_los_probability(self, distance_2D: np.array) -> np.array:
        """
        Returns the line-of-sight (LOS) probability

        Parameters
        ----------
            distance_2D : Two-dimensional array with 2D distance values from
                          base station to user terminal [m]

        Returns
        -------
            LOS probability as a numpy array with same length as distance
        """

        p_los = np.ones(distance_2D.shape)
        idl = np.where(distance_2D > 18)
        p_los[idl] = (18/distance_2D[idl] + np.exp(-distance_2D[idl]/36)*(1-18/distance_2D[idl]))

        return p_los
