from sharc.propagation.propagation import Propagation

import numpy as np


class PropagationImtP1411(Propagation):
    """
    Implements the propagation model described in ITU-R P.1411-9, section 4.1
    This class is supposed to be used for BS-BS and UE-UE propagation

    Frequency in MHz and distance in meters!
    """

    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)

        self.fspl_to_los_dist = 4
        self.los_dist = 5
        self.los_to_nlos_dist = 15
        self.nlos_dist = 30

        self.fspl_alpha = 2.0
        self.fspl_beta = 32.44
        self.fspl_gamma = 2.0
        self.fspl_sigma = 0.0

        self.los_alpha = 2.12
        self.los_beta = 29.20
        self.los_gamma = 2.11
        self.los_sigma = 5.06

        self.nlos_alpha = 4.00
        self.nlos_beta = 10.20
        self.nlos_gamma = 2.36
        self.nlos_sigma = 7.60

        self.shadow = True

    def get_loss(self, *args, **kwargs) -> np.array:
        if "distance_3D" in kwargs:
            d = kwargs["distance_3D"]
        else:
            d = kwargs["distance_2D"]

        f = kwargs["frequency"] / 1e3
        self.shadow = kwargs.pop("shadow", True)
        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        loss = np.zeros_like(d)

        fspl, fspl_los, los, los_nlos, nlos = self.get_loss_range(d)

        loss[fspl] = self.calculate_loss(self.fspl_alpha, self.fspl_beta, self.fspl_gamma, self.fspl_sigma, d[fspl],
                                         f[fspl])
        loss[fspl_los] = self.interpolate(d[fspl_los], f[fspl_los],'FSPL_TO_LOS')
        loss[los] = self.calculate_loss(self.los_alpha, self.los_beta, self.los_gamma, self.los_sigma, d[los], f[los])
        loss[los_nlos] = self.interpolate(d[los_nlos], f[los_nlos], 'LOS_TO_NLOS')
        loss[nlos] = self.calculate_loss(self.nlos_alpha, self.nlos_beta, self.nlos_gamma, self.nlos_sigma, d[nlos],
                                         f[nlos])

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss

    def interpolate(self, dist, freq, loss_range):
        if loss_range == 'FSPL_TO_LOS':
            low_dist = self.fspl_to_los_dist
            low_sigma = self.fspl_sigma
            low_loss = self.calculate_loss(self.fspl_alpha, self.fspl_beta, self.fspl_gamma, 0.0, low_dist, freq)
            up_dist = self.los_dist
            up_sigma = self.los_sigma
            up_loss = self.calculate_loss(self.los_alpha, self.los_beta, self.los_gamma, 0.0, up_dist, freq)
        elif loss_range == 'LOS_TO_NLOS':
            low_dist = self.los_to_nlos_dist
            low_sigma = self.los_sigma
            low_loss = self.calculate_loss(self.los_alpha, self.los_beta, self.los_gamma, 0.0, low_dist, freq)
            up_dist = self.nlos_dist
            up_sigma = self.nlos_sigma
            up_loss = self.calculate_loss(self.nlos_alpha, self.nlos_beta, self.nlos_gamma, 0.0, up_dist, freq)

        loss = low_loss + (dist - low_dist)*(up_loss - low_loss)/(up_dist - low_dist)

        if self.shadow:
            interp_sigma = low_sigma + (dist - low_dist)*(up_sigma - low_sigma)/(up_dist - low_dist)
            loss += self.random_number_gen.normal(0.0, interp_sigma)

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

        loss = 10 * alpha * np.log10(d) + 10 * gamma * np.log10(f) + beta + shadow_loss
        return loss

    def get_loss_range(self, d: np.array):
        """
        Defines the distances to which FSPL, LOS and NLOS are supposed to be used.
        :param d:
        :return loss_range:
        """
        fspl_range = d <= self.fspl_to_los_dist
        fspl_to_los_range = np.logical_and(d > self.fspl_to_los_dist, d <= self.los_dist)
        los_range = np.logical_and(d > self.los_dist, d <= self.los_to_nlos_dist)
        los_to_nlos_range = np.logical_and(d > self.los_to_nlos_dist, d <= self.nlos_dist)
        nlos_range = d > self.nlos_dist

        return fspl_range, fspl_to_los_range, los_range, los_to_nlos_range, nlos_range


if __name__ == '__main__':
    # Imports
    import matplotlib.pyplot as plt

    # Create propagation object
    rnd = np.random.RandomState()
    propag = PropagationImtP1411(rnd)

    # Input parameters
    dist = np.linspace(0.1, 100, num=500)
    freq = 40e6*np.ones_like(dist)

    # No shadowing
    shad = False
    # Calculate loss
    loss = propag.get_loss(distance_3D=dist, frequency=freq, shadow=shad)

    # Plot with shadowing loss
    plt.plot(dist, loss, linewidth=1.0)
    plt.xlabel('Distance [m]')
    plt.ylabel('Loss [dB]')
    plt.title('No shadow loss')
    plt.grid()
    plt.xlim((0, 100))
    plt.ylim((np.min(loss), np.max(loss)))
    plt.show()

    # With shadowing
    shad = True
    # Calculate loss
    loss = propag.get_loss(distance_3D=dist, frequency=freq, shadow=shad)

    # Plot with shadowing loss
    plt.plot(dist, loss, linewidth=1.0)
    plt.xlabel('Distance [m]')
    plt.ylabel('Loss [dB]')
    plt.title('With shadow loss')
    plt.grid()
    plt.xlim((0, 100))
    plt.ylim((np.min(loss), np.max(loss)))
    plt.show()
