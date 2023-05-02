from scipy.special import cbrt

from numpy import sqrt, sin, cos, arcsin, arccos, pi

import numpy as np

import math


class normalize_linear_potential:

    def __init__(self):
        u = math.cbrt(1 + math.sqrt(2)) / math.sqrt(3)
        r_initial = u + 1 / (3 * u)

        lmbd = 2 * r_initial ** 2 - r_initial ** 4 + 8 * math.sqrt(2 / 27.) * r_initial

        self.alpha_normalize = 2 / lmbd
        self.beta_normalize = 1 / lmbd
        self.kappa_normalize = -8 * sqrt(2 / 27.0) / lmbd
        self.phi = pi / 4.0
        self.qh_b_param = - math.sqrt(1 / 27.0)

    def calculate_polar_nlgp(self, r, phi):
        """Calculates the negative log-gaussian process for a given set of polar coordinates."""
        rsq = np.square(r)
        alpha_norm = -self.alpha_normalize * rsq
        beta_norm = self.beta_normalize * np.square(rsq)
        kappa_norm = self.kappa_normalize * np.sin(phi) * r
        return alpha_norm + beta_norm + kappa_norm


    def generic_lowest_radius(self, phi):
        radii = np.zeros_like(phi)

        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                radii[i, j] = self.__global_minimum_radius(phi[i, j])

        return radii