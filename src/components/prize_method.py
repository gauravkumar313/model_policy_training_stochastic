import numpy as np
from numpy import pi, sin, sign
from normalize_linear_potential import normalize_linear_potential
import math


class prize_method:

    def __init__(self, phi, max_required_step):
        self.phi = phi
        self.max_required_step = max_required_step

        if max_required_step <= 0:
            raise ValueError("Argument max_required_step must have a positive value.")

        self.__prize_method = None
        self.__vectorized_prize_method = None
        self.optimum_radius = None
        self.optimum_value = None

        self.__initialize()

    def __initialize(self):
        self.__prize_method = self.__prize_method_factory(self.phi, self.max_required_step)
        self.__vectorized_prize_method = np.vectorize(self.__prize_method)
        self.optimum_radius = self.__calculate_performance_diameter(self.phi, self.max_required_step)
        self.optimum_value = self.__prize_method(self.optimum_radius)

    def __prize_method_factory(self, phi, max_required_step):
        def prize_method(radius):
            if radius < max_required_step:
                return 1
            else:
                return np.exp(-phi * (radius - max_required_step) ** 2)

        return prize_method

    def __vectorized_prize_method(self, radii):
        return self.__vectorized_prize_method(radii)

    def __calculate_performance_diameter(self, phi, max_required_step):
        return np.sqrt(1 / phi) + max_required_step

    def prize_method(self, reward):
        return self.__prize_method(reward)

    def __angle_change_factory(self, opt_value, max_value, min_value):
        """
            Returns a function that transforms values from the range [min_value, max_value] to a new range
            such that the optimal value opt_value maps to the midpoint between min_value and max_value,
            and values close to opt_value are scaled down more than values further away.
            """
        # Calculate the radius of the interval around opt_value where values are scaled down more
        opt_radius = abs(opt_value - min_value)

        def transform(x):
            """
            Transforms a value x to a new value that maps to the same relative position within
            the range [min_value, max_value], but with values close to opt_value scaled down more.
            """
            if abs(x - opt_value) <= opt_radius:
                # Scale down values within the interval around opt_value
                return x * abs(min_value - max_value) / (2 * opt_radius)
            else:
                # Scale down values outside the interval using an exponential function
                exp = (2 - opt_radius) / (2 - abs(min_value - max_value))
                scale = (2 - abs(min_value - max_value)) / (2 - opt_radius) ** exp
                return np.sign(x - opt_value) * (abs(min_value - max_value) / 2 +
                                                 scale * abs(x - opt_value) ** exp)

        return transform



    def __calculate_performance_diameter(self, phi, max_required_step):
        phi_mod = math.fmod(phi, 2 * math.pi)
        sin_phi = math.sin(phi_mod)

        opt = max(abs(sin_phi), max_required_step)
        if sin_phi < 0:
            opt *= -1

        return opt

    def __prize_method_factory(self, phi_mod, max_required_step):
        nlgp = polar_nlgp_factory()
        phi_mod = np.mod(phi_mod, 2 * np.pi)
        opt_rad = self.__calculate_performance_diameter(phi_mod, max_required_step)
        min_rad = nlgp.find_minimum_radius(phi_mod)
        angle_change = self.__angle_change_factory(opt_rad, min_rad)
        polar_nlgp = nlgp.calculate_polar_nlgp(phi_mod, angle_change)
        return polar_nlgp