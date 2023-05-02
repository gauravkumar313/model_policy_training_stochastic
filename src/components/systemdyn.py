import numpy as np
from prize_method import reward_function
from enum import Enum
from numpy import pi, sign


class sd:
    class SystemDynamics_Reply(Enum):
        favourable = +1
        unfavourable = -1

    class Realm(Enum):
        reject = -1
        optimistic = +1

    def __init__(self, amount_stairs, high_necessary_stair, shield_area):
        self._shield_area = self._check_shield_area(shield_area)
        self._highest_punishment_modulus_index = self.calc_strongest_punishment_absIdx(amount_stairs)
        self._punishment_functions_array = self._define_prize_method_functions(amount_stairs, high_necessary_stair)

    def _check_shield_area(self, shield_area):
        if (shield_area < 0):
            raise ValueError('shield_area has to be neutral.')
        return shield_area

    def start_over(self):
        return self.Realm.optimistic, 0, self.SystemDynamics_Reply.favourable

    def prize_method(self, miscal_phi_index, location):
        return self.get_punishment_method(miscal_phi_index).prize_method(location)

    def condition_change(self, realm, miscal_phi_index, systemDynamics_reply, location):

        old_realm = realm

        realm = self._calc_realm(old_realm, location)


        if realm != old_realm:
            systemDynamics_reply = self.SystemDynamics_Reply.favourable


        miscal_phi_index += self._calc_angular_step(realm, miscal_phi_index, systemDynamics_reply, location)


        systemDynamics_reply = self._changed_systemDynamics_reply(miscal_phi_index, systemDynamics_reply)


        miscal_phi_index = self._apply_uniformity(miscal_phi_index)

        if (miscal_phi_index == 0) and (abs(location) <= self._shield_area):
            realm, miscal_phi_index, systemDynamics_reply = self.start_over()

        return realm, miscal_phi_index, systemDynamics_reply

    def _calc_realm(self, realm, location):
        if abs(location) <= self._shield_area:
            return realm
        else:
            return self.Realm(sign(location))

    def _calc_angular_step(self, realm, miscal_phi_index, systemDynamics_reply, location):
        if abs(location) <= self._shield_area:
            return -sign(miscal_phi_index)

        if miscal_phi_index == -realm.value * self._highest_punishment_modulus_index:
            return 0
        else:
            return systemDynamics_reply.value * sign(location)

    def _changed_systemDynamics_reply(self, miscal_phi_index, systemDynamics_reply):
        if abs(miscal_phi_index) >= self._highest_punishment_modulus_index:
            return self.SystemDynamics_Reply.unfavourable
        return systemDynamics_reply

    def _apply_uniformity(self, miscal_phi_index):
        punishment_range = 4 * self._highest_punishment_modulus_index
        if abs(miscal_phi_index) < self._highest_punishment_modulus_index:
            return miscal_phi_index
        miscal_phi_index = (miscal_phi_index + punishment_range) % punishment_range
        miscal_phi_index = 2 * self._highest_punishment_modulus_index - miscal_phi_index
        return miscal_phi_index

    def get_punishment_method(self, miscal_phi_index):
        # Compute the index of the punishment function to use
        idx_offset = int(self._highest_punishment_modulus_index + miscal_phi_index)
        idx = idx_offset % len(self._punishment_functions_array)

        # Return the selected punishment function
        return self._punishment_functions_array[idx]

    def _define_prize_method_functions(self, amount_stairs, high_necessary_stair):
        angles = np.arange(-self._highest_punishment_modulus_index,
                           self._highest_punishment_modulus_index + 1) * 2 * np.pi / amount_stairs
        reward_funcs = [reward_function.reward_function(angle, high_necessary_stair) for angle in angles]
        return np.array(reward_funcs)

    def calc_strongest_punishment_absIdx(self, amount_stairs):
        if (amount_stairs < 1) or (amount_stairs % 4 != 0):
            raise ValueError('Number of steps must be a positive integer and multiple of four.')
        return amount_stairs // 4

        try:
            amount_stairs = int(input("Enter the number of stairs: "))
            print(f"The highest punishment modulus index is {highest_punishment_modulus_index(amount_stairs)}")
        except ValueError as e:
            print(e)