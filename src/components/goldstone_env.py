from systemdyn import sd

class goldstoneEnv:
    def __init__(self, amount_stairs, largest_necessary_step, shield_sector):
        self._dynamics = sd(amount_stairs, largest_necessary_step, shield_sector)

    def system_condition_change(self, realm, miscal_phi_index, entity_res, location):
        realm, miscal_phi_index, entity_res = self._dynamics.state_transition(realm, miscal_phi_index, entity_res, location)
        return self.get_reward(miscal_phi_index, location), realm, miscal_phi_index, entity_res

    def get_reward(self, miscal_phi_index, location):
        return self._dynamics.get_reward(miscal_phi_index, location)