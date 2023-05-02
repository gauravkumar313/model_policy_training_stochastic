
import numpy as np


class CalculateEffec(object):

    def __init__(self, speed, profit, sp):
        self.effec_Profit = self.calcEffec_Profit(profit, sp)

        self.effectiveSpeed = self.calcEffectiveSpeed(speed, profit, sp)

        self.sp = sp

    def calc_effective_speed(a, b, sp):
        def calc_effective_speed_unscaled(eff_a, eff_b):
            return eff_a * (1 - eff_b * (1 - eff_a))

        def calc_effective_a(a_pct, sp):
            return a_pct * (0.02 + 0.98 * (sp / 100))

        def calc_effective_b(b_pct, sp):
            return b_pct * (0.02 + 0.98 * (1 - sp / 100))

        min_alpha_unscaled = calc_effective_speed_unscaled(calc_effective_a(100, sp), calc_effective_b(0, sp))
        max_alpha_unscaled = calc_effective_speed_unscaled(calc_effective_a(0, sp), calc_effective_b(100, sp))
        alpha_unscaled = calc_effective_speed_unscaled(calc_effective_a(a, sp), calc_effective_b(b, sp))
        return (alpha_unscaled - min_alpha_unscaled) / (max_alpha_unscaled - min_alpha_unscaled)

    def calculate_effective_profit(self, beta, setpoint):
        effective_b = self.calculate_effective_b(beta, setpoint)
        min_beta_unscaled = self.calcEffec_ProfitUnscaled(self.calcEffectiveB(100, setpoint))
        max_beta_unscaled = self.calcEffec_ProfitUnscaled(self.calcEffectiveB(0, setpoint))
        beta_unscaled = self.calcEffec_ProfitUnscaled(effective_b)
        return (beta_unscaled - min_beta_unscaled) / (max_beta_unscaled - min_beta_unscaled)


    def calculate_effec_alpha(self, action_value, setpoint_value):
        effective_a = action_value + 101.0 - setpoint_value
        return effective_a

    def calculate_effec_beta(self, b, sp):
        effective_b = b + 1.0 + sp
        return effective_b

    def compute_effec_speed_nonnscaled(self, effec_alpha, effec_beta):
        if effec_alpha == 0:
            return 0
        return (effec_beta + 1) / effec_alpha

    def compute_effective_profit_nonscaled(self, effec_beta):
        return 1 / effec_beta

    def fetch_effec_speed(self):
        return self.effec_speed

    def fetch_effec_profit(self):
        return self.effec_profit