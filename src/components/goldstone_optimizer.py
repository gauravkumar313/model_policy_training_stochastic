from calculate_effec import CalculateEffec
from goldstone_env import goldstoneEnv as env

from sortedcontainers import SortedDict
import numpy as np
import random
import math


class GoldstoneOptimizer(object):

    def __init__(self, setpoint=50, stationary_setpoint=True,inital_seed=None):

        np.random.seed(inital_seed)


        self.maximum_req_step_parameter = math.sin(math.radians(15))
        self.shift_PointDependency_factor = 0.02
        multiple_shift_PointDependency_factor = 100.0 * self.shift_PointDependency_factor

        self.shift_bound_factor = 1.5
        doubled_shift_bound_factor = 2.0 * self.shift_bound_factor
        self.shift_scale_factor = multiple_shift_PointDependency_factor + doubled_shift_bound_factor

        self.cost_reward_cumulative = 1.0
        self.cost_reward_factor = 3.0

        self.cost_reward_group_size =  25.0

        self.stationary_setpoint = stationary_setpoint
        stationary_setpoint_val = 24
        maximum_req_step_parameter_val = self.maximum_req_step_parameter/2.0

        self.gsEnvironment = env(stationary_setpoint_val, self.maximum_req_step_parameter, maximum_req_step_parameter_val)

        self.system_state = SortedDict()

        self.system_state['curr_op_cost'] = 0
        self.system_state['hide_gain'] = 0.0
        self.system_state['oper_cost'] = 0
        self.system_state['fatigue_bif'] = 0.0
        self.system_state['op_cost_buffr'].fill(0)
        self.system_state['hide_eff'] = 0.0
        self.system_state['op_cost_buffr'] = np.empty(10)
        self.system_state['hide_vel'] = 0.0

        # Define goldstone variables
        gs_phi_idx_val = 0
        gs_domain_val = self.gsEnvironment._dynamics.Domain.positive.value
        ve_val = 0.0

        gs_sys_resp_val = self.gsEnvironment._dynamics.System_Response.advantageous.value
        MC_val = 0.0
        ge_val = 0.0

        # Assign to system_state
        self.system_state = {'miscal_domain': gs_domain_val, 'miscal_sys_response': gs_sys_resp_val,
                             'miscal_phi_idx': gs_phi_idx_val, 'eff_gain': ge_val, 'eff_vel': ve_val, 'miscal': MC_val}

        self.observable_keys = ['setPoint', 'velocity', 'g', 'h', 'f', 'consumption', 'cost', 'reward']
        self.system_state.update(
            {'setPoint': setpoint, 'velocity': 50.0, 'g': 50.0, 'h': 50.0, 'f': 0.0, 'consumption': 0.0, 'cost': 0.0, 'reward': 0.0})

        self.initialize = True
        self.generateSequence()
        self.step([0, 0, 0])

    def get_visible_state(self):
        return [item for key in self.observable_keys for item in self.system_state[key].ravel()]



    def step(self,change):
        self.set_new_set_point()
        self.addAction(change)
        self.update_tiredness()
        self.renew_price_function()
        self.calculate_operational_cost()
        self.change_goldstone()
        self.change_function_prices()
        self.update_cost_and_price()

    def set_new_set_point(self):
        if self.stationary_setpoint:
            return
        else:
            if self.set_point_step == self.set_point_length:
                self.generateSequence()

            new_setpoint = self.system_state['setPoint'] + self.set_point_change
            if new_setpoint < 0 or new_setpoint > 100:

                if np.random.rand() > 0.5:
                    self.set_point_change = self.set_point_change * (-1)

            if new_setpoint < 0:
                new_setpoint = 0
            elif new_setpoint > 100:
                new_setpoint = 100

            self.system_state['setPoint'] = new_setpoint

            self.set_point_step = self.set_point_step + 1

    def add_action_method(self,change):
        if change > 1:
            change = 1
        elif change < -1:
            change = -1

        vel = self.system_state['velocity'] + change[0]
        if vel < 0:
            self.system_state['velocity'] = 0.0
        elif vel > 100:
            self.system_state['velocity'] = 100.0
        else:
            self.system_state['velocity'] = vel

        g = self.system_state['g'] + 10 * change[1]
        self.system_state['g'] = max(min(g, 100.), 0.)

        MAX_H = 100.0
        MIN_H = 0.0
        MAX_REQ_STEP_PARAM = 1.0
        MIN_REQ_STEP_PARAM = 0.0
        SCALE_FACTOR = 0.9

        change_h = ((MAX_REQ_STEP_PARAM / SCALE_FACTOR) * 100.0 / self.shift_scale_factor) * change[2]
        new_h = self.system_state['h'] + change_h

        if new_h > MAX_H:
            self.system_state['h'] = MAX_H
        elif new_h < MIN_H:
            self.system_state['h'] = MIN_H
        else:
            self.system_state['h'] = new_h

        h = self.system_state['h']
        set_point = self.system_state['setPoint']
        hide_eff = np.clip(
            self.shift_scale_factor * h / 100. - self.shift_PointDependency_factor * set_point - self.shift_bound_factor,
            -self.shift_bound_factor, self.shift_bound_factor)
        self.system_state['hide_eff'] = hide_eff

    def setEffectiveAction(self, effectiveAction_gain, effectiveAction_velocity):
        self.system_state['eff_gain'] = effectiveAction_gain
        self.system_state['eff_vel'] = effectiveAction_velocity

    def update_tiredness(self):
        # Exponential factor for fatigue accumulation
        fatigue_multiplier_start = 1.2
        max_allow_tolerance = 0.05
        fatigue_multiplier = 1.1
        fatigue_lambda = 0.1
        max_fatigue_multiplier = 5.0

        dynamic_parameter = 0.0

        vel = self.system_state['velocity']
        g = self.system_state['g']
        sp = self.system_state['setPoint']

        velocityHidden = self.system_state['hide_vel']


        gainHidden = self.system_state['hide_gain']

        effectiveAction = EffectiveAction(vel, g, sp)
        effectiveAction_gain = effectiveAction.getEffectiveGain()

        effectiveAction_velocity = effectiveAction.getEffectiveVelocity()
        self.setEffectiveAction(effectiveAction_gain, effectiveAction_velocity)

        noise_value_exponential = -np.log(1-np.random.uniform())/fatigue_lambda

        noise_generate_exponential = -np.log(1-np.random.uniform())/fatigue_lambda
        noiseUniformGain = random.rand()
        noiseUniformVelocity = random.rand()

        noiseBinomialGain = int(random.random() < max(min(effectiveAction_gain, 0.999), 0.001))
        noiseBinomialVelocity = int(random.random() < max(min(effectiveAction_velocity, 0.999), 0.001))


        n_v = 2.0 * (1.0/(1.0+np.exp(-noise_value_exponential)) - 1.0/2.0)


        n_g = 2.0 * (1.0/(1.0+np.exp(-noise_generate_exponential)) - 1.0/2.0)

        n_g = n_g + (1-n_g) * noiseUniformGain * noiseBinomialGain * effectiveAction_gain
        n_v = n_v + (1-n_v) * noiseUniformVelocity * noiseBinomialVelocity * effectiveAction_velocity

        if max_allow_tolerance >= effectiveAction_gain:
            new_gain = effectiveAction_gain
        elif gainHidden >= fatigue_multiplier_start:
            new_gain = min(max_fatigue_multiplier, fatigue_multiplier * gainHidden)
        else:
            new_gain = 0.9 * gainHidden + n_g / 3.0
        gainHidden = new_gain

        if max_allow_tolerance >= effectiveAction_velocity:
            velocityHidden = effectiveAction_velocity
        elif fatigue_multiplier_start >= velocityHidden:
            velocityHidden = min(max_fatigue_multiplier, fatigue_multiplier * velocityHidden)
        else:
            velocityHidden = 0.9 * velocityHidden +  (n_v / 3.0 if n_v is not None else 0)

        if max(velocityHidden, gainHidden) == max_fatigue_multiplier:
            alpha = 1.0 / (1.0 + np.exp(-random.uniform(1.8, 3.0)))
        else:
            alpha = max(n_v, n_g)

        fatigue_bif = max(0,(( 30000.0/(( 5*vel )+100 ))-0.01*( g ** 2 )))
        self.system_state['hidden_velocity'] = velocityHidden
        self.system_state['hidden_gain'] = gainHidden
        self.system_state['fatigue_factor'] = (fatigue_bif*(1+2*alpha)) / 3.
        self.system_state['fatigue'] = fatigue_bif

    def renew_price_function(self):

        profit_price = 2.50

        speed_price = 4.0

        setpoint_price = 2.0

        sp = self.system_state['setPoint']

        g = self.system_state['g']

        vel = self.system_state['velocity']

        setpoint_price_p = setpoint_price * sp
        profit_price_gain = profit_price * g
        speed_price_vel = speed_price * vel
        prices = setpoint_price_p + profit_price_gain + speed_price_vel
        normalized_costs = prices / 100.0
        o = np.exp(normalized_costs)
        self.system_state['curr_op_cost'] = o

        if self.initialize:
            self.system_state['op_cost_buffr'][:-1] = self.system_state['op_cost_buffr'][1:]
            self.system_state['op_cost_buffr'][-1] = o
        else:
            self.system_state['op_cost_buffr'] += o
            self.initialized = True

    def calculate_operational_cost(self):
        kernel = [0.11111, 0.22222, 0.33333, 0.22222, 0.11111, 0., 0., 0., 0., 0.]
        op_cost_buffr = self.system_state['op_cost_buffr']
        convolved = [sum(kernel[i] * op_cost_buffr[max(i - j, 0)] for i in range(len(kernel)))
                     for j in range(len(op_cost_buffr))]
        self.system_state['oper_cost'] = sum(convolved)

    def change_goldstone(self):
        miscal_domain = self.system_state['miscal_domain']


        effective_shift = self.system_state['hide_eff']

        system_response = self.system_state['miscal_sys_response']


        miscal_phi_index = self.system_state['miscal_phi_idx']

        result = self.gsEnvironment.system_condition_change(self.gsEnvironment._dynamics.Domain(miscal_domain),
                                                            miscal_phi_index,
                                                            self.gsEnvironment._dynamics.System_Response(
                                                                system_response), effective_shift)
        self.system_state['miscal'] = -result[0]
        self.system_state['miscal_domain'] = result[1].value
        self.system_state['miscal_sys_response'] = result[2].value
        self.system_state['miscal_phi_idx'] = result[3]

    def change_function_prices(self):
        miscal = self.system_state['miscal']
        oper_cost = self.system_state['oper_cost']
        e_new_hidden = oper_cost - (self.cost_reward_group_size * (miscal - 1.0))
        c = e_new_hidden - np.random.randn() * (1 + 0.005 * e_new_hidden)
        new_system_state = self.system_state.copy()
        new_system_state['consumption'] = c

    def update_state(self, cost):
        self.state['cost'] = cost
        self.state['reward'] = -cost

    def update_cost_and_price(self):

        c = self.system_state['consumption']

        tiredness = self.system_state['f']

        cost_reward_consumption = self.cost_reward_cumulative * c


        tiredness_factor = self.cost_reward_factor * tiredness
        cost = tiredness_factor + cost_reward_consumption

        self.update_state(cost)

    def generateSequence(self):
        self.set_point_step = 0

        self.set_point_length = np.random.randint(1, 100)
        set_point_ch = 2 * np.random.rand() - 1
        if np.random.rand() < 0.10:
            set_point_ch = 0.0
        self.set_point_change =  set_point_ch

