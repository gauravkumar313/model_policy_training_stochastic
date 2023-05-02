import lasagne

import theano
import lasagne.utils as utils

import theano.tensor as T
import numpy as np
from lasagne.updates import total_norm_constraint


from collections import OrderedDict



class AdamPolicySearchAlpha:
    def __init__(self, params_dict, params_task_dict, X_data, model_net, policy_net):
        self.rng = np.random.RandomState()

        self.model_net = model_net
        self.policy_net = policy_net

        self.params_dict = params_dict
        self.params_task_dict = params_task_dict

        x_data = T.matrix('x_data')
        cost = self.compute_cost(x_data)

        self.fwpass = theano.function(inputs=[x_data], outputs=cost, allow_input_downcast=True)

        all_params = lasagne.layers.get_all_params(self.policy_net, trainable=True)
        updates = self.adam(cost, all_params, learning_rate=params_dict['learning_rate'])
        self.train_func = theano.function(inputs=[x_data], outputs=[cost], updates=updates)

        self.predict_func = theano.function(inputs=[x_data], outputs=self.predict(x_data))

    def control_method(self, st):
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        n = self.samples
        st_s = T.tile(st, [n, 1])

        onoise = srng.normal(size=(st_s.shape[0], 1, self.T))
        inoise = T.sqrt(st.shape[1]) * srng.normal(size=(n, st.shape[0], self.T))

        def step(x_t, onoise_t, inoise_t):
            return self.step_function(x_t, 0, onoise_t, inoise_t)[2]

        R, _ = theano.scan(fn=step, sequences=[st_s, onoise, inoise])
        return R.mean()

    def step_function(state, time_step, observation_noise, input_noise):
        obs_noise_t = observation_noise[:, :, time_step]
        in_noise_t = input_noise[:, :, time_step:time_step + 1]

        # get action
        action = self.predict(state)

        a_t1 = self.action_function(state, action)

        state_t3 = state[:, 1:].reshape((state.shape[0], self.task_params['history'], 4))
        state_t3_t1 = T.set_subtensor(state_t3[:, :, :3],
                                      T.concatenate((state_t3[:, 1:, :3], T.shape_padaxis(a_t1, 1)), axis=1))
        x_t1 = T.concatenate((state[:, :1], state_t3_t1.reshape((state.shape[0], state.shape[1] - 1))), axis=1)

        x_t1 = x_t1.reshape(
            (self.params['samples'], T.cast(x_t1.shape[0] / self.params['samples'], 'int64'), x_t1.shape[1]))
        delta_rt1, var_delta_rt1 = self.model.predict(x_t1, mode='symbolic', provide_noise=True, noise=in_noise_t)
        delta_rt1 = delta_rt1.reshape((delta_rt1.shape[0] * delta_rt1.shape[1], delta_rt1.shape[2]))
        var_delta_rt1 = var_delta_rt1.reshape((var_delta_rt1.shape[0] * var_delta_rt1.shape[1], var_delta_rt1.shape[2]))

        delta_rt1 = obs_noise_t * T.sqrt(var_delta_rt1) + delta_rt1

        r_t1 = state[:, -1:] + delta_rt1[:, 0:1]

        reward = 1. / (1. + T.exp(-r_t1))  # undo logit
        reward = reward * (self.model.params['bounds'][3] - self.model.params['bounds'][1]) + \
                 self.model.params['bounds'][1]
        reward = T.exp(reward) - 1

        state_t3_t1 = T.set_subtensor(state_t3_t1[:, :, 3:],
                                      T.concatenate((state_t3_t1[:, 1:, 3:], T.shape_padaxis(r_t1, 1)), axis=1))
        state_t1 = T.concatenate((state[:, :1], state_t3_t1.reshape((state.shape[0], state.shape[1] - 1))), axis=1)

        return [state_t1, time_step + 1, reward[:, 0]]

    def do_action(states, deltas):
        max_required_step = np.sin((15. / 180) * np.pi)
        gs_bound = 1.5
        gs_setpoint_dependency = 0.02
        gs_scale = 2.0 * gs_bound + 100. * gs_setpoint_dependency
        shift_step = (max_required_step / 0.9) * 100. / gs_scale

        a_0 = states[:, -4:-1]

        v_n = a_0[:, 0] + 1.0 * deltas[:, 0]  #
        g_n = a_0[:, 1] + 10. * deltas[:, 1]
        s_n = a_0[:, 2] + shift_step * deltas[:, 2]

        # velocity
        a_0 = np.hstack([np.clip(v_n, 0., 100.).reshape(-1, 1), a_0[:, 1:]])
        # gain
        a_0 = np.hstack([a_0[:, 0].reshape(-1, 1), np.clip(g_n, 0., 100.).reshape(-1, 1), a_0[:, 2:]])
        # shift
        a_0 = np.hstack([a_0[:, :2], np.clip(s_n, 0., 100.).reshape(-1, 1)])

        return a_0

    def predict_output(self, X):
        X_normalized = (X - self.model.mean_X.astype(theano.config.floatX)) / self.model.std_X.astype(
            theano.config.floatX)
        predicted_output = lasagne.layers.get_output(self.policy, X_normalized)
        return predicted_output

    def adam_optimizer(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        all_gradients = T.grad(cost=loss, wrt=parameters)
        all_gradients = total_norm_constraint(all_gradients, 10)

        gradient_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), all_gradients)))
        not_finite = T.or_(T.isnan(gradient_norm), T.isinf(gradient_norm))

        t_prev = theano.shared(utils.floatX(0.))
        updates = OrderedDict()

        t = t_prev + 1
        alpha_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        for param, grad_t in zip(parameters, all_gradients):
            grad_t = T.switch(not_finite, 0.1 * param, grad_t)
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = beta1 * m_prev + (1 - beta1) * grad_t
            v_t = beta2 * v_prev + (1 - beta2) * grad_t ** 2
            step = alpha_t * m_t / (T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates


