import numpy as np
from src.components.goldstone_optimizer import GoldstoneOptimizer

import theano
import lasagne

from scipy.special import logit
import pickle as pickle
import theano.tensor as T

def create_policy_network(input_var=None):
    input_layer = lasagne.layers.InputLayer(shape=(None, 61), input_var=input_var)
    hidden_layer1 = lasagne.layers.DenseLayer(input_layer, num_units=20, nonlinearity=lasagne.nonlinearities.rectify)
    hidden_layer2 = lasagne.layers.DenseLayer(hidden_layer1, num_units=20, nonlinearity=lasagne.nonlinearities.rectify)
    output_layer = lasagne.layers.DenseLayer(hidden_layer2, num_units=3, nonlinearity=lasagne.nonlinearities.tanh)
    return output_layer

policy_net = create_policy_network()
input_var = T.matrix()
get_policy_output = theano.function(inputs=[input_var], outputs=lasagne.layers.get_output(policy_net, input_var))



def predict_state_reward(state, normalization_params, bounds, policy_net):
    reward = np.log(1 + (state[:, -1:]))

    lower_bound = bounds[0]
    upper_bound = bounds[2]

    eps = 1e-5

    reward = np.clip(reward, lower_bound + eps, upper_bound - eps)

    reward = logit((reward - lower_bound) / (upper_bound - lower_bound))

    state[:, -1:] = reward

    state_norm = np.zeros((1, 61))
    state_norm[0, :] = np.hstack((state[0, 0], state[:, [1, 2, 3, -1]].ravel()))

    state_norm = (state_norm - normalization_params[0]) / normalization_params[1]

    return np.round(policy_net(state_norm)[0, :], 4)



def evaluate_policy(controller_file, do_warmup=True):
    # Load controller weights and normalization parameters
    with open(controller_file, 'rb') as f:
        controller_data = pickle.load(f)

    controller_weights = controller_data['controller_weights']
    model_norm = controller_data['model_norm']
    bounds = controller_data['params_model']['bounds']

    np.random.seed(1)

    n_evals = 25
    setpoints = range(10, 110, 10)
    history = 15

    data = np.zeros((len(setpoints), n_evals, 100))

    for i, setpoint in enumerate(setpoints):
        for j in range(n_evals):
            env = GoldstoneOptimizer(setpoint)

            if do_warmup:
                burnin = 15 + np.random.randint(50)
                state = np.zeros((burnin, env.visibleState()[:-1].shape[0]))
                for t in range(burnin):
                    action = 2 * np.random.rand(3) - 1
                    env.step(action)
                    state[t, :] = env.visibleState()[:-1]
            else:
                state = env.visibleState()[:-1].reshape((1, -1))
                state = np.tile(state, [history, 1])

            trajectory = np.zeros(100)

            for t in range(100):
                action = predict(np.copy(state), model_norm, bounds)
                env.step(action)
                state = np.vstack((state[1:, :], env.visibleState()[:-1]))
                trajectory[t] = env.visibleState()[-2]

            data[i, j, :] = trajectory

    mean_reward = np.mean(data)
    print(f"{controller_file}: {mean_reward}")

    return data


results = evaluate_policy('AlphaWrapper_0.5_0.5.p')


