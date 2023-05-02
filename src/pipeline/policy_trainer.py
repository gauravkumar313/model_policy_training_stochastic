import random

import fstr as fstr
import lasagne
import tensorflow as tf


import numpy as np

import sys
sys.path.append('../')

from src.components.AlphaWrapper import AlphaWrapper

import pickle


initial_seed = 3
random.seed(initial_seed)

# generate random number
random_number = random.random()

params_task = TaskParams(task='Professional Standard', initial_seed=0, history=15)


model_directory = '../../model/data/'
X_instruct_data = np.loadtxt(model_directory+'X_instruct_data.txt')
Y_instruct_data = np.loadtxt(model_directory+'Y_instruct_data.txt')[:,None]


# Reshape the input data
X_reshaped = X_instruct_data[:, 1:].reshape((X_instruct_data.shape[0], params_task['history'], 4))
X_reshaped[:, :, -1] = np.hstack((X_reshaped[:, 1:, -1], Y_instruct_data))

# Flatten the reshaped data
X_flattened = X_reshaped.reshape((X_reshaped.shape[0], X_reshaped.shape[1]*X_reshaped.shape[2]))

# Add the first column of the input data to the flattened data
X_start = np.hstack((X_instruct_data[:, :1], X_flattened))

# Update the state and action dimensions
params_task['state_dim'] = X_start.shape[1]
params_task['action_dim'] = 3

from scipy.special import logit

model_dir = ''
alpha = sys.argv[1]

model_params_file = f'AlphaWrapper{alpha}.p'
with open(model_dir + model_params_file, 'rb') as f:
    model_params = pickle.load(f)['params_model']
    model_params['saved'] = model_dir + model_params_file

X1 = X_start[:, 1:].reshape((X_start.shape[0], params_task['history'], 4))
At = X1[:, :, :3]
Rt = X1[:, :, 3:4]
a_x, a_y, b_x, b_y = model_params['bounds']
Rt = np.log(1 + Rt)

a_x = 0
b_x = 1
Rt = np.clip(Rt, a_x + 1e-5, b_x - 1e-5)
Rt = logit((Rt - a_x) / (b_x - a_x))

# reshape X1 and concatenate with X_start
X1_reshaped = X1.reshape(X1.shape[0], -1, 1)
X1_reshaped = np.concatenate((X1_reshaped, Rt), axis=2)
X_start = np.concatenate((X_start[:, :1], X1_reshaped.reshape(X1_reshaped.shape[0], -1)), axis=1)

# load the model
print(model_params['saved'])
model = AlphaWrapper(model_params, params_task, X_start, X_start[:, -1:])
model.loadWeights()

# define the policy search parameters
params_controller = {
    'saved': False,
    'learning_rate': 0.0001,
    'name': 'controller',
    'T': 75,
    'epochs': 750,
    'batchsize': 25,
    'samples': 25
}


def policy_output(input_var=None):
    with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
        l_in = tf.keras.layers.Input(shape=(X_start.shape[1],), tensor=input_var)
        l_hid1 = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(l_in)
        l_hid2 = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(l_hid1)
        l_out = tf.keras.layers.Dense(units=3, activation=tf.nn.tanh)(l_hid2)
        return l_out

    import AdamPolicySearchAlpha
    # Initialize controller
    controller = AdamPolicySearchAlpha(params_controller, params_task, X_start, model, policy())

    # Initialize variables
    errs = []
    trace = []
    time_epochs = []
    start = timer()

    n_batches = len(X) // batch_size
    for i in range(n_batches):
        yield X[i * batch_size:(i + 1) * batch_size]


    for j in range(params_controller['epochs']):
        errs = []
        inds = np.random.permutation(X_start.shape[0])
        for ind in batch_generator(inds, params_controller['batchsize']):
            model.bb_alpha.network.update_randomness(params_controller['samples'])
            e = controller.train_func(X_start[ind])
            errs.append(e)

        end = timer()
        time_e = end - start
        time_epochs.append(time_e)
        atime_e = np.mean(time_epochs[-5:])
        rest_time = int(atime_e * (params_controller['epochs'] - (j + 1)))
        rest_hours, rest_seconds = divmod(rest_time, 60 * 60)
        rest_minutes, _ = divmod(rest_seconds, 60)
        err = np.mean(errs)

        logging.info(f"Remaining: {rest_hours}h:{rest_minutes}m, Policy Cost: {err}")
        trace.append(err)
        start = timer()

        # Get weights
    weights = [p.get_value() for p in lasagne.layers.get_all_params(controller.policy, trainable=True)]

model_norm = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]

saved_data = {
    'model_params': model_params,
    'controller_params': params_controller,
    'task_params': params_task,
    'model_norm': model_norm,
    'trace': time_epochs,
    'controller_weights': wts
}
policy_directory = '/'
pickle.dump(saved_data,open(policy_directory + fstr,'wb'))
