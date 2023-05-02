from AlphaWrapper import AlphaWrapper
import random
import pickle

import sys

import numpy as np

initial_seed = 3
random.seed(initial_seed)

# generate random number
random_number = random.random()

model_directory = '../../model/data/'

# Load data from files
def load_data():
    X_instruct_data = np.loadtxt(model_directory + 'X_instruct_data.txt')
    Y_instruct_data = np.loadtxt(model_directory + 'Y_instruct_data.txt')[:, None]
    X_trial_data = np.loadtxt(model_directory + 'X_trial_data.txt')
    Y_trial_data = np.loadtxt(model_directory + 'Y_trial_data.txt')[:, None]
    return X_instruct_data, Y_instruct_data, X_trial_data, Y_trial_data

X_instruct_data, Y_instruct_data, X_trial_data, Y_trial_data = load_data()

parameters_job = {
    'seed': initial_seed,
    'history': 15,
    'state_dim': X_instruct_data.shape[1],
    'r_dim': Y_instruct_data.shape[1]
}

state_dim = parameters_job['state_dim']
tar_dim = parameters_job['r_dim']


# Set parameters for the model
mode = 'AlphaWrapper'
saved = False
epochs = 3000
batchsize = int(X_instruct_data.shape[0] / 25)
alpha = float(sys.argv[1])
learn_rate = np.float32(0.001)
samples = 50
dimh = 75
graph = [parameters_job['state_dim'], dimh, dimh, 1]

parameters_prototype = {
    'mode': mode,
    'saved': saved,
    'epochs': epochs,
    'batchsize': batchsize,
    'alpha': alpha,
    'learn_rate': learn_rate,
    'samples': samples,
    'dimh': dimh,
    'graph': graph
}

X_data = X_instruct_data
Y_data = X_instruct_data



# Reshape X_data to get At and Rt arrays
X_data1 = X_data[:, 1:].reshape((X_data.shape[0], parameters_job['history'], 4))

# Split At_Rt into At (actions) and Rt (rewards)
At = X_data1[:, :, :3]  # Get first 3 columns as actions
Rt = X_data1[:, :, 3:4]  # Get last column as rewards


# Calculate log-scaled minimum and maximum of Rt and Y_data
a_x = np.log(1 + np.min(Rt, axis=0) * 0.95)
a_y = np.log(1 + np.min(Y_data, axis=0) * 0.95)
b_x = np.log(1 + np.max(Rt, axis=0) * 1.05)
b_y = np.log(1 + np.max(Y_data, axis=0) * 1.05)

# Log-scale Rt and Y_data
Rt = np.log(1 + Rt)
Y_data = np.log(1 + Y_data)

from scipy.special import logit

# Apply logit transformation
Rt = logit((Rt - a_x) / (b_x - a_x))
Y_data = logit((Y_data - a_y) / (b_y - a_y))

# Update X_data with the transformed Rt
X_data1[:, :, 3:4] = Rt
X_data = np.hstack((X_data[:, 0:1], X_data1.reshape((X_data1.shape[0], parameters_job['history']*4))))

# Update bounds in parameters_prototype
parameters_prototype['bounds'] = [a_x, a_y, b_x, b_y]


Y_data[:,0] = Y_data[:,0] - X_data[:,-1]

model = AlphaWrapper(parameters_prototype,parameters_job,X_data,Y_data)
model.train()



model_norm = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]
model_weights = model.get_weights()
saved = {'params_model': parameters_prototype,
         'params_task': parameters_job,
         'model_norm': model_norm,
         'model_weights': model_weights}

pickle_directory = '../../model/'
output_string = parameters_prototype['mode'] + '_' + str(parameters_prototype['alpha'])  + '.p'
with open(pickle_directory + output_string, 'wb') as f:
    pickle.dump(saved, f)

