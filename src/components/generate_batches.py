import numpy as np

data_history = 15 # Hyperparameters

# Batch data generation functionality

from goldstone_optimizer import GoldstoneOptimizer

setpoints_list = range(10,110,10)

# constants
num_episodes = 10
num_train_episodes = 7
num_test_episodes = 3
num_steps = 1000

input_dim = 15 * (3 + 1) + 1
output_dim = 1

counts = setpoints_list.__len__()
updated_data_history_train = counts*num_train_episodes*(num_steps-(data_history+5))
updated_data_history_test = counts*num_test_episodes*(num_steps-(data_history+5))

X_instruct_data = np.empty(updated_data_history_train, input_dim)
X_instruct_data.fill(0)

Y_instruct_data = np.empty(updated_data_history_train, output_dim)
Y_instruct_data.fill(0)

X_trial_data = np.empty(updated_data_history_test, input_dim)
X_trial_data.fill(0)

Y_trial_data = np.empty(updated_data_history_test, output_dim)
Y_trial_data.fill(0)

idx_instruct = 0
idx_trial = 0
for index in np.arange(0, num_episodes + 1, 1):
    for setpoint in setpoints_list:
        environment = GoldstoneOptimizer(setpoint)
        statuses = np.empty((num_steps,environment.get_visible_state().shape[0]))
        statuses.fill(0)

        for inner_idx in np.arange(0, num_steps + 1, 1):
            environment.step(2*np.random.rand(3)-1)
            statuses[inner_idx, : ] = environment.get_visible_state()

            if inner_idx >= data_history + 5:
                accomplishment = np.concatenate((statuses[inner_idx - (data_history - 1):inner_idx + 1, 1:4],
                                       statuses[inner_idx - data_history:inner_idx, -2:-1]), axis=1)

                accomplishment = np.concatenate((np.array([setpoint]), accomplishment.ravel()))


                if index < num_train_episodes:
                    X_instruct_data[idx_instruct] = accomplishment
                    Y_instruct_data[idx_instruct] = statuses[inner_idx,-2:-1]
                    idx_instruct += 1
                else:
                    X_trial_data[idx_trial] = accomplishment
                    Y_trial_data[idx_trial] = statuses[inner_idx,-2:-1]
                    idx_trial += 1


output_directory =  'data_output/'
np.savetxt(output_directory + 'X_train_data.txt',X_instruct_data,fmt='%5.4f')
np.savetxt(output_directory + 'Y_train_data.txt',Y_instruct_data,fmt='%5.4f')
np.savetxt(output_directory + 'X_test_data.txt',X_trial_data,fmt='%5.4f')
np.savetxt(output_directory + 'Y_test_data.txt',Y_trial_data,fmt='%5.4f')
