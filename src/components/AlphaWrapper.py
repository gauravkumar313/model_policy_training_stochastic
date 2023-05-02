import pickle as pickle

from alpha_minimize import Alpha_Minimize as alpha_minimize
import theano


import theano.tensor as T

import numpy as np



class AlphaWrapper:

    def __init__(self, model_params, task_params, x_data, y_data):

        self.model_params = model_params
        self.task_params = task_params
        self.x_data, self.y_data = self.normalizeself(x_data, y_data)

        num_train_samples = self.x_data.shape[0]
        self.x_data = T.cast(theano.shared(self.x_data), theano.config.floatX)
        self.y_data = T.cast(theano.shared(self.y_data), theano.config.floatX)

        self.bb_alpha = alpha_minimize(self.model_params['graph'], self.model_params['samples'],
                             self.model_params['alpha'], self.model_params['learn_rate'],
                             1.0, self.model_params['batch_size'], self.x_data, self.y_data,
                             num_train_samples)


    def normalizeself(self, X, Y):
        X_norm = np.empty_like(X)
        Y_norm = np.empty_like(Y)

        # Normalize X
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.  # Avoid division by zero
        for i in range(X.shape[0]):
            X_norm[i, :] = (X[i, :] - X_mean) / X_std

        # Normalize Y
        Y_mean = np.mean(Y, axis=0)
        Y_std = np.std(Y, axis=0)
        Y_std[Y_std == 0] = 1.  # Avoid division by zero
        for i in range(Y.shape[0]):
            Y_norm[i, :] = (Y[i, :] - Y_mean) / Y_std

        return X_norm, Y_norm

    def load_weights(self):
        if self.params['saved'] is not False:
            with open(self.params['saved'], 'rb') as f:
                dat = pickle.load(f)

            weights = dat['model_weights']
            for j, w in enumerate(weights):
                p = self.bb_alpha.network.params[j]

                if p.name not in ['log_v_z_par', 'm_z_par']:
                    p.set_value(np.array(w, dtype=theano.config.floatX))

            if 'model_norm' in dat:
                self.mean_X, self.std_X, self.mean_Y, self.std_Y = dat['model_norm']

    def train_epochs(self, epochs=0):
        num_epochs = epochs or self.params['epochs']
        self.bb_alpha.train(num_epochs)

    def predict_model(self, X_test, mode='numerical', provide_noise=False, noise=None):
        X_norm = (X_test - self.mean_X) / self.std_X

        if mode == 'symbolic':
            output_fn = self.bb_alpha.network.output_gn if provide_noise else self.bb_alpha.network.output
            m = output_fn(X_norm, False, X_norm.shape[0], use_indices=False)
            log_v_noise = self.bb_alpha.network.log_v_noise
            noise_variance = np.tile(np.exp(log_v_noise[0, :]), [m.shape[0], m.shape[1], 1])
        else:
            X_batch = np.tile(X_norm, [self.params['samples'], 1, 1])
            m = self.bb_alpha.fwpass(X_batch, X_norm.shape[0])
            log_v_noise = self.bb_alpha.network.log_v_noise.get_value()[0, :]
            noise_variance = np.tile(np.exp(log_v_noise), [m.shape[0], m.shape[1], 1])

        mean = m * self.std_Y + self.mean_Y
        variance = noise_variance * self.std_Y ** 2

        return mean, variance

    def fetch_weights(self):
        non_log_v_z_par = []
        non_m_z_par = []

        for p in self.bb_alpha.network.params:
            if p.name != 'log_v_z_par':
                non_log_v_z_par.append(p.get_value())
            elif p.name != 'm_z_par':
                non_m_z_par.append(p.get_value())

        return non_log_v_z_par, non_m_z_par


