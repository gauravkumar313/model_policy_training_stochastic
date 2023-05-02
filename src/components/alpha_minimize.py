from datetime import timedelta

import theano


import time

import theano.tensor as T

import models.bb_alpha_z.network as network

from theano.tensor.nnet import LogSumExp


import numpy as np


import copy

class Alpha_Minimize:

    def __init__(self, layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, X_train, y_train, N_train):
        self.layer_sizes = copy.copy(layer_sizes)
        self.layer_sizes[0] = self.layer_sizes[0] + 1
        self.batch_size = batch_size
        self.N_train = N_train
        self.X_train = X_train
        self.y_train = y_train
        self.rate = learning_rate
        self.network = network(self.layer_sizes, n_samples, v_prior, N_train)
        self.index = T.lscalar()
        self.indexes_train = theano.shared(value=np.array(range(0, N_train), dtype=np.int32), borrow=True)
        self.x = T.tensor3('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.lr = T.fscalar()
        self.samp = T.bscalar()

    def fwpass(self):
        output = self.network.output(self.x, False, samples=self.samp, use_indices=False)
        self.fwpass = theano.function(inputs=[self.x, self.samp], outputs=output, allow_input_downcast=True)

    def estimate_marginal_ll(self, alpha, n_samples):
        ll_train = self.network.log_likelihood_values(self.x, self.y, self.indexes, 0.0, 1.0)
        est_marginal_ll = (-1.0 * self.N_train / (self.x.shape[1] * alpha) * \
                           T.sum(LogSumExp(alpha * (T.sum(ll_train,
                                                          2) - self.network.log_f_hat() - self.network.log_f_hat_z()),
                                           0) + \
                                 T.log(1.0 / n_samples)) - self.network.log_normalizer_q() - 1.0 * self.N_train /
                           self.x.shape[1] * self.network.log_normalizer_q_z() + \
                           self.network.log_Z_prior())
        upd = standard_adam(est_marginal_ll, self.network.params,
                   self.indexes_train[self.index * self.batch_size:(self.index + 1) * self.batch_size], self.rate,
                   rescale_local=np.float32(self.N_train / self.batch_size))
        self.estimate_marginal_ll = theano.function(inputs=[self.index], outputs=est_marginal_ll, updates=upd, givens={
            self.x: T.tile(self.X_train[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                           [n_samples, 1, 1]),
            self.y: self.y_train[self.index * self.batch_size: (self.index + 1) * self.batch_size],
            self.indexes: self.indexes_train[self.index * self.batch_size: (self.index + 1) * self.batch_size]})

    def error_minibatch_train(self):
        error = T.sum((T.mean(self.network.output(self.x, self.indexes), 0, keepdims=True)[0, :, :] - self.y) ** 2) / \
                self.layer_sizes[-1]
        self.error_minibatch_train = theano.function(inputs=[self.index], outputs=error, givens={self.x: T.tile(self.X_train[self.index * self.batch_size: (self.index + 1)])})


    def train_model(model, train_data, test_data, batch_size=32, n_epochs=10):
        n_train_batches = len(train_data) // batch_size
        n_test_batches = len(test_data) // batch_size

        for epoch in range(n_epochs):
            start_time = time.monotonic()
            train_energy = []
            permutation = np.random.permutation(len(train_data))

            for batch_idx in range(n_train_batches):
                batch_indices = permutation[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                x_batch = train_data[batch_indices]
                y_batch = model.predict(x_batch)
                train_energy.append(model.train_on_batch(x_batch, y_batch))

            test_energy = []
            for batch_idx in range(n_test_batches):
                x_batch = test_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                y_batch = model.predict(x_batch)
                test_energy.append(model.test_on_batch(x_batch, y_batch))

            train_energy = np.mean(train_energy, axis=0)
            test_energy = np.mean(test_energy, axis=0)

            end_time = time.monotonic()
            elapsed_time = timedelta(seconds=end_time - start_time)

            print(
                f'Epoch {epoch + 1}/{n_epochs}, train_energy: {train_energy:.3f}, test_energy: {test_energy:.3f}, time: {elapsed_time}')



def to_string(x, digits=2):
    return f"{x:.{digits}f}"



def generate_batches(num_samples, batch_size):
    num_batches = (num_samples + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)
        batches.append(slice(start_index, end_index))
    return batches


def logger_add_exponential(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def standard_adam(loss, all_params, index_z, learning_rate=0.001, rescale_local=10):
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    gamma = 1 - 1e-8
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate

    for i, (param, grad) in enumerate(zip(all_params[:-2], all_grads[:-2])):
        # Update for all parameters except means and variances
        m_previous = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        t = theano.shared(np.ones(param.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * grad
        v = b2 * v_previous + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = param - (alpha * m_hat) / (T.sqrt(v_hat) + e)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((param, theta))
        updates.append((t, t + 1.))

    for i, (param, grad) in enumerate(zip(all_params[-2:], all_grads[-2:])):
        m_previous = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        t = theano.shared(np.ones(param.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * grad
        v = b2 * v_previous + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = param - (alpha * rescale_local * m_hat) / (T.sqrt(v_hat) + e)
        theta = T.set_subtensor(theta[index_z], param[index_z])  # keep means and variances at index_z fixed
        updates.append((m_previous, T.set_subtensor(m_previous[index_z], m[index_z])))
        updates.append((v_previous, T.set_subtensor(v_previous[index_z], v[index_z])))
        updates.append((param, theta))
        updates.append((t, T.set_subtensor(t[index_z], t[index_z] + 1.)))

    return updates

