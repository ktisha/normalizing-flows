import os.path
import pickle

import numpy as np
import theano
from lasagne.updates import rmsprop
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt
from theano import tensor as T

from densities_2d import Potential, plot_potential, plot_sample, uniform, mvn_logpdf


def planar_flow(W, U, b, K):
    Z_K = Z_0 = T.matrix("Z_0")
    logdet = 0
    for k in range(K):
        wTu = W[k].dot(U[k])
        m_wTu = -1 + T.log1p(T.exp(wTu))
        U_hat_k = U[k] + (m_wTu - wTu) * W[k] / T.square(W[k].norm(L=2))
        tanh_k = T.tanh(W[k].dot(Z_K.T) + b[k])[:, np.newaxis]

        Z_K = Z_K + tanh_k.dot(U_hat_k[np.newaxis, :])

        # tanh'(z) = 1 - [tanh(z)]^2.
        psi_k = (1 - T.square(tanh_k)) * W[k]
        # we use .5 log(x^2) instead of log|x|.
        logdet = .5 * T.log(T.square(1 + psi_k.dot(U_hat_k))) + logdet

    return Z_0, Z_K, logdet


class PlanarFlow:
    def __init__(self, K, batch_size=2500, n_iter=1000):
        self.D = 2
        self.K = K
        self.batch_size = batch_size
        self.n_iter = n_iter

    def __getstate__(self):
        state = dict(vars(self))
        del state["flow_"]
        return state

    def __setstate__(self, state):
        vars(self).update(state)

        W = theano.shared(self.W_, "W")
        U = theano.shared(self.U_, "U")
        b = theano.shared(self.b_, "b")

        Z_0, Z_K, _logdet = planar_flow(W, U, b, self.K)
        self.flow_ = theano.function([Z_0], Z_K)

    def _assemble(self, potential):
        W = theano.shared(uniform((self.K, self.D)), "W")
        U = theano.shared(np.zeros((self.K, self.D)), "U")
        b = theano.shared(uniform(self.K), "b")

        Z_0, Z_K, logdet = planar_flow(W, U, b, self.K)
        self.flow_ = theano.function([Z_0], Z_K)

        mean = theano.shared(as_floatX(np.zeros(self.D)), "mean")
        covar = theano.shared(as_floatX(np.ones(self.D)), "covar")
        log_q = mvn_logpdf(Z_0, mean, covar) - logdet

        # KL[q_K(z)||exp(-U(z))] ≅ mean(log q_K(z) + U(z)) + const(z)
        # XXX the loss is equal to KL up to an additive constant, thus the
        #     computed value might get negative (while KL cannot).
        kl = (log_q + potential(Z_K)).mean()
        params = [mean, covar, W, U, b]
        updates = rmsprop(kl, params, learning_rate=1e-3)
        return (params, theano.function([Z_0], kl, updates=updates))

    def fit(self, potential):
        (mean, covar, W, U, b), step = self._assemble(potential)
        self.kl_ = np.empty(self.n_iter)
        for i in range(self.n_iter):
            Z_0 = np.random.normal(mean.get_value(),
                                   np.sqrt(covar.get_value()),
                                   size=(self.batch_size, self.D))
            self.kl_[i] = step(as_floatX(Z_0))
            if np.isnan(self.kl_[i]):
                raise ValueError
            elif i % 1000 == 0:
                print("{}/{}: {:8.6f}".format(i + 1, self.n_iter, self.kl_[i]))

        self.mean_ = mean.get_value()
        self.covar_ = covar.get_value()
        self.W_ = W.get_value()
        self.U_ = U.get_value()
        self.b_ = b.get_value()
        return self

    def sample(self, n_samples=1):
        Z_0 = np.random.normal(self.mean_, np.sqrt(self.covar_),
                               size=(n_samples, self.D))
        return self.transform(as_floatX(Z_0))

    def transform(self, Z_0):
        return self.flow_(Z_0)


if __name__ == "__main__":
    Z01 = np.linspace(-4, 4, num=1000)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    potentials = [1, 2, 3, 4]
    ks = [32, 16, 8, 4, 2]

    _fig, grid = plt.subplots(len(potentials), len(ks) + 1,
                              sharex="col", sharey="row")
    for n, row in zip(potentials, grid):
        Z = T.matrix("Z")
        p = theano.function([Z], T.exp(-Potential(n)(Z)))
        plot_potential(Zgrid, p(Zgrid), row[0])

        for i, k in enumerate(ks, 1):
            path = "./potential_{}_k{}.pickle".format(n, k)
            print(path)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    nf = pickle.load(f)
            else:
                nf = PlanarFlow(k, batch_size=1000, n_iter=500000)
                nf.fit(Potential(n))

                log_Z = Potential(n).integrate(-4, 4)
                print("KL {:.2f}".format(nf.kl_[-1] + log_Z))

                with open(path, "wb") as f:
                    pickle.dump(nf, f)

            plot_sample(nf.sample(10000000), k, row[i])

    plt.show()
