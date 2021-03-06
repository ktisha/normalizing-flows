import os.path
import pickle

import numpy as np
import theano
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt
from theano import tensor as T

from .densities_2d import mvn_logpdf, Potential, plot_potential, plot_sample, \
     Flow


def radial_flow(z0, alpha, beta, K, D):
    Z_K = Z_0 = T.matrix("Z_0")
    logdet = 0

    m_beta = -1 + T.log1p(T.exp(beta))
    beta_hat = -alpha + m_beta

    for k in range(K):
        z_difference = Z_K - z0[k]
        r = T.sqrt(T.square(z_difference).sum(axis=1))

        h_beta = beta_hat[k] / (alpha[k] + r)
        hh = - z_difference / (r * T.square(alpha[k] + r))[:, None]

        # here we assumed that the last term is \beta * h' * (z-z_0) instead of h' * r
        df = (T.power((1 + h_beta), D - 1) *
              (1 + h_beta + beta_hat[k] * (hh * z_difference).sum(axis=1)))

        logdet = .5 * T.log(T.square(df)) + logdet
        Z_K = Z_K + h_beta[:, None] * z_difference

    return Z_0, Z_K, logdet


class RadialFlow(Flow):
    def __init__(self, K, batch_size=2500, n_iter=1000):
        super().__init__(K, batch_size, n_iter)

    def __getstate__(self):
        state = dict(vars(self))
        del state["flow_"]
        return state

    def __setstate__(self, state):
        vars(self).update(state)

        z0 = theano.shared(self.z0_, "z0")
        alpha = theano.shared(self.alpha_, "alpha")
        beta = theano.shared(self.beta_, "beta")

        Z_0, Z_K, _logdet = radial_flow(z0, alpha, beta, self.K, self.D)
        self.flow_ = theano.function([Z_0], Z_K)

    def _assemble(self, potential):
        z0 = theano.shared(np.random.random((self.K, self.D)), "z0")
        alpha = theano.shared(np.zeros(self.K), "alpha")
        beta = theano.shared(np.zeros(self.K), "beta")

        Z_0, Z_K, logdet = radial_flow(z0, alpha, beta, self.K, self.D)
        self.flow_ = theano.function([Z_0], Z_K)

        mean = theano.shared(as_floatX(np.zeros(self.D)), "mean")
        covar = theano.shared(as_floatX(np.ones(self.D)), "covar")
        log_q = mvn_logpdf(Z_0, mean, covar) - logdet

        kl = (log_q + potential(Z_K)).mean()
        params = [mean, covar, z0, alpha, beta]

        updates = adam(kl, params)
        return (params, theano.function([Z_0], kl, updates=updates))

    def _assemble_jacobian(self):
        z0 = T.matrix("z0")
        alpha = T.vector("alpha")
        beta = T.vector("beta")

        Z_0, Z_K, logdet = radial_flow(z0, alpha, beta, self.K, self.D)

        self.logdet_ = theano.function([Z_0, z0, alpha, beta], logdet)
        self.jacobian_ = theano.function(
            [Z_0, z0, alpha, beta],
            T.jacobian(Z_K.sum(axis=0), Z_0))

    def fit(self, potential):
        (mean, covar, z0, alpha, beta), step = self._assemble(potential)
        self._assemble_jacobian()
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

                z0_value = z0.get_value()
                alpha_value = alpha.get_value()
                beta_value = beta.get_value()

                print(self.logdet_(Z_0, z0_value, alpha_value, beta_value))

                # XXX passing Z_0 to jacobian_ directly yields incorrect
                # results. I wonder why ...
                J = np.empty(self.batch_size)
                for k, Z_0k in enumerate(Z_0):
                    Jk = self.jacobian_([Z_0k], z0_value,
                                        alpha_value, beta_value)
                    _sign, J[k] = np.linalg.slogdet(*np.rollaxis(Jk, axis=1))
                print(J)

        self.mean_ = mean.get_value()
        self.covar_ = covar.get_value()
        self.z0_ = z0.get_value()
        self.alpha_ = alpha.get_value()
        self.beta_ = beta.get_value()
        return self


if __name__ == "__main__":
    Z01 = np.linspace(-4, 4, num=1000)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    potentials = [1, 2]
    ks = [2]

    _fig, grid = plt.subplots(len(potentials), len(ks) + 1,
                              sharex="col", sharey="row")
    for n, row in zip(potentials, grid):
        Z = T.matrix("Z")
        p = theano.function([Z], T.exp(-Potential(n)(Z)))
        plot_potential(Zgrid, p(Zgrid), row[0])

        for i, k in enumerate(ks, 1):
            path = "./radial_{}_k{}.pickle".format(n, k)
            print(path)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    nf = pickle.load(f)
            else:
                nf = RadialFlow(k, batch_size=3, n_iter=50000)
                nf.fit(Potential(n))

                log_Z = Potential(n).integrate(-4, 4)
                print("KL {:.2f}".format(nf.kl_[-1] + log_Z))

                # with open(path, "wb") as f:
                #     pickle.dump(nf, f)

            plot_sample(nf.sample(1000000), k, row[i])

    plt.show()
