import os.path
import pickle

import numpy as np
import theano
from lasagne.updates import rmsprop, apply_nesterov_momentum
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt
from theano import tensor as T


def mvn_logpdf(X, mean, covar):
    """Return a theano expression representing the values of the log probability
    density function of the multivariate normal with diagonal covariance.

    >>> X = T.matrix("X")
    >>> mean = T.vector("mean")
    >>> covar = T.vector("covar")
    >>> f = theano.function([X, mean, covar], mvn_logpdf(X, mean, covar))

    >>> from scipy.stats import multivariate_normal
    >>> X = np.array([[-2, 0], [1, -4]])
    >>> mean, covar = np.array([-1, 1]), np.array([.4, .2])
    >>> np.allclose(multivariate_normal.logpdf(X, mean, np.diag(covar)),
    ...             f(X, mean, covar))
    True
    """
    # XXX the cast is necessary because shapes are int64.
    N = X.shape[1].astype(X.dtype)
    return -.5 * (N * T.log(2 * np.pi)
                  + T.log(covar).sum()
                  + (T.square(X - mean) / covar).sum(axis=1))


def potential(Z, n):
    Z1, Z2 = Z[:, 0], Z[:, 1]
    w1 = T.sin(2 * np.pi * Z1 / 4)
    if n == 1:
        return (.5 * T.square((Z.norm(2, axis=1) - 2) / 0.4)
                - T.log(T.exp(-.5 * T.square((Z1 - 2) / 0.6))
                        + T.exp(-.5 * T.square((Z1 + 2) / 0.6))))
    elif n == 2:
        return .5 * T.square(Z2 - w1)
    elif n == 3:
        w2 = 3 * T.exp(-.5 * T.square((Z1 - 1) / 0.6))
        return -T.log(T.exp(-.5 * T.square((Z2 - w1) / 0.35))
                      + T.exp(-.5 * T.square((Z2 - w1 + w2) / 0.35)))
    elif n == 4:
        w3 = 3 * T.nnet.sigmoid((Z1 - 1) / 0.3)
        return -T.log(T.exp(-.5 * T.square((Z2 - w1) / 0.4))
                      + T.exp(-.5 * T.square((Z2 - w1 + w3) / 0.35)))


def planar_flow(W, U, b, K):
    Z_K = Z_0 = T.matrix("Z_0")
    logdet = []
    for k in range(K):
        wTu = W[k].dot(U[k])
        m_wTu = -1 + T.log(1 + T.exp(wTu))
        U_hat_k = U[k] + (m_wTu - wTu) * W[k] / W[k].norm(L=2)
        tanh_k = T.tanh(W[k].dot(Z_K.T) + b[k])[:, np.newaxis]

        Z_K = Z_K + U_hat_k * tanh_k

        # tanh'(z) = 1 - [tanh(z)]^2.
        psi_k = (1 - T.square(tanh_k)) * W[k]
        # we use .5 log(x^2) instead of log|x|.
        logdet.append(.5 * T.log(T.square(1 + psi_k.dot(U_hat_k))))

    return Z_0, Z_K, logdet


class NormalizingFlow:
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
        mean = theano.shared(as_floatX(np.zeros(self.D)), "mean")
        covar = theano.shared(as_floatX(np.ones(self.D)), "covar")

        W = theano.shared(as_floatX(np.random.random((self.K, self.D))), "W")
        U = theano.shared(as_floatX(np.random.random((self.K, self.D))), "U")
        b = theano.shared(as_floatX(np.random.random(self.K)), "b")

        Z_0, Z_K, logdet = planar_flow(W, U, b, self.K)
        self.flow_ = theano.function([Z_0], Z_K)

        log_q = mvn_logpdf(Z_0, mean, covar)
        for k in range(self.K):
            log_q -= logdet[k]

        # KL[q_K(z)||exp(-U(z))] ≅ mean(log q_K(z) + U(z)) + const(z)
        # XXX the loss is equal to KL up to an additive constant, thus the
        #     computed value might get negative (while KL cannot).
        kl = (log_q + potential(Z_K)).mean()
        params = [mean, covar, W, U, b]
        updates = apply_nesterov_momentum(
            rmsprop(kl, params, learning_rate=1e-4), params)
        return params, theano.function([Z_0], kl, updates=updates)

    def fit(self, potential):
        (mean, covar, W, U, b), step = self._assemble(potential)
        self.kl_ = np.empty(self.n_iter)
        for i in range(self.n_iter):
            Z_0 = np.random.normal(mean.get_value(), covar.get_value(),
                                   size=(self.batch_size, self.D))
            self.kl_[i] = step(as_floatX(Z_0))
            print("{}/{}: {:8.6f}".format(i + 1, self.n_iter, self.kl_[i]))

        self.mean_ = mean.get_value()
        self.covar_ = covar.get_value()
        self.W_ = W.get_value()
        self.U_ = U.get_value()
        self.b_ = b.get_value()
        return self

    def sample(self, n_samples=1):
        Z_0 = np.random.normal(self.mean_, self.covar_,
                               size=(n_samples, self.D))
        return self.flow_(as_floatX(Z_0))


def plot_potential(Z, p, where=plt):
    # XXX the pictures in the paper seem to have the y-axis flipped.
    where.scatter(Z[:, 0], -Z[:, 1], c=np.exp(-p(Z)), s=5, edgecolor="")


def plot_potential_sample(Z, k, where=plt):
    H, xedges, yedges = np.histogram2d(Z[:, 0], Z[:, 1], bins=100)
    H = np.flipud(np.rot90(H))
    Hmasked = np.ma.masked_where(H == 0, H)
    where.pcolormesh(xedges, yedges, Hmasked)
    where.xlim([-4, 4])
    where.ylim([-4, 4])
    where.set_title("K = {}".format(k))


if __name__ == "__main__":
    n_samples = 100000
    _fig, grid = plt.subplots(4, 4, sharex="col", sharey="row")
    for n, row in enumerate(grid, 1):
        Z = T.matrix("Z")
        p = theano.function([Z], potential(Z, n))
        plot_potential(as_floatX(np.random.uniform(-4, 4, size=(n_samples, 2))),
                       p, row[0])

        for i, k in enumerate([2, 8, 16], 1):
            path = "./potential_{}_k{}.pickle".format(n, k)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    nf = pickle.load(f)
            else:
                nf = NormalizingFlow(k, batch_size=10000, n_iter=10000)
                nf.fit(lambda Z: potential(Z, n))
                with open(path, "wb") as f:
                    pickle.dump(nf, f)

            plot_potential_sample(nf.sample(n_samples), k, row[i])

    plt.show()
