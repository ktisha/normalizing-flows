import numpy as np
import theano
from matplotlib import pyplot as plt
from theano import tensor as T


def mvn_logpdf(X, mean, covar):
    """Return a theano expression representing the values of the log probability
    density function of the multivariate normal with diagonal covariance.
    """
    return -.5 * (X.shape[1] * T.log(2 * np.pi) + T.log(covar).sum()
                  + ((mean ** 2) / covar).sum()
                  - 2 * X.dot((mean / covar).T)
                  + T.dot(X ** 2, (1 / covar).T))


def potential(Z, n):
    Z1 = Z[:, 0]
    Z2 = Z[:, 1]

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


class Potential:
    def __init__(self, n):
        self.n = n

    def __call__(self, Z):
        Z = T.dtensor3("Z")
        return theano.function([Z], potential(Z, self.n))

    def plot(self, Z, where=plt):
        # XXX the pictures in the paper seem to have the y-axis flipped.
        where.scatter(Z[:, 0], -Z[:, 1], c=np.exp(-self(Z)), s=5, edgecolor="")
        where.set_title(self.n)


def planar_flow(Z_0, W, U, b, K):
    Z_K = Z_0
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

    return Z_K, logdet


class NormalizingFlow:
    def __init__(self, K, n_iter=1000, batch_size=2500, alpha=0.01):
        self.D = 2
        self.K = K
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.alpha = alpha

    def _assemble(self, potential):
        mean = theano.shared(np.zeros(self.D), "mean")
        covar = theano.shared(np.ones(self.D), "covar")

        Z_0 = T.dmatrix("Z_0")
        W = theano.shared(np.random.random((self.K, self.D)), "W")
        U = theano.shared(np.random.random((self.K, self.D)), "U")
        b = theano.shared(np.random.random(self.K), "b")

        Z_K, logdet = planar_flow(Z_0, W, U, b, self.K)
        self.flow_ = theano.function([Z_0], Z_K)

        log_q = mvn_logpdf(Z_0, mean, covar)
        for k in range(self.K):
            log_q -= logdet[k]

        # KL[q_K(z)||exp(-U(z))] ≅ mean(log q_K(z) + U(z)) + const(z)
        # XXX the loss is equal to KL up to an additive constant, thus the
        #     computed value might get negative (while KL cannot).
        kl = (log_q + potential(Z_K)).mean()
        dmean, dcovar, dW, dU, db = T.grad(kl, [mean, covar, W, U, b])
        return mean, covar, theano.function([Z_0], kl, updates=[
            (mean, mean - self.alpha * dmean),
            (covar, covar - self.alpha * dcovar),
            (W, W - self.alpha * dW),
            (U, U - self.alpha * dU),
            (b, b - self.alpha * db)])

    def fit(self, potential):
        mean, covar, step = self._assemble(potential)
        self.kl_ = []
        for i in range(self.n_iter):
            Z_0 = np.random.multivariate_normal(
                mean.get_value(), np.diag(covar.get_value()),
                size=self.batch_size)
            self.kl_.append(step(Z_0))
            print(self.kl_[-1])

        self.mean_ = mean.get_value()
        self.covar_ = covar.get_value()
        return self

    def sample(self, n_samples=1):
        Z_0 = np.random.multivariate_normal(
            self.mean_, np.diag(self.covar_), size=n_samples)
        return self.flow_(Z_0)


def plot_potentials(Z):
    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex="col", sharey="row")
    Potential(1).plot(Z, ax1)
    Potential(2).plot(Z, ax2)
    Potential(3).plot(Z, ax3)
    Potential(4).plot(Z, ax4)


def plot_potential_sample(Z):
    H, xedges, yedges = np.histogram2d(Z[:, 0], Z[:, 1], bins=100)
    H = np.flipud(np.rot90(H))
    Hmasked = np.ma.masked_where(H == 0, H)
    plt.pcolormesh(xedges, yedges, Hmasked)


if __name__ == "__main__":
    n_samples = 1000000  # <-- the more the better.
    # Z = np.random.uniform(-4, 4, size=(n_samples, 2))
    # plot_potentials()

    # Z = Potential(1).sample(n_samples)
    # plot_potential_sample(Z)

    nf = NormalizingFlow(16, batch_size=5000, n_iter=10000)
    nf.fit(lambda Z: potential(Z, 4))
    plt.plot(range(len(nf.kl_)), nf.kl_)
    plt.grid(True)
    plt.show()

    plot_potential_sample(nf.sample(n_samples))
    plt.show()
