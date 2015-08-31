import numpy as np
import theano
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy.special import expit
from theano import tensor as T


def mvn_logpdf(X, mean, covar):
    """Return a theano expression representing the values of the log probability
    density function of the multivariate normal with diagonal covariance.
    """
    return -.5 * (X.shape[1] * T.log(2 * np.pi) + T.log(covar).sum()
                  + ((mean ** 2) / covar).sum()
                  - 2 * X.dot((mean / covar).T)
                  + T.dot(X ** 2, (1 / covar).T))


class Potential:
    def __init__(self, n):
        self.n = n

    def __call__(self, Z):
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]

        w1 = np.sin(2 * np.pi * Z1 / 4)
        if self.n == 1:
            return (.5 * np.square((LA.norm(Z, 2, axis=1) - 2) / 0.4)
                    - np.logaddexp(-.5 * np.square((Z1 - 2) / 0.6),
                                   -.5 * np.square((Z1 + 2) / 0.6)))
        elif self.n == 2:
            return .5 * np.square(Z2 - w1)
        elif self.n == 3:
            w2 = 3 * np.exp(-.5 * np.square((Z1 - 1) / 0.6))
            return -np.logaddexp(-.5 * np.square((Z2 - w1) / 0.35),
                                 -.5 * np.square((Z2 - w1 + w2) / 0.35))
        elif self.n == 4:
            w3 = 3 * expit((Z1 - 1) / 0.3)
            return -np.logaddexp(-.5 * np.square((Z2 - w1) / 0.4),
                                 -.5 * np.square((Z2 - w1 + w3) / 0.35))

    def sample(self, n_samples=1):
        # Dead simple rejection sampling with Uniform([-4, 4]^2) proposal
        # distribution. The while loop is required, because acceptance
        # rate is unlikely to be >10%.
        acc = []
        left = n_samples
        while left:
            X = np.random.uniform(-4, 4, size=(left, 2))
            p = np.exp(-self(X))
            mask = np.random.random() < p
            print("accepted {:.4f}% ({} remaining)"
                  .format(100 * mask.mean(), (~mask).sum()))
            acc.append(X[mask])
            left -= mask.sum()
        return np.concatenate(acc)

    def plot(self, Z, where=plt):
        # XXX the pictures in the paper seem to have the y-axis flipped.
        where.scatter(Z[:, 0], -Z[:, 1], c=np.exp(-self(Z)), s=5, edgecolor="")
        where.set_title(self.n)


def planar_flow(D, K):
    mean = T.zeros(D)
    covar = T.ones(D)

    z = T.dmatrix("z")
    W = T.dmatrix("W")
    U = T.dmatrix("U")
    b = T.dvector("b")

    logdet = theano.shared(0, "logdet")
    f_k = z
    for k in range(K):
        wTu = W[k].dot(U[k])
        m_wTu = -1 + T.log1p(T.exp(wTu))
        u_hat = U[k] + (m_wTu - wTu) * W[k] / T.square(W[k].norm(L=2))
        tanh_k = T.tanh(W[k].dot(f_k.T) + b[k])[:, np.newaxis]

        # tanh'(z) = 1 - [tanh(z)]^2.
        psi_k = (1 - T.square(tanh_k)) * W[k]
        logdet += .5 * T.log(T.square(1 + psi_k.dot(U[k]))).sum()

        f_k = f_k + u_hat * tanh_k

    # Here we assume the flow goes backwards from the observed x_i to
    # z_0 which has a known MVN distribution.
    nll = (-mvn_logpdf(f_k, mean, covar) + logdet).sum()
    nll_grad = T.grad(nll, [mean, covar, W, U, b])
    return (theano.function([z, mean, covar, W, U, b], nll),
            theano.function([z, mean, covar, W, U, b], nll_grad))


class NormalizingFlow:
    def __init__(self, K, n_iter=1000, batch_size=2500, alpha=0.0001):
        self.K = K
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.alpha = alpha

    def fit(self, X):
        n_samples, n_features = X.shape
        self.nll_ = []
        self.mean_ = np.zeros(n_features)
        self.covar_ = np.ones(n_features)
        self.W = np.random.random(size=(self.K, n_features))
        self.U = np.random.random(size=(self.K, n_features))
        self.b = np.random.random(size=self.K)
        nll, nll_grad = planar_flow(n_features, self.K)
        for i in range(self.n_iter):
            indices = np.random.choice(n_samples, size=self.batch_size)
            Xb = X[indices]

            self.nll_.append(
                nll(Xb, self.mean_, self.covar_, self.W, self.U, self.b))
            print(self.nll_[-1])
            dmean, dcovar, dW, dU, db = nll_grad(
                Xb, self.mean_, self.covar_, self.W, self.U, self.b)

            self.mean_ -= self.alpha * dmean
            self.covar_ -= self.alpha * dcovar
            self.W -= self.alpha * dW
            self.U -= self.alpha * dU
            self.b -= self.alpha * db
        return self

    def sample(self, n_samples=1):
        pass


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

    Z = Potential(1).sample(n_samples)
    nf = NormalizingFlow(4, n_iter=5000).fit(Z)
    plt.semilogy(range(len(nf.nll_)), nf.nll_)
    plt.grid(True)
    plt.show()
