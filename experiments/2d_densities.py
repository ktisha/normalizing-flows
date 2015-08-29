import theano
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy.special import expit
from theano import tensor as T


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

    def plot(self, Z, where=plt):
        # XXX the pictures in the paper seem to have the y-axis flipped.
        where.scatter(Z[:, 0], -Z[:, 1], c=np.exp(-self(Z)), s=5, edgecolor="")
        where.set_title(self.n)


def planar_flow(K):
    z = T.dvector("z")
    W = T.dmatrix("W")
    U = T.dmatrix("U")
    b = T.dvector("b")

    f_k = z
    for k in range(K):
        wTu = W[k].dot(U[k])
        m_wTu = -1 + T.log1p(T.exp(wTu))
        u_hat = U[k] + (m_wTu - wTu) * W[k] / T.square(W[k].norm(L=2))
        f_k = f_k + u_hat * T.tanh(W[k].dot(f_k) + b[k])

    return theano.function([z, W, U, b], f_k)


class NormalizingFlow:
    def __init__(self, K, n_iter=1000, batch_size=100):
        self.K = K
        self.n_iter = n_iter
        self.batch_size = batch_size

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_ = np.zeros(n_features)
        self.covar_ = np.diag(np.ones(n_features))
        self.W = np.random.random(size=(self.K, n_features))
        self.U = np.random.random(size=(self.K, n_features))
        self.b = np.random.random(size=self.K)
        self.flow = planar_flow(self.K)
        for iter in range(self.n_iter):
            indices = np.random.choice(len(X), size=self.batch_size)
            Xb = X[indices]
            zb0 = np.random.multivariate_normal(self.mean_, self.covar_,
                                                size=self.batch_size)
            zbk = self.flow(zb0, self.W, self.U, self.b)

    def sample(self, n_samples=1):
        pass


if __name__ == "__main__":
    Z = np.random.uniform(-4, 4, size=(100000, 2))

    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex="col", sharey="row")
    Potential(1).plot(Z, ax1)
    Potential(2).plot(Z, ax2)
    Potential(3).plot(Z, ax3)
    Potential(4).plot(Z, ax4)
    plt.show()
