import numpy as np
import theano
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt
from scipy import integrate
from theano import tensor as T

from tomato.utils import mvn_logpdf, logaddexp


def uniform(size):
    return as_floatX(np.random.uniform(-4, 4, size=size))


class Potential:
    def __init__(self, n):
        self.n = n

    def __call__(self, Z):
        Z1, Z2 = Z[:, 0], Z[:, 1]
        w1 = T.sin(2 * np.pi * Z1 / 4)
        if self.n == 1:
            return (.5 * T.square((Z.norm(2, axis=1) - 2) / 0.4)
                    - logaddexp(-.5 * T.square((Z1 - 2) / 0.6),
                                -.5 * T.square((Z1 + 2) / 0.6)))
        elif self.n == 2:
            return .5 * T.square((Z2 - w1) / 0.4)
        elif self.n == 3:
            w2 = 3 * T.exp(-.5 * T.square((Z1 - 1) / 0.6))
            return -logaddexp(-.5 * T.square((Z2 - w1) / 0.35),
                              -.5 * T.square((Z2 - w1 + w2) / 0.35))
        elif self.n == 4:
            w3 = 3 / (1 + T.exp(-(Z1 - 1) / 0.3))
            return -logaddexp(-.5 * T.square((Z2 - w1) / 0.4),
                              -.5 * T.square((Z2 - w1 + w3) / 0.35))

    def integrate(self, a, b):
        Z = T.dmatrix("Z")
        f = theano.function([Z], T.exp(-self(Z)))
        estimate, _error = integrate.dblquad(
            lambda z2, z1: f(as_floatX(np.array([[z1, z2]]))),
            a, b, lambda z1: a, lambda z1: b)
        return np.log(estimate)


def plot_potential(Z, p, where=plt):
    """
    The pictures in the paper seem to have the y-axis flipped.
    :type where: matplotlib.axes.Axes
    """
    where.scatter(Z[:, 0], -Z[:, 1], c=p, s=5, edgecolor="")
    where.set_xlim((-4, 4))
    where.set_ylim((-4, 4))


def plot_sample(Z, k, where=plt, set_limits=True):
    """
    The pictures in the paper seem to have the y-axis flipped.
    :type where: matplotlib.axes.Axes
    """
    H, xedges, yedges = np.histogram2d(Z[:, 0], -Z[:, 1], bins=100)
    H = np.flipud(np.rot90(H))
    Hmasked = np.ma.masked_where(H == 0, H)
    where.pcolormesh(xedges, yedges, Hmasked)
    if set_limits:
        where.set_xlim((-4, 4))
        where.set_ylim((-4, 4))
    where.set_title("K = {}".format(k))

class Flow:
    def __init__(self, K, batch_size=2500, n_iter=1000):
        self.D = 2
        self.K = K
        self.batch_size = batch_size
        self.n_iter = n_iter

    def _assemble(self, potential):
        raise NotImplementedError()

    def fit(self, potential):
        raise NotImplementedError()

    def sample(self, n_samples=1):
        Z_0 = np.random.normal(self.mean_, np.sqrt(self.covar_),
                               size=(n_samples, self.D))
        return self.transform(as_floatX(Z_0))

    def transform(self, Z_0):
        return self.flow_(Z_0)
