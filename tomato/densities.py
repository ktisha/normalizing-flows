import numpy as np
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
from scipy import integrate

from .utils import logaddexp


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
            lambda z2, z1: f(np.array([[z1, z2]])),
            a, b, lambda z1: a, lambda z1: b)
        return np.log(estimate)


def plot_sample(Z, k, where=plt.axes()):
    #                                          vvv
    # the pictures in the paper seem to have the y-axis flipped.
    H, xedges, yedges = np.histogram2d(Z[:, 0], -Z[:, 1], bins=100)
    H = np.flipud(np.rot90(H))
    Hmasked = np.ma.masked_where(H == 0, H)
    where.pcolormesh(xedges, yedges, Hmasked)
    where.set_xlim((-4, 4))
    where.set_ylim((-4, 4))
    where.set_title("K = {}".format(k))