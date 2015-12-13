import numpy as np
import theano
import theano.tensor as T
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
        f = self.compile()
        estimate, _error = integrate.dblquad(
            lambda z2, z1: f(np.array([[z1, z2]])),
            a, b, lambda z1: a, lambda z1: b)
        return np.log(estimate)

    def compile(self):
        Z = T.dmatrix("Z")  # float64 is required for integration.
        return theano.function([Z], T.exp(-self(Z)))
