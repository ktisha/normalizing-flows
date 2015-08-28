import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy.special import expit


class potential:
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


if __name__ == "__main__":
    Z = np.random.uniform(-4, 4, size=(100000, 2))

    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex="col", sharey="row")
    potential(1).plot(Z, ax1)
    potential(2).plot(Z, ax2)
    potential(3).plot(Z, ax3)
    potential(4).plot(Z, ax4)
    plt.show()
