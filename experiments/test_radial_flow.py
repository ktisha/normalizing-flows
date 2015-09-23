from matplotlib import pyplot as plt
import numpy as np
import theano
from theano import tensor as T

from densities_2d import plot_sample
from radial_flow import radial_flow

def plot(*args):
    len_args = len(args)

    rows = int(len_args/5)
    if len_args % 5 != 0:
        rows += 1

    _fig, _grid = plt.subplots(rows, 5)

    row = -1
    col = 0
    for i in range(len_args):
        col += 1

        if i % 5 == 0:
            col = 0
            row += 1

        plot_sample(args[i], flow_len, _grid[i] if len_args < 5 else _grid[row][col], False)



def make_flow(k):
    z0 = T.matrix("z0")
    alpha = T.vector("alpha")
    beta = T.vector("beta")
    Z_0, Z_K, l = radial_flow(z0, alpha, beta, k, 2)
    return theano.function([Z_0, z0, alpha, beta], Z_K)


if __name__ == '__main__':
    flow_len = 16
    f = make_flow(flow_len)

    Z_0 = np.random.normal(0, 1, [100000, 2])
    z0 = np.random.normal(2, 10, [flow_len, 2])
    alpha = np.random.random(flow_len)
    beta = np.random.random(flow_len)

    array = [Z_0]

    for m in range(1, 8):
        array.append(f(Z_0, z0, alpha, beta*m))

    for m in range(1, 8):
        array.append(f(Z_0, z0, alpha*m, beta))

    for m in range(1, 8):
        array.append(f(Z_0, z0*m, alpha, beta))

    plot(*array)

    plt.savefig("image.png")
