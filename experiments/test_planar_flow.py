from matplotlib import pyplot as plt
import numpy as np
import theano
from theano import tensor as T
from densities_2d import planar_flow


def plot_sample(Z, where=plt):
    H, xedges, yedges = np.histogram2d(Z[:, 1], Z[:, 0], bins=100)
    H = np.flipud(np.rot90(H))
    Hmasked = np.ma.masked_where(H == 0, H)
    where.pcolormesh(xedges, yedges, Hmasked)

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

        plot_sample(args[i], _grid[i] if len_args < 5 else _grid[row][col])



def make_flow(k):
    W = T.matrix("W")
    U = T.matrix("U")
    b = T.vector("b")
    Z_0, Z_K, l = planar_flow(W, U, b, k)
    return theano.function([Z_0, W, U, b], Z_K)


if __name__ == '__main__':
    flow_len = 4
    f = make_flow(flow_len)

    Z_0 = np.random.normal(0, 1, [100000, 2])
    W = np.random.normal(0, 1, [flow_len, 2])
    U = np.random.normal(0, 1, [flow_len, 2])
    b = np.random.random(flow_len)

    array = [Z_0]

    for m in range(1, 4):
        array.append(f(Z_0, W*m, U, b))

    for m in range(1, 4):
        array.append(f(Z_0, W, U*m, b))

    for m in range(1, 4):
        array.append(f(Z_0, W, U*2, b*m))

    plot(*array)

    plt.savefig("image.png")
