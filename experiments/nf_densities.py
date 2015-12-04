import argparse
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt
from scipy import stats

from tomato.densities import Potential
from tomato.layers import planar_flow
from tomato.utils import mvn_std_logpdf


def main(num_flows, num_iter, batch_size, potential):
    p = Potential(potential)
    num_features = 2

    print("Building model and compiling functions...")
    net = OrderedDict()
    net["z_0"] = InputLayer((batch_size, num_features))
    net["z_k"], logdet = planar_flow(net["z_0"], num_flows)

    z_0_var, z_k_var, *logdet_vars = get_output(list(net.values()) + logdet,
                                                deterministic=False)

    # KL[q_K(z)||exp(-U(z))] ≅ mean(log q_K(z) + U(z)) + const(z)
    # XXX the loss is equal to KL up to an additive constant, thus the
    #     computed value might get negative (while KL cannot).
    log_q = mvn_std_logpdf(z_0_var) - sum(logdet_vars)
    kl = (log_q + p(z_k_var)).mean()

    params = get_all_params(net["z_k"], trainable=True)
    updates = adam(kl, params, learning_rate=1e-4)
    train_step = theano.function([z_0_var], kl, updates=updates)
    flow = theano.function([z_0_var], T.exp(log_q))

    print("Starting training...")
    train_loss = np.empty(num_iter)
    for i in range(num_iter):
        z_0 = as_floatX(np.random.normal(size=(batch_size, num_features)))
        train_loss[i] = train_step(z_0)
        if np.isnan(train_loss[i]):
            raise ValueError
        elif i and i % 1000 == 0:
            print("{:4d}/{}: {:8.6f}".format(i, num_iter, train_loss[i]))

    log_partition = p.integrate(-4, 4)
    print("Done. Expected KL {:.2f}".format(train_loss[-1] + log_partition))

    # XXX either this is a bad way to viz. NF or optimization
    #     doesn't work.
    print("Sampling...")
    num_steps = 500
    grid = []
    for i, j in np.ndindex(num_steps - 1, num_steps - 1):
        grid.append([(i + 1) / float(num_steps),
                     (j + 1) / float(num_steps)])

    z_0 = as_floatX(stats.norm.ppf(grid))

    _fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    plot_sample(z_0, p.compile(), where=ax1)
    plot_sample(z_0, flow, where=ax2)
    plt.show()


def plot_sample(Z, f, where):
    where.scatter(Z[:, 0], -Z[:, 1], c=f(Z), s=5, edgecolor="")
    where.set_xlim((-4, 4))
    where.set_ylim((-4, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approximate 2D density with planar NF")
    parser.add_argument("-F", dest="num_flows", type=int, default=2)
    parser.add_argument("-I", dest="num_iter", type=int, default=10000)
    parser.add_argument("-B", dest="batch_size", type=int, default=1000)
    parser.add_argument(dest="potential", type=int)

    args = parser.parse_args()
    main(**vars(args))
