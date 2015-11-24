import argparse
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX
from matplotlib import pyplot as plt

from tomato.densities import Potential, plot_sample
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
    updates = adam(kl, params)

    train_step = theano.function([z_0_var], kl, updates=updates)
    flow = theano.function([z_0_var], z_k_var)

    print("Starting training...")
    train_loss = np.empty(num_iter)
    for i in range(num_iter):
        z_0 = np.random.normal(size=(batch_size, num_features))
        train_loss[i] = train_step(as_floatX(z_0))
        if np.isnan(train_loss[i]):
            raise ValueError
        elif i and i % 1000 == 0:
            print("{:4d}/{}: {:8.6f}".format(i, num_iter, train_loss[i]))

    log_partition = p.integrate(-4, 4)
    print("Done. Expected KL {:.2f}".format(train_loss[-1] + log_partition))

    print("Sampling...")
    num_samples = 100000
    z_0 = np.random.normal(size=(num_samples, num_features))
    z_k = flow(z_0)
    plot_sample(z_k, num_flows)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approximate 2D density with planar NF")
    parser.add_argument("-F", dest="num_flows", type=int, default=2)
    parser.add_argument("-I", dest="num_iter", type=int, default=10000)
    parser.add_argument("-B", dest="batch_size", type=int, default=1000)
    parser.add_argument(dest="potential", type=int)

    args = parser.parse_args()
    main(**vars(args))
