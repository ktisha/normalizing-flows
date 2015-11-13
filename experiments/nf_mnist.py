import argparse
import pickle
import re
import time
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, FeaturePoolLayer, \
    get_output, get_all_params, get_all_param_values, \
    set_all_param_values, concat
from lasagne.nonlinearities import rectify, identity
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX

from .datasets import load_mnist_dataset
from .layers import GaussianNoiseLayer
from .layers import PlanarFlowLayer, IndexLayer
from .utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches


maxout = FeaturePoolLayer


def build_model(batch_size, num_features, num_latent, num_hidden, num_flows):
    net = {}

    pool_size = num_hidden // 100

    # q(z|x)
    net["enc_input"] = InputLayer((batch_size, num_features))
    net["enc_hidden1"] = DenseLayer(net["enc_input"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["enc_hidden2"] = maxout(net["enc_hidden1"], pool_size)
    net["z_mu"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                             nonlinearity=identity)
    net["z_log_covar"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                                    nonlinearity=identity)

    net["z"] = GaussianNoiseLayer(net["z_mu"], net["z_log_covar"])

    z = net["z"]
    logdet = []
    for _ in range(num_flows):
        flow_layer = PlanarFlowLayer(z)
        z = IndexLayer(flow_layer, 0)
        logdet.append(IndexLayer(flow_layer, 1))

    net["z_k"] = z
    net["logdet"] = logdet

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z_k"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = maxout(net["dec_hidden1"], pool_size)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                             nonlinearity=identity)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def elbo_nf(X_var, x_mu_var, x_log_covar_var, z_0_var, z_k_var, logdet_var,
            beta_t):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return (
        beta_t * (mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
                  + mvn_std_logpdf(z_k_var))
        - mvn_std_logpdf(z_0_var) + logdet_var
    ).mean()


def main(num_latent, num_hidden, num_flows, batch_size, num_epochs):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    beta_var = T.scalar("beta_t")  # Inverse temperature.
    net = build_model(batch_size, num_features, num_latent, num_hidden,
                      num_flows)

    vars = ["x_mu", "x_log_covar", "z", "z_k"]
    x_mu_var, x_log_covar_var, z_0_var, z_k_var, *logdet_var = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=False
    )
    elbo_train = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                         z_0_var, z_k_var, sum(logdet_var), beta_var)

    x_mu_var, x_log_covar_var, z_0_var, z_k_var, *logdet_var = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=True
    )
    elbo_val = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                       z_0_var, z_k_var, sum(logdet_var), beta_var)

    params = get_all_params(concat([net["x_mu"], net["x_log_covar"]]),
                            trainable=True)
    updates = adam(-elbo_train, params)
    train_nelbo = theano.function([X_var, beta_var], -elbo_train,
                                  updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val, givens={beta_var: 1.0})

    print("Starting training...")
    train_errs = []
    val_errs = []
    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        train_err, train_batches = 0, 0
        for Xb, yb in iter_minibatches(X_train, y_train,
                                       batch_size=batch_size):
            beta = min(1, 0.01 + float(epoch) / num_epochs)
            train_err += train_nelbo(Xb, beta)
            train_batches += 1

        val_err, val_batches = 0, 0
        for Xb, yb in iter_minibatches(X_val, y_val, batch_size=batch_size):
            val_err += val_nelbo(Xb)
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.perf_counter() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        train_errs.append(train_err)
        val_errs.append(val_err)

        threshold = np.mean(val_errs[:100])
        if len(val_errs) >= 250 and val_err > threshold:
            print("Stopped early {} > {}!".format(val_err, threshold))
            break

    prefix = "nf_mnist_L{}_H{}_F{}".format(num_latent, num_hidden, num_flows)
    pd.DataFrame.from_dict({
        "train_err": train_errs,
        "val_err": val_errs
    }).to_csv(prefix + ".csv", index=False)

    all_param_values = get_all_param_values(
        concat([net["x_mu"], net["x_log_covar"]]))
    with open(prefix + ".pickle", "wb") as handle:
        pickle.dump(all_param_values, handle)


def load_model(path):
    print("Loading data...")
    X_train, *_rest = load_mnist_dataset()
    num_features = X_train.shape[1]

    [chunk] = re.findall(r"nf_mnist_L(\d+)_H(\d+)_F(\d+)", str(path))
    num_latent, num_hidden, num_flows = map(int, chunk)

    print("Building model and compiling functions...")
    net = build_model(1, num_features, num_latent, num_hidden, num_flows)
    with path.open("rb") as handle:
        set_all_param_values(concat([net["x_mu"], net["x_log_covar"]]),
                             pickle.load(handle))
    return net


def plot_manifold(path):
    net = load_model(path)
    z_var = T.matrix()
    decoder = theano.function([z_var], get_output(net["x_mu"], {net["z"]: z_var}))

    figure = plt.figure()

    Z01 = np.linspace(-1, 1, num=16)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    for (i, z_i) in enumerate(Zgrid, 1):
        figure.add_subplot(16, 16, i)
        image = decoder(np.array([z_i])).reshape((28, 28))
        plt.axis('off')
        plt.imshow(image, cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(path.stem + "_manifold.png")))


def plot_sample(path):
    net = load_model(path)
    z_var = T.matrix()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}))
    z_covar = theano.function(
        [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var})))

    figure = plt.figure()

    n_samples = 256
    z = np.random.normal(size=(n_samples, 2)).astype(theano.config.floatX)
    for i, z_i in enumerate(z, 1):
        mu, covar = z_mu(np.array([z_i])), z_covar(np.array([z_i]))
        x = np.random.normal(mu, covar)
        figure.add_subplot(16, 16, i)
        plt.axis("off")
        plt.imshow(x.reshape((28, 28)), cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(path.stem + "_sample.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn NF-VAE from MNIST data")
    parser.add_argument("-L", dest="num_latent", type=int, default=100)
    parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    parser.add_argument("-F", dest="num_flows", type=int, default=2)
    parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    parser.add_argument("-B", dest="batch_size", type=int, default=500)

    # path = Path("nf_mnist_L2_H500_F8.pickle")
    # plot_manifold(path)
    args = parser.parse_args()
    main(**vars(args))
