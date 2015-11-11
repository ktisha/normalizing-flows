import argparse
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values, \
    concat
from lasagne.nonlinearities import rectify, identity
from lasagne.updates import rmsprop
from lasagne.utils import floatX as as_floatX

from .datasets import load_mnist_dataset
from .layers import GaussianNoiseLayer
from .utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches


def build_model(batch_size, num_features, num_latent, num_hidden):
    net = {}
    # q(z|x)
    net["enc_input"] = InputLayer((batch_size, num_features))
    net["enc_hidden1"] = DenseLayer(net["enc_input"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["enc_hidden2"] = DenseLayer(net["enc_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["z_mu"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                             nonlinearity=identity)
    net["z_log_covar"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                                    nonlinearity=identity)

    net["z"] = GaussianNoiseLayer(net["z_mu"], net["z_log_covar"])

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                             nonlinearity=identity)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def elbo(X_var, x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return (
        mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
        + mvn_std_logpdf(z_var)
        - mvn_log_logpdf(z_var, z_mu_var, z_log_covar_var)
    ).mean()


def load_model(path):
    print("Loading data...")
    X_train, *_rest = load_mnist_dataset()
    num_features = X_train.shape[1]

    [chunk] = re.findall(r"vae_mnist_L(\d+)_H(\d+)", str(path))
    num_latent, num_hidden = map(int, chunk)

    print("Building model and compiling functions...")
    net = build_model(1, num_features, num_latent, num_hidden)
    with path.open("rb") as handle:
        set_all_param_values(concat([net["x_mu"], net["x_log_covar"]]),
                             pickle.load(handle))
    return net


def plot_manifold(path):
    net = load_model(path)
    z_var = T.vector()
    decoder = theano.function([z_var], get_output(net["x_mu"], {net["z"]: z_var}))

    figure = plt.figure()

    Z01 = np.linspace(-1, 1, num=16)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    for (i, z_i) in enumerate(Zgrid, 1):
        figure.add_subplot(16, 16, i)
        image = decoder(z_i).reshape((28, 28))
        plt.axis('off')
        plt.imshow(image, cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(path.stem + "_manifold.png")))


def plot_sample(path):
    net = load_model(path)
    z_var = T.vector()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}))
    z_covar = theano.function(
        [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var})))

    figure = plt.figure()

    n_samples = 256
    [chunk] = re.findall(r"vae_mnist_L(\d+)_H(\d+)", str(path))
    num_latent, _ = map(int, chunk)

    z = np.random.normal(size=(n_samples, num_latent)).astype(theano.config.floatX)
    for i, z_i in enumerate(z, 1):
        mu, covar = z_mu(z_i), z_covar(z_i)
        x = np.random.normal(mu, covar)
        figure.add_subplot(16, 16, i)
        plt.axis("off")
        plt.imshow(x.reshape((28, 28)), cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(path.stem + "_sample.png")))


def main(num_latent, num_hidden, batch_size, num_epochs):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(batch_size, num_features, num_latent, num_hidden)

    vars = ["x_mu", "x_log_covar", "z", "z_mu", "z_log_covar"]
    x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var = get_output(
        [net[var] for var in vars],
        X_var, deterministic=False
    )
    elbo_train = elbo(X_var, x_mu_var, x_log_covar_var,
                      z_var, z_mu_var, z_log_covar_var)

    x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var = get_output(
        [net[var] for var in vars],
        X_var, deterministic=True
    )
    elbo_val = elbo(X_var, x_mu_var, x_log_covar_var,
                    z_var, z_mu_var, z_log_covar_var)

    params = get_all_params(concat([net["x_mu"], net["x_log_covar"]]),
                            trainable=True)
    updates = rmsprop(-elbo_train, params, learning_rate=1e-3)
    train_nelbo = theano.function([X_var], -elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val)

    print("Starting training...")
    train_errs = []
    val_errs = []
    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        train_err, train_batches = 0, 0
        for Xb, yb in iter_minibatches(X_train, y_train,
                                       batch_size=batch_size):
            train_err += train_nelbo(Xb)
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

    prefix = "vae_mnist_L{}_H{}".format(num_latent, num_hidden)
    pd.DataFrame.from_dict({
        "train_err": train_errs,
        "val_err": val_errs
    }).to_csv(prefix + ".csv", index=False)

    all_param_values = get_all_param_values(
        concat([net["x_mu"], net["x_log_covar"]]))
    with open(prefix + ".pickle", "wb") as handle:
        pickle.dump(all_param_values, handle)


def plot_errors(filename, plot_name='train_val_errors.png'):
    errors = np.genfromtxt(filename, delimiter=',')
    fig, ax = plt.subplots()
    ax.set_ylim([-100000, 20000])
    train_errors = errors[1:, 0]
    val_errors = errors[1:, 1]

    ax.scatter(range(len(train_errors)), train_errors, s=4, color="blue")
    ax.scatter(range(len(train_errors)), val_errors, s=4, color="red")
    fig.savefig(plot_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn VAE from MNIST data")
    parser.add_argument("-L", dest="num_latent", type=int, default=100)
    parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    parser.add_argument("-B", dest="batch_size", type=int, default=500)

    args = parser.parse_args()
    path = Path("vae_mnist_L8_H500.pickle")
    plot_sample(path)
    # main(**vars(args))
