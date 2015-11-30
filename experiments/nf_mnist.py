import argparse
import pickle
import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import theano
from lasagne.objectives import binary_crossentropy

import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, \
    get_output, get_all_params, get_all_param_values, \
    set_all_param_values, concat
from lasagne.nonlinearities import identity, rectify, sigmoid
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX

from tomato.datasets import load_mnist_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.layers import planar_flow
from tomato.utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches, \
    stopwatch

np.random.seed(42)

def build_model(batch_size, num_features, num_latent, num_hidden, num_flows, continuous=False):
    activation = identity if continuous else sigmoid

    net = {}

    # q(z|x)
    net["enc_input"] = InputLayer((batch_size, num_features))
    net["enc_hidden1"] = DenseLayer(net["enc_input"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["enc_hidden2"] = DenseLayer(net["enc_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    # net["enc_hidden3"] = maxout(net["enc_hidden2"], pool_size)
    net["z_mu"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                             nonlinearity=identity)
    net["z_log_covar"] = DenseLayer(net["enc_hidden2"], num_units=num_latent,
                                    nonlinearity=identity)

    net["z"] = GaussianNoiseLayer(net["z_mu"], net["z_log_covar"])

    net["z_k"], net["logdet"] = planar_flow(net["z"], num_flows)

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z_k"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    # net["dec_hidden3"] = maxout(net["dec_hidden2"], pool_size)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                             nonlinearity=activation)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def elbo_nf(X_var, x_mu_var, x_log_covar_var,
            z_0_var, z_k_var, z_mu_var, z_log_covar_var,
            logdet_var, beta_t, continuous):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var).sum() if continuous \
        else -binary_crossentropy(x_mu_var, X_var).sum()
    return T.mean(
        beta_t * (logpxz
                  + mvn_std_logpdf(z_k_var).sum())
        - (mvn_log_logpdf(z_0_var, z_mu_var, z_log_covar_var).sum() - logdet_var)
    )

def fit_model(num_latent, num_hidden, num_flows, batch_size, num_epochs, continuous=False):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    beta_var = T.scalar("beta_t")  # Inverse temperature.
    net = build_model(batch_size, num_features, num_latent, num_hidden,
                      num_flows)

    vars = ["x_mu", "x_log_covar", "z", "z_k", "z_mu", "z_log_covar"]
    (x_mu_var, x_log_covar_var, z_0_var,
     z_k_var, z_mu_var, z_log_covar_var, *logdet_vars) = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=False
    )
    elbo_train = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                         z_0_var, z_k_var, z_mu_var, z_log_covar_var,
                         sum(logdet_vars), beta_var, continuous)

    (x_mu_var, x_log_covar_var, z_0_var,
     z_k_var, z_mu_var, z_log_covar_var, *logdet_vars) = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=True
    )
    elbo_val = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                       z_0_var, z_k_var, z_mu_var, z_log_covar_var,
                       sum(logdet_vars), beta_var, continuous)


    layer = concat([net["x_mu"], net["x_log_covar"]]) if continuous else [net["x_mu"]]
    params = get_all_params(layer, trainable=True)

    updates = adam(-elbo_train, params)
    train_nelbo = theano.function([X_var, beta_var], -elbo_train,
                                  updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val,
                                givens={beta_var: as_floatX(1)})

    print("Starting training...")
    sw = stopwatch()
    train_errs = []
    val_errs = []
    for epoch in range(num_epochs):
        with sw:
            train_err, train_batches = 0, 0
            # Causes ELBO to go to infinity. Should investigate further.
            # beta = min(1, 0.01 + float(epoch) / num_epochs)
            beta = 1
            for Xb, yb in iter_minibatches(X_train, y_train, batch_size):
                train_err += train_nelbo(Xb, beta)
                train_batches += 1

            val_err, val_batches = 0, 0
            for Xb, yb in iter_minibatches(X_val, y_val, batch_size):
                val_err += val_nelbo(Xb)
                val_batches += 1

        print("Epoch {} of {} took {}".format(epoch + 1, num_epochs, sw))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        assert not np.isnan(train_err) and not np.isnan(val_err)
        train_errs.append(train_err)
        val_errs.append(val_err)

    prefix = "nf_mnist_L{}_H{}_F{}".format(num_latent, num_hidden, num_flows)
    np.savetxt(prefix + ".csv", np.column_stack([train_errs, val_errs]),
               delimiter=",")

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


def plot_manifold(path, num_steps):
    net = load_model(path)
    z_var = T.matrix()
    decoder = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}))

    figure = plt.figure()

    Z01 = np.linspace(-8, 8, num=num_steps)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    for (i, z_i) in enumerate(Zgrid, 1):
        figure.add_subplot(num_steps, num_steps, i)
        image = decoder(np.array([z_i])).reshape((28, 28))
        plt.axis('off')
        plt.imshow(image, cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(
        "{}_manifold_{}.png".format(path.stem, num_steps))))


def plot_sample(path, num_samples):
    net = load_model(path)
    z_var = T.matrix()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}))
    z_covar = theano.function(
        [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var})))

    [chunk] = re.findall(r"L(\d+)_H(\d+)", str(path))
    num_latent, _ = map(int, chunk)

    figure = plt.figure()

    num_subplots = int(np.sqrt(num_samples))
    z = as_floatX(np.random.normal(size=(num_samples, num_latent)))
    for i, z_i in enumerate(z, 1):
        x = np.random.normal(z_mu(np.array([z_i])), z_covar(np.array([z_i])))
        figure.add_subplot(num_subplots, num_subplots, i)
        plt.axis("off")
        plt.imshow(x.reshape((28, 28)), cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name(
        "{}_sample_{}.png".format(path.stem, num_samples))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn NF-VAE from MNIST data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=100)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-F", dest="num_flows", type=int, default=2)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    fit_parser.set_defaults(command=fit_model)

    manifold_parser = subparsers.add_parser("manifold")
    manifold_parser.add_argument("path", type=Path)
    manifold_parser.add_argument("-N", dest="num_steps", type=int, default=16)
    manifold_parser.set_defaults(command=plot_manifold)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("path", type=Path)
    sample_parser.add_argument("-N", dest="num_samples", type=int, default=256)
    sample_parser.set_defaults(command=plot_sample)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)
