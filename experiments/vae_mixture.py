import argparse
import pickle
import re
from collections import namedtuple, Counter
from pathlib import Path

import theano
import theano.tensor as T
from lasagne.init import Constant
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values, \
    concat
from lasagne.layers.merge import ConcatLayer
from lasagne.nonlinearities import identity, tanh, softmax, sigmoid
from lasagne.updates import adam
from theano.gradient import grad

from tomato.datasets import load_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.plot_utils import plot_manifold
from tomato.plot_utils import plot_sample
from tomato.utils import mvn_log_logpdf, \
    iter_minibatches, Stopwatch, Monitor, mvn_std_logpdf, bernoulli_logpmf, mvn_log_logpdf_weighted

theano.config.floatX = 'float64'


class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_latent", "num_hidden", "num_components", "continuous"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("vae_mixture_{dataset}_B{batch_size}_E{num_epochs}_"
               "N{num_features}_L{num_latent}_H{num_hidden}_N{num_components}_{flag}")
        return Path(fmt.format(flag="DC"[self.continuous], **self._asdict()))

    @classmethod
    def from_path(cls, path):
        [(dataset, *chunks, dc)] = re.findall(
            r"vae_(\w+)_B(\d+)_E(\d+)_N(\d+)_L(\d+)_H(\d+)_N(\d+)_([DC])", str(path))
        (batch_size, num_epochs,
         num_features, num_latent, num_hidden, num_components) = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features, num_latent,
                   num_hidden, num_components, continuous=dc == "C")


def build_model(p, bias=Constant(0)):
    net = {}
    # q(z|x)
    net["enc_input"] = InputLayer((None, p.num_features))
    net["enc_hidden"] = DenseLayer(net["enc_input"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    net["z_mus"] = z_mus = []
    net["z_log_covars"] = z_log_covars = []
    net["zs"] = zs = []
    for i in range(p.num_components):
        mu = DenseLayer(net["enc_hidden"], num_units=p.num_latent, nonlinearity=identity)
        z_mus.append(mu)
        covar = DenseLayer(net["enc_hidden"], num_units=p.num_latent, nonlinearity=identity)
        z_log_covars.append(covar)
        zs.append(GaussianNoiseLayer(mu, covar))

    net["z_weights"] = DenseLayer(net["enc_hidden"], num_units=p.num_components,
                                  nonlinearity=softmax, W=Constant(0))

    z_input = [net["z_weights"]]
    z_input.extend(zs)
    net["z"] = ConcatLayer(z_input)
    # q(x|z)
    net["dec_hidden"] = DenseLayer(net["z"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    net["x_mu"] = DenseLayer(net["dec_hidden"], num_units=p.num_features,
                             nonlinearity=sigmoid, b=bias)
    if p.continuous:
        net["x_log_covar"] = DenseLayer(net["dec_hidden"],
                                        num_units=p.num_features,
                                        nonlinearity=identity)

        net["dec_output"] = concat([net["x_mu"], net["x_log_covar"]])
    else:
        net["dec_output"] = net["x_mu"]
    return net


def elbo(X_var, net, p, **kwargs):
    x_mu_var = get_output(net["x_mu"], X_var, **kwargs)  # (input, features)
    if p.continuous:
        x_log_covar_var = get_output(net["x_log_covar"], X_var, **kwargs)  # (input, features)
        logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
    else:
        logpxz = bernoulli_logpmf(X_var, x_mu_var)

    z_mu_vars = T.stacklists(get_output(net["z_mus"], X_var, **kwargs))
    z_log_covar_vars = T.stacklists(get_output(net["z_log_covars"], X_var, **kwargs))
    z_vars = T.stacklists(get_output(net["zs"], X_var, **kwargs))
    z_weight_vars = get_output(net["z_weights"], X_var, **kwargs).T

    logpz = mvn_std_logpdf(z_vars).sum(axis=0)
    logqzx = (mvn_log_logpdf_weighted(z_vars, z_mu_vars, z_log_covar_vars, z_weight_vars) * z_weight_vars).sum(axis=0)

    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return T.mean(
        logpxz
        + logpz
        - logqzx
    )


def likelihood(X_var, net, p, n_samples, **kwargs):
    x_mu_var = get_output(net["x_mu"], X_var, **kwargs)  # (input, features)
    if p.continuous:
        x_log_covar_var = get_output(net["x_log_covar"], X_var, **kwargs)  # (input, features)
        logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
    else:
        logpxz = bernoulli_logpmf(X_var, x_mu_var)

    logpzs = []
    logqzxs = []
    for i in range(n_samples):
        z_mu_vars = T.stacklists(get_output(net["z_mus"], X_var, **kwargs))
        z_log_covar_vars = T.stacklists(get_output(net["z_log_covars"], X_var, **kwargs))
        z_vars = T.stacklists(get_output(net["zs"], X_var, **kwargs))
        z_weight_vars = get_output(net["z_weights"], X_var, **kwargs).T

        logpz = mvn_std_logpdf(z_vars).sum(axis=0)
        logpzs.append(logpz)
        logqzx = (mvn_log_logpdf_weighted(z_vars, z_mu_vars, z_log_covar_vars, z_weight_vars) * z_weight_vars).sum(axis=0)
        logqzxs.append(logqzx)

    logpz = T.stacklists(logpzs)
    logqzx = T.stacklists(logqzxs)
    logw = logpxz + logpz - logqzx

    max_w = T.max(logw, 0, keepdims=True)
    adjusted_w = logw - max_w
    ll = max_w + T.log(T.mean(T.exp(adjusted_w), 0, keepdims=True))
    return T.mean(ll)


def load_model(path):
    print("Building model and compiling functions...")
    net = build_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        set_all_param_values(net["dec_output"], pickle.load(handle))
    return net


def fit_model(**kwargs):
    print("Loading data...")
    X_train, X_val = load_dataset(kwargs["dataset"], kwargs["continuous"])
    train_mean = np.mean(X_train, axis=0)
    train_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(p, train_bias)

    elbo_train = elbo(X_var, net, p, deterministic=False)
    elbo_val = elbo(X_var, net, p, deterministic=True)

    params = get_all_params(net["dec_output"], trainable=True)

    updates = grad(-elbo_train, params, disconnected_inputs='warn')
    updates = adam(updates, params, learning_rate=1e-3, epsilon=1e-4, beta1=0.99)
    train_nelbo = theano.function([X_var], elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], elbo_val)
    validation_likelihood = theano.function([X_var], likelihood(X_var, net, p, 200))
    print("Starting training...")
    monitor = Monitor(p.num_epochs, stop_early=False)
    sw = Stopwatch()
    while monitor:
        with sw:
            train_err, train_batches = 0, 0
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_batches += 1
                train_err += (train_nelbo(Xb) - train_err) / train_batches

            val_err, val_batches, val_likelihood = 0, 0, 0
            for Xb in iter_minibatches(X_val, 50):
                val_batches += 1
                val_err += (val_nelbo(Xb) - val_err) / val_batches

                vlb = validation_likelihood(Xb)
                val_likelihood += (vlb - val_likelihood) / val_batches

        snapshot = get_all_param_values(net["dec_output"])
        monitor.report(snapshot, sw, train_err, val_err, val_likelihood)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(monitor.best, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn VAE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(42)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=2)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=100)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-N", dest="num_components", type=int, default=1)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    manifold_parser = subparsers.add_parser("manifold")
    manifold_parser.add_argument("path", type=Path)
    manifold_parser.add_argument("-N", dest="num_steps", type=int, default=32)
    manifold_parser.set_defaults(command=plot_manifold, load_model=load_model,
                                 load_params=Params.from_path)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("path", type=Path)
    sample_parser.add_argument("-N", dest="num_samples", type=int, default=256)
    sample_parser.set_defaults(command=plot_sample, load_model=load_model,
                               load_params=Params.from_path)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)

    # net = load_model(Path("mar9/vae_mixture_mnist_B500_E100_N784_L2_H500_N2_D.pickle"))
    # X_var = T.matrix()
    # X_train, X_val, y_train, y_val = load_dataset("mnist", False, True)
    # # x_weights = get_output(net["z_weights"], X_var, deterministic=True)
    # # weights_func = theano.function([X_var], x_weights)
    # # print(X_val.shape)
    # # for y in set(y_train):
    # #     print("y " + str(y))
    # #     mask = y_train == y
    # #     X_vali = X_train[mask, :]
    # #     weights = weights_func(X_vali)
    # #     # print(weights)
    # #     print(Counter(np.argmax(weights, axis=1)))
    #
    #
    # z_mu_function = get_output(net["z_mus"], X_var, deterministic=True)
    # z_log_function = get_output(net["z_log_covars"], X_var, deterministic=True)
    # z_mu = theano.function([X_var], z_mu_function)
    # z_covar = theano.function([X_var], z_log_function)
    #
    # mus = z_mu(X_val)
    # covars = np.exp(z_covar(X_val))
    # print(mus[0].shape)
    # # x1 = np.random.multivariate_normal(np.mean(mus[1], axis=0), np.diag(np.mean(covars[1], axis=0)), 1000)
    # # x2 = np.random.multivariate_normal(np.mean(mus[0], axis=0), np.diag(np.mean(covars[0], axis=0)), 1000)
    #
    # plt.scatter(mus[1][y_val == 0][:, 0], mus[1][y_val == 0][:, 1])
    # plt.scatter(mus[1][y_val == 1][:, 0], mus[1][y_val == 1][:, 1], c='r')
    # plt.scatter(mus[1][y_val == 2][:, 0], mus[1][y_val == 2][:, 1], c='g')
    # plt.scatter(mus[1][y_val == 3][:, 0], mus[1][y_val == 3][:, 1], c='m')
    # plt.scatter(mus[1][y_val == 4][:, 0], mus[1][y_val == 4][:, 1], c='c')
    # # plt.scatter(mus[1][:, 0], mus[1][:, 1], c='g')
    # # plt.ylim([0, 50])
    # # H, xedges, yedges = np.histogram2d(mus[1][:, 0], mus[1][:, 1], bins=100)
    # # H1, xedges1, yedges1 = np.histogram2d(mus[0][:, 0], mus[0][:, 1], bins=100)
    # # Hmasked = np.ma.masked_where(H == 0, H)
    # # Hmasked1 = np.ma.masked_where(H1 == 0, H1)
    # # plt.pcolormesh(xedges, yedges, Hmasked)
    # # plt.pcolormesh(xedges1, yedges1, Hmasked1)
    # plt.show()
    # for y in set(y_train):
    #     mask = y_train == y
    #     x1 = np.random.multivariate_normal(np.mean(mus[1][mask], axis=0), np.diag(np.mean(covars[1][mask], axis=0)), 1000)
    #     x2 = np.random.multivariate_normal(np.mean(mus[0][mask], axis=0), np.diag(np.mean(covars[0][mask], axis=0)), 1000)
    #     plt.scatter(x1[:, 0], x1[:, 1])
    #     plt.scatter(x2[:, 0], x2[:, 1], c='r')
    #     plt.show()
    #     plt.clf()
