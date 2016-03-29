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
from lasagne.nonlinearities import identity, tanh, softmax, sigmoid

from lasagne.updates import adam


from tomato.datasets import load_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.plot_utils import plot_manifold, plot_sample, plot_full_histogram, plot_histogram_by_class, plot_mu_by_class, \
    plot_mu_by_components, plot_components_mean_by_components, plot_object_by_components
from tomato.utils import mvn_log_logpdf, bernoulli_logpmf,  \
    iter_minibatches, Stopwatch, Monitor, mvn_std_logpdf, mvn_log_logpdf_weighted, bernoulli, mvn_logvar_pdf, \
    mvn_log_std_weighted

theano.config.floatX = 'float64'

import numpy as np
# np.random.seed(42)

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
            r"vae_mixture_(\w+)_B(\d+)_E(\d+)_N(\d+)_L(\d+)_H(\d+)_N(\d+)_([DC])", str(path))
        (batch_size, num_epochs,
         num_features, num_latent, num_hidden, num_components) = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features, num_latent,
                   num_hidden, num_components, continuous=dc == "C")


def build_rec_model(p):
    net = {}
    # q(z|x)
    net["enc_input"] = InputLayer((None, p.num_features))
    net["enc_hidden"] = DenseLayer(net["enc_input"], num_units=p.num_hidden,
                                   nonlinearity=tanh)
    net["enc_hidden"] = DenseLayer(net["enc_hidden"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    net["z_weights"] = DenseLayer(net["enc_hidden"], num_units=p.num_components,
                                  nonlinearity=softmax, W=Constant(0))

    net["z_mus"] = z_mus = []
    net["z_log_covars"] = z_log_covars = []
    net["zs"] = zs = []
    for i in range(p.num_components):
        z_mu = DenseLayer(net["enc_hidden"], num_units=p.num_latent, nonlinearity=identity)
        z_mus.append(z_mu)
        z_log_covar = DenseLayer(net["enc_hidden"], num_units=p.num_latent, nonlinearity=identity)
        z_log_covars.append(z_log_covar)
        zs.append(GaussianNoiseLayer(z_mu, z_log_covar))

    return net


def build_gen_model(p, bias=Constant(0)):
    net = {}

    # q(x|z)
    net["z"] = InputLayer((None, p.num_latent))
    net["dec_hidden"] = DenseLayer(net["z"], num_units=p.num_hidden,
                                   nonlinearity=tanh)
    net["dec_hidden"] = DenseLayer(net["dec_hidden"], num_units=p.num_hidden,
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


def elbo(X_var, gen_net, rec_net, p, **kwargs):
    z_mu_vars = T.stacklists(get_output(rec_net["z_mus"], X_var, **kwargs))  # (n_components, batch_size, latent)
    z_log_covar_vars = T.stacklists(
        get_output(rec_net["z_log_covars"], X_var, **kwargs))  # (n_components, batch_size, latent)
    z_vars = T.stacklists(get_output(rec_net["zs"], X_var, **kwargs))  # (n_components, batch_size, latent)
    z_weight_vars = get_output(rec_net["z_weights"], X_var, **kwargs).T  # (n_components, batch_size)

    logpxzs = []           # (n_comp, batch, features)
    for i in range(p.num_components):
        x_mu_var = get_output(gen_net["x_mu"], z_vars[i], **kwargs)  # (input, features)
        logpxzs.append(bernoulli_logpmf(X_var, x_mu_var))

    logpxz = T.stacklists(logpxzs)

    logpz = mvn_std_logpdf(z_vars)

    logqzx = mvn_log_logpdf_weighted(z_vars, z_mu_vars, z_log_covar_vars, z_weight_vars)

    logw = (logpxz + logpz - logqzx)

    logw = T.sum(T.mul(logw, z_weight_vars), axis=0)

    return T.mean(
        logw
    )


def load_model(path):
    print("Building model and compiling functions...")
    rec_net = build_rec_model(Params.from_path(str(path)))
    gen_net = build_gen_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        param_values = pickle.load(handle)
        set_all_param_values(rec_net["zs"], param_values[:-12])
        set_all_param_values(rec_net["z_weights"], param_values[-12:-6])
        set_all_param_values(gen_net["dec_output"], param_values[-6:])
    return rec_net, gen_net


def fit_model(**kwargs):
    print("Loading data...")
    X_train, X_val = load_dataset(kwargs["dataset"], kwargs["continuous"])
    train_mean = np.mean(X_train, axis=0)
    train_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print(p)
    print("Building model and compiling functions...")
    X_var = T.matrix("X")

    rec_net = build_rec_model(p)
    gen_net = build_gen_model(p, train_bias)

    elbo_train = elbo(X_var, gen_net, rec_net, p, deterministic=False)
    elbo_val = elbo(X_var, gen_net, rec_net, p, deterministic=True)

    params = get_all_params(rec_net["zs"], trainable=True)
    params.extend(get_all_params(rec_net["z_weights"], trainable=True))
    params.extend(get_all_params(gen_net["dec_output"], trainable=True))

    updates = theano.gradient.grad(-elbo_train, params, disconnected_inputs="warn")
    updates = adam(updates, params, learning_rate=1e-3, epsilon=1e-4, beta1=0.99)
    train_nelbo = theano.function([X_var], elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], elbo_val)
    # validation_likelihood = theano.function([X_var], likelihood(X_var, net, p, 200))
    print("Starting training...")
    monitor = Monitor(p.num_epochs, stop_early=False)
    sw = Stopwatch()
    while monitor:
        with sw:
            train_err, train_batches = 0, 0
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_err += train_nelbo(Xb)
                train_batches += 1

            val_err, val_batches = 0, 0
            for Xb in iter_minibatches(X_val, p.batch_size):
                val_err += val_nelbo(Xb)
                val_batches += 1

        snapshot = get_all_param_values(rec_net["zs"])
        snapshot.extend(get_all_param_values(rec_net["z_weights"]))
        snapshot.extend(get_all_param_values(gen_net["dec_output"]))

        monitor.report(snapshot, sw, train_err / train_batches,
                       val_err / val_batches)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(monitor.best, handle)


def print_weights(X_var, X_train, y_train, rec_net):
    x_weights = get_output(rec_net["z_weights"], X_var, deterministic=True)
    weights_func = theano.function([X_var], x_weights)
    for y in set(y_train):
        mask = y_train == y
        X_vali = X_train[mask, :]
        weights = weights_func(X_vali)
        print("y " + str(y) + " len= " + str(weights.shape[0]))
        print(weights)
        w_max = np.max(weights, axis=1)
        print(np.isclose(w_max, np.ones(w_max.shape[0]), 0.01, 0.01).sum())
        print(Counter(np.argmax(weights, axis=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn VAE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=2)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=200)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=2)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-N", dest="num_components", type=int, default=2)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)

    # path = Path("vae_mixture_mnist_B500_E2_N784_L2_H200_N2_D.pickle")
    # rec_net, gen_net = load_model(path)
    # p = Params.from_path(str(path))
    # X_var = T.matrix()
    # X_train, X_val, y_train, y_val = load_dataset("mnist", False, True)
    # X_val = X_val[:10000]
    # y_val = y_val[:10000]
    #
    # print_weights(X_var, X_val, y_val, rec_net)

    # z_mu_function = get_output(rec_net["z_mus"], X_var, deterministic=True)
    # z_log_function = get_output(rec_net["z_log_covars"], X_var, deterministic=True)
    # z_mu = theano.function([X_var], z_mu_function)
    # z_covar = theano.function([X_var], z_log_function)
    # mus = z_mu(X_val)
    # covars = np.exp(z_covar(X_val))

    # plot_full_histogram(mus, covars, p.num_components)
    # plot_histogram_by_class(mus, covars, y_train, p.num_components)
    # plot_mu_by_class(mus, y_train, p.num_components)
    # plot_mu_by_components(mus, y_val, p.num_components)
    # plot_object_by_components(mus, covars, y_val, p.num_components)
