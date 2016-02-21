import argparse
import pickle
import re
from collections import namedtuple
from pathlib import Path

import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values, \
    concat
from lasagne.nonlinearities import identity, tanh, softmax, sigmoid
from lasagne.updates import adam
from theano.gradient import grad

from tomato.datasets import load_dataset
from tomato.layers import GMMNoiseLayer
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


def build_model(p):
    net = {}
    # q(z|x)
    net["enc_input"] = InputLayer((None, p.num_features))
    net["enc_hidden"] = DenseLayer(net["enc_input"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    comps = []
    for i in range(p.num_components):
        net["z_mu" + str(i)] = DenseLayer(net["enc_hidden"], num_units=p.num_latent,
                                          nonlinearity=identity)
        net["z_log_covar" + str(i)] = DenseLayer(net["enc_hidden"], num_units=p.num_latent,
                                                 nonlinearity=identity)

    net["z_weights"] = DenseLayer(net["enc_hidden"], num_units=p.num_components,
                                          nonlinearity=softmax)

    comps.extend([net["z_mu" + str(i)] for i in range(p.num_components)])
    comps.extend([net["z_log_covar" + str(i)] for i in range(p.num_components)])
    comps.append(net["z_weights"])

    net["z"] = GMMNoiseLayer(comps, p.num_components)

    # q(x|z)
    net["dec_hidden"] = DenseLayer(net["z"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    net["x_mu"] = DenseLayer(net["dec_hidden"], num_units=p.num_features,
                             nonlinearity=sigmoid)
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

    z_var = get_output(net["z"], X_var, **kwargs)  # (input, latent)
    logpz = mvn_std_logpdf(z_var)

    z_mu_vars = T.stacklists(get_output([net["z_mu" + str(i)] for i in range(p.num_components)], X_var, **kwargs))
    z_log_covar_vars = T.stacklists(
        get_output([net["z_log_covar" + str(i)] for i in range(p.num_components)], X_var, **kwargs))
    z_weight_vars = get_output(net["z_weights"], X_var, **kwargs).T

    z_weight_vars = theano.gradient.zero_grad(z_weight_vars)

    logqzx = mvn_log_logpdf_weighted(z_var, z_mu_vars, z_log_covar_vars, z_weight_vars)

    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return T.mean(
        logpxz
        + logpz
        - logqzx
    )


def load_model(path):
    print("Building model and compiling functions...")
    net = build_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        set_all_param_values(net["dec_output"], pickle.load(handle))
    return net


def fit_model(**kwargs):
    print("Loading data...")
    X_train, X_val = load_dataset(kwargs["dataset"], kwargs["continuous"])
    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(p)

    elbo_train = elbo(X_var, net, p, deterministic=False)
    elbo_val = elbo(X_var, net, p, deterministic=True)

    params = get_all_params(net["dec_output"], trainable=True)

    updates = grad(-elbo_train, params, disconnected_inputs='warn')
    updates = adam(updates, params, learning_rate=1e-3)
    train_nelbo = theano.function([X_var], -elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val)

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

        snapshot = get_all_param_values(net["dec_output"])
        monitor.report(snapshot, sw, train_err / train_batches,
                       val_err / val_batches)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(monitor.best, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn VAE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=2)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=300)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-N", dest="num_components", type=int, default=10)
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
