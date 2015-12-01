import argparse
import pickle
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values, \
    concat
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import rectify, identity, sigmoid, tanh
from lasagne.updates import adam

from tomato.datasets import load_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.plot_utils import plot_manifold, plot_sample
from tomato.utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches, \
    stopwatch

np.random.seed(42)


class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_latent", "num_hidden", "continuous"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("vae_{dataset}_B{batch_size}_E{num_epochs}_"
               "N{num_features}_L{num_latent}_H{num_hidden}_{flag}")
        return fmt.format(flag="DC"[self.continuous], **self._asdict())

    @classmethod
    def from_path(cls, path):
        [(dataset, *chunks, dc)] = re.findall(
            r"vae_(\w+)_B(\d+)_E(\d+)_N(\d+)_L(\d+)_H(\d+)_([DC])", path)
        (batch_size, num_epochs,
         num_features, num_latent, num_hidden) = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features, num_latent,
                   num_hidden, continuous=dc == "C")


def build_model(p):
    activation = identity if p.continuous else sigmoid

    net = {}
    # q(z|x)
    net["enc_input"] = InputLayer((p.batch_size, p.num_features))
    net["enc_hidden1"] = DenseLayer(net["enc_input"], num_units=p.num_hidden,
                                    nonlinearity=rectify)
    net["enc_hidden2"] = DenseLayer(net["enc_hidden1"], num_units=p.num_hidden,
                                    nonlinearity=rectify)
    net["z_mu"] = DenseLayer(net["enc_hidden2"], num_units=p.num_latent,
                             nonlinearity=identity)
    net["z_log_covar"] = DenseLayer(net["enc_hidden2"], num_units=p.num_latent,
                                    nonlinearity=identity)

    net["z"] = GaussianNoiseLayer(net["z_mu"], net["z_log_covar"])

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z"], num_units=p.num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=p.num_hidden,
                                    nonlinearity=rectify)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=p.num_features,
                             nonlinearity=activation)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"],
                                    num_units=p.num_features,
                                    nonlinearity=identity)
    return net


def elbo(X_var, x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var,
         continuous=False):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var) if continuous \
        else -binary_crossentropy(x_mu_var, X_var).sum(axis=1)
    return T.mean(
        logpxz
        + mvn_std_logpdf(z_var)
        - mvn_log_logpdf(z_var, z_mu_var, z_log_covar_var)
    )


def load_model(path):
    print("Building model and compiling functions...")
    net = build_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        set_all_param_values(concat([net["x_mu"], net["x_log_covar"]]),
                             pickle.load(handle))
    return net


def fit_model(**kwargs):
    X_train, X_val = load_dataset(kwargs["dataset"])
    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print("Loading data...")
    X_train, X_val = load_dataset(p.dataset)

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(p)

    vars = ["x_mu", "x_log_covar", "z", "z_mu", "z_log_covar"]
    x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var = get_output(
        [net[var] for var in vars],
        X_var, deterministic=False
    )

    elbo_train = elbo(X_var, x_mu_var, x_log_covar_var,
                      z_var, z_mu_var, z_log_covar_var, p.continuous)

    x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var = get_output(
        [net[var] for var in vars],
        X_var, deterministic=True
    )
    elbo_val = elbo(X_var, x_mu_var, x_log_covar_var,
                    z_var, z_mu_var, z_log_covar_var, p.continuous)

    layer = (concat([net["x_mu"], net["x_log_covar"]]) if p.continuous else
             [net["x_mu"]])
    params = get_all_params(layer, trainable=True)

    updates = adam(-elbo_train, params, learning_rate=1e-3)

    train_nelbo = theano.function([X_var], -elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val)

    print("Starting training...")
    sw = stopwatch()
    train_errs = []
    val_errs = []
    for epoch in range(p.num_epochs):
        with sw:
            train_err, train_batches = 0, 0
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_err += train_nelbo(Xb)
                train_batches += 1

            val_err, val_batches = 0, 0
            for Xb in iter_minibatches(X_val, p.batch_size):
                val_err += val_nelbo(Xb)
                val_batches += 1

        print("Epoch {} of {} took {}".format(epoch + 1, p.num_epochs, sw))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        assert not np.isnan(train_err) and not np.isnan(val_err)
        train_errs.append(train_err)
        val_errs.append(val_err)

    prefix = p.to_path()
    np.savetxt(prefix + ".csv", np.column_stack([train_errs, val_errs]),
               delimiter=",")

    all_param_values = get_all_param_values(
        concat([net["x_mu"], net["x_log_covar"]]))
    with open(prefix + ".pickle", "wb") as handle:
        pickle.dump(all_param_values, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn VAE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=100)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    manifold_parser = subparsers.add_parser("manifold")
    manifold_parser.add_argument("path", type=Path)
    manifold_parser.add_argument("-N", dest="num_steps", type=int, default=32)
    manifold_parser.set_defaults(command=plot_manifold, load_model=load_model)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("path", type=Path)
    sample_parser.add_argument("-N", dest="num_samples", type=int, default=256)
    sample_parser.set_defaults(command=plot_sample, load_model=load_model,
                               load_params=Params.from_path)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)
