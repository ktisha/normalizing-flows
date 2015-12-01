import argparse
import pickle
import re
from pathlib import Path
from collections import namedtuple

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, \
    get_output, get_all_params, get_all_param_values, \
    set_all_param_values, concat
from lasagne.nonlinearities import identity, rectify, sigmoid
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam
from lasagne.utils import floatX as as_floatX

from tomato.datasets import load_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.layers import planar_flow
from tomato.plot_utils import plot_manifold, plot_sample
from tomato.utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches, \
    Stopwatch, Monitor

np.random.seed(42)


class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_latent", "num_hidden", "num_flows", "continuous"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("nf_{dataset}_B{batch_size}_E{num_epochs}_"
               "N{num_features}_L{num_latent}_H{num_hidden}_"
               "F{num_flows}_{flag}")
        return Path(fmt.format(flag="DC"[self.continuous], **self._asdict()))

    @classmethod
    def from_path(cls, path):
        [(dataset, *chunks, dc)] = re.findall(
            r"nf_(\w+)_B(\d+)_E(\d+)_N(\d+)_L(\d+)_H(\d+)_F(\d+)_([DC])", path)
        (batch_size, num_epochs,
         num_features, num_latent, num_hidden, num_flows) = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features, num_latent,
                   num_hidden, num_flows, continuous=dc == "C")


def build_model(p):
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

    net["z_k"], net["logdet"] = planar_flow(net["z"], p.num_flows)

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z_k"], num_units=p.num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=p.num_hidden,
                                    nonlinearity=rectify)

    if p.continuous:
        net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=p.num_features,
                                 nonlinearity=identity)
        net["x_log_covar"] = DenseLayer(net["dec_hidden2"],
                                        num_units=p.num_features,
                                        nonlinearity=identity)
        net["dec_output"] = concat([net["x_mu"], net["x_log_covar"]])
    else:
        net["x_mu"] = net["dec_output"] = DenseLayer(
            net["dec_hidden2"], num_units=p.num_features,
            nonlinearity=sigmoid)
    return net


def elbo(X_var, beta_var, net, p, **kwargs):
    x_mu_var = get_output(net["x_mu"], X_var, **kwargs)
    z_0_var = get_output(net["z"], X_var, **kwargs)
    z_k_var = get_output(net["z_k"], X_var, **kwargs)
    z_mu_var = get_output(net["z_mu"], X_var, **kwargs)
    z_log_covar_var = get_output(net["z_log_covar"], X_var, **kwargs)
    logdet_sum_var = sum(get_output(net["logdet"], X_var, **kwargs))

    if p.continuous:
        x_log_covar_var = get_output(net["x_log_covar"], X_var, **kwargs)
        logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
    else:
        logpxz = -binary_crossentropy(x_mu_var, X_var).sum(axis=1)

    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return T.mean(
        beta_var * (logpxz + mvn_std_logpdf(z_k_var))
        - (mvn_log_logpdf(z_0_var, z_mu_var, z_log_covar_var) - logdet_sum_var)
    )


def fit_model(**kwargs):
    X_train, X_val = load_dataset(kwargs["dataset"])
    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    beta_var = T.scalar("beta_t")  # Inverse temperature.
    net = build_model(p)

    elbo_train = elbo(X_var, beta_var, net, p, deterministic=False)
    elbo_val = elbo(X_var, beta_var, net, p, deterministic=True)

    params = get_all_params(net["dec_output"], trainable=True)
    updates = adam(-elbo_train, params)
    train_nelbo = theano.function([X_var, beta_var], -elbo_train,
                                  updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val,
                                givens={beta_var: as_floatX(1)})

    print("Starting training...")
    monitor = Monitor(p.num_epochs)
    sw = Stopwatch()
    while monitor:
        with sw:
            train_err, train_batches = 0, 0
            # Causes ELBO to go to infinity. Should investigate further.
            beta = min(1, 0.01 + float(monitor.epoch) / p.num_epochs)
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_err += train_nelbo(Xb, beta)
                train_batches += 1

            val_err, val_batches = 0, 0
            for Xb in iter_minibatches(X_val, p.batch_size):
                val_err += val_nelbo(Xb)
                val_batches += 1

        monitor.report(sw, train_err, train_batches, val_err, val_batches)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    all_param_values = get_all_param_values(
        concat([net["x_mu"], net["x_log_covar"]]))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(all_param_values, handle)


def load_model(path):
    print("Building model and compiling functions...")
    net = build_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        set_all_param_values(concat([net["x_mu"], net["x_log_covar"]]),
                             pickle.load(handle))
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn NF-VAE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-L", dest="num_latent", type=int, default=100)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-F", dest="num_flows", type=int, default=2)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    manifold_parser = subparsers.add_parser("manifold")
    manifold_parser.add_argument("path", type=Path)
    manifold_parser.add_argument("-N", dest="num_steps", type=int, default=16)
    manifold_parser.set_defaults(command=plot_manifold, load_model=load_model)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("path", type=Path)
    sample_parser.add_argument("-N", dest="num_samples", type=int, default=256)
    sample_parser.set_defaults(command=plot_sample, load_model=load_model,
                               load_params=Params.from_path)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)
