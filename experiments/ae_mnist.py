import argparse
import pickle
import re
from collections import namedtuple
from pathlib import Path

import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import tanh
from lasagne.objectives import squared_error
from lasagne.updates import adam

from tomato.datasets import load_dataset
from tomato.utils import iter_minibatches, Stopwatch, Monitor


class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_hidden", "continuous"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("ae_{dataset}_B{batch_size}_E{num_epochs}_"
               "N{num_features}_H{num_hidden}_{flag}")
        return Path(fmt.format(flag="DC"[self.continuous], **self._asdict()))

    @classmethod
    def from_path(cls, path):
        [(dataset, *chunks, dc)] = re.findall(
            r"ae_(\w+)_B(\d+)_E(\d+)_N(\d+)_H(\d+)_([DC])", str(path))
        batch_size, num_epochs, num_features, num_hidden = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features,
                   num_hidden, continuous=dc == "C")


def build_model(p):
    net = {}
    net["enc_input"] = InputLayer((None, p.num_features))
    net["enc_hidden"] = DenseLayer(net["enc_input"], num_units=p.num_hidden,
                                   nonlinearity=tanh)
    net["dec_hidden"] = DenseLayer(net["enc_hidden"], num_units=p.num_hidden,
                                   nonlinearity=tanh)
    net["dec_output"] = DenseLayer(net["dec_hidden"], num_units=p.num_features,
                                   nonlinearity=tanh)
    return net


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
    X_output_var = get_output(net["dec_output"], X_var)
    params = get_all_params(net["dec_output"], trainable=True)
    mse_var = squared_error(X_var, X_output_var).mean()
    updates = adam(mse_var, params, learning_rate=2e-3)
    mse = theano.function([X_var], mse_var, updates=updates)

    print("Starting training...")
    monitor = Monitor(p.num_epochs)
    sw = Stopwatch()
    while monitor:
        with sw:
            train_err, train_batches = 0, 0
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_err += mse(Xb)
                train_batches += 1

            val_err, val_batches = 0, 0
            for Xb in iter_minibatches(X_val, p.batch_size):
                val_err += mse(Xb)
                val_batches += 1

        snapshot = get_all_param_values(net["dec_output"])
        monitor.report(snapshot, sw, train_err / train_batches,
                       val_err / val_batches)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(monitor.best, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn AE from data")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("dataset", type=str)
    fit_parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=500)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)
