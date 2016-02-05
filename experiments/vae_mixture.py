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
from tomato.utils import mvn_log_logpdf,  \
    iter_minibatches, Stopwatch, Monitor, mvn_std_logpdf, normalize, logsumexp, mvn_log_logpdf_weighted

theano.config.floatX = 'float64'

class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_latent", "num_hidden", "num_components", "continuous"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("vae_{dataset}_B{batch_size}_E{num_epochs}_"
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
        net["z_weight" + str(i)] = DenseLayer(net["enc_hidden"], num_units=1,
                                                 nonlinearity=identity)

    comps.extend([net["z_mu" + str(i)] for i in range(p.num_components)])
    comps.extend([net["z_log_covar" + str(i)] for i in range(p.num_components)])
    comps.extend([net["z_weight" + str(i)] for i in range(p.num_components)])

    net["z"] = GMMNoiseLayer(comps, p.num_components)

    # q(x|z)
    net["dec_hidden"] = DenseLayer(net["z"], num_units=p.num_hidden,
                                   nonlinearity=tanh)

    net["x_mu"] = DenseLayer(net["dec_hidden"], num_units=p.num_features,
                             nonlinearity=identity)
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
    x_log_covar_var = get_output(net["x_log_covar"], X_var, **kwargs)  # (input, features)
    logpxz = mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)

    z_var = get_output(net["z"], X_var, **kwargs)  # (input, latent)
    logpz = mvn_std_logpdf(z_var)

    z_mu_vars = get_output([net["z_mu" + str(i)] for i in range(p.num_components)], X_var, **kwargs)
    z_log_covar_vars = get_output([net["z_log_covar" + str(i)] for i in range(p.num_components)], X_var, **kwargs)
    z_weight_vars = get_output([net["z_weight" + str(i)] for i in range(p.num_components)], X_var, **kwargs)

    z_weight_vars = T.stacklists(z_weight_vars)
    z_weight_vars = theano.gradient.zero_grad(z_weight_vars)
    z_weight_vars = T.addbroadcast(z_weight_vars, 2)
    z_weight_vars = normalize(z_weight_vars)

    logqzx = mvn_log_logpdf_weighted(z_var, z_mu_vars, z_log_covar_vars, z_weight_vars)

    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return T.mean(
        logpxz.sum()
        + logpz.sum()
        - logqzx.sum()
    )


def load_model(path):
    print("Building model and compiling functions...")
    net = build_model(Params.from_path(str(path)))
    with path.open("rb") as handle:
        set_all_param_values(net["dec_output"], pickle.load(handle))
    return net


def fit_model(X_train, X_val, p):

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

    return net


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from lasagne.utils import floatX as as_floatX

    dataset = "gauss"
    np.random.seed(123)

    if dataset == "gauss":
        N = 100000

        X = np.concatenate([np.random.multivariate_normal(np.array([100, 100]), np.random.random_sample((2, 2)), [N/2]),
                            # np.random.multivariate_normal(np.array([1000,1000]), np.random.random_sample((2, 2)), [N/2]),
                            np.random.multivariate_normal(np.array([-100,-100]), np.random.random_sample((2, 2)), [N/2])])

        np.random.shuffle(X)
        X_train = X[:2*N/3, :]
        X_val = X[2*N/3:, :]

        num_features = X_train.shape[1]
        p = Params(num_features=num_features, dataset=dataset, batch_size=50, num_epochs=100,
                   num_latent=10, num_hidden=500, continuous=True, num_components=2)

        path = Path(str(p.to_path()) + ".pickle")
        net = fit_model(X_train, X_val, p)
        # net = load_model(path)

        f = plt.figure()
        plt.scatter(X[:, 0], X[:, 1], lw=.3, s=3, cmap=plt.cm.cool)

        #  plot samples
        num_samples = 2000
        z_var = T.matrix()
        x_mu = theano.function([z_var], get_output(net["x_mu"], {net["z"]: z_var},
                                                   deterministic=False))

        Z0 = np.concatenate([np.random.normal(loc=40, size=(num_samples/2, p.num_latent)),
                             # np.random.normal(loc=300, size=(num_samples/2, p.num_latent)),
                             np.random.normal(loc=-100, size=(num_samples/2, p.num_latent))])
        Z = as_floatX(Z0)

        # Z = as_floatX(np.random.normal(size=(num_samples, p.num_latent)))
        x_covar = theano.function([z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var},
                                                            deterministic=False)))
        X_decoded = []
        for z_i in Z:
            mu = x_mu(np.array([z_i]))
            print(mu)
            covar = x_covar(np.array([z_i]))

            x_i = np.random.normal(mu, covar)
            X_decoded.append(x_i)

        X_decoded = np.array(X_decoded).reshape((len(X_decoded), 2))

        plt.scatter(X_decoded[:, 0], X_decoded[:, 1], color="red", lw=.3, s=3, cmap=plt.cm.cool)
        plt.show()
        plt.savefig("x.png")

    else:
        X_train, X_val = load_dataset(dataset, True)
        num_features = X_train.shape[1]
        p = Params(num_features=num_features, dataset=dataset, batch_size=500, num_epochs=100,
                   num_latent=100, num_hidden=500, continuous=True, num_components=10)
        path = Path(str(p.to_path()) + ".pickle")

        net = fit_model(X_train, X_val, p)
        # net = load_model(path)
        plot_sample(path, load_model, p.from_path, 10)
        plot_manifold(path, load_model, p.from_path, 10)