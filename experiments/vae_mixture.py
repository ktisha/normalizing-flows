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
from matplotlib.patches import Ellipse
from theano.sandbox.rng_mrg import MRG_RandomStreams

from tomato.datasets import load_dataset
from tomato.layers import GaussianNoiseLayer
from tomato.plot_utils import plot_likelihood
from lasagne.utils import floatX as as_floatX

from tomato.potentials import Potential
from tomato.utils import bernoulli_logpmf, \
    iter_minibatches, Stopwatch, Monitor, mvn_std_logpdf, logsumexp, mvn_logpdf_weighted, kl_mvn_log_mvn_std, bernoulli, \
    bernoulli_logit_density, mvn_logpdf, mvn_log_logpdf

theano.config.floatX = 'float32'

import numpy as np
# np.random.seed(42)

class Params(namedtuple("Params", [
    "dataset", "batch_size", "num_epochs", "num_features",
    "num_latent", "num_hidden", "num_components", "continuous",
    "importance_weighted"
])):
    __slots__ = ()

    def to_path(self):
        fmt = ("vae_mixture_{dataset}_B{batch_size}_E{num_epochs}_"
               "N{num_features}_L{num_latent}_H{num_hidden}_N{num_components}_{flag}_{iw}")
        return Path(fmt.format(flag="DC"[self.continuous],
                               iw="NY"[self.importance_weighted],
                               **self._asdict()))

    @classmethod
    def from_path(cls, path):
        [(dataset, *chunks, dc, iw)] = re.findall(
            r"vae_mixture_(\w+)_B(\d+)_E(\d+)_N(\d+)_L(\d+)_H(\d+)_N(\d+)_([DC])_([NY])", str(path))
        (batch_size, num_epochs,
         num_features, num_latent, num_hidden, num_components) = map(int, chunks)
        return cls(dataset, batch_size, num_epochs, num_features, num_latent,
                   num_hidden, num_components, continuous=dc == "C", importance_weighted=iw == "Y")


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
                             nonlinearity=identity, b=bias)
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
    z_log_covar_vars = T.stacklists(get_output(rec_net["z_log_covars"], X_var, **kwargs))
    z_vars = get_output(rec_net["zs"], X_var, **kwargs)        # (n_components, batch_size, latent)
    z_weight_vars = get_output(rec_net["z_weights"], X_var, **kwargs).T      # (n_components, batch_size)

    logpxzs = []           # (n_comp, batch, features)
    for i in range(p.num_components):
        x_mu_var = get_output(gen_net["x_mu"], z_vars[i], **kwargs)  # (batch_size, features)
        if p.continuous:
            x_covar_var = get_output(gen_net["x_log_covar"], z_vars[i], **kwargs)  # (batch_size, features)
            logpxzs.append(mvn_log_logpdf(X_var, x_mu_var, x_covar_var))
        else:
            logpxzs.append(bernoulli_logit_density(X_var, x_mu_var))
    logpxz = T.stacklists(logpxzs)

    logqzxs = []
    z_log_covar_vars = T.exp(z_log_covar_vars)
    for i in range(p.num_components):
        logqzxs.append(mvn_logpdf_weighted(z_vars[i], z_mu_vars, z_log_covar_vars, z_weight_vars))
    logqzx = T.stacklists(logqzxs)

    z_vars = T.stacklists(z_vars)
    logpz = mvn_std_logpdf(z_vars)

    logw = (logpxz + logpz - logqzx)
    if not p.importance_weighted:
        logw = T.sum(T.mul(logw, z_weight_vars), axis=0)
    else:
        logw = logw + T.log(z_weight_vars)
        logw = logsumexp(logw, axis=0)

    return T.mean(
        logw
    )


def likelihood(X_var, gen_net, rec_net, p, n_samples=100, **kwargs):
    n_samples = int(n_samples)
    X_vars = T.tile(X_var, [n_samples, 1, 1])    # (n_samples, batch, features)
    s = T.shape(X_vars)
    batch_size = s[1]
    X_vars = T.reshape(X_vars, [s[0] * batch_size, s[2]])

    z_mu_vars = T.stacklists(get_output(rec_net["z_mus"], X_vars, **kwargs))  # (n_components, n_samples x batch_size, latent)
    z_log_covar_vars = T.stacklists(get_output(rec_net["z_log_covars"], X_vars, **kwargs))
    z_vars = get_output(rec_net["zs"], X_vars, **kwargs)
    z_vars = T.stacklists(z_vars)         # (n_components, n_samples x batch_size, latent)

    z_weight_vars = get_output(rec_net["z_weights"], X_vars, **kwargs).T  # (n_components, n_samples x batch_size)


    logpxzs = []  # (n_comp, n_samples, batch)
    for i in range(p.num_components):
        x_mu_var = get_output(gen_net["x_mu"], z_vars[i], **kwargs)  # (n_samples x batch_size, features)
        x_mu_var = T.reshape(x_mu_var, [n_samples, batch_size, p.num_features])
        logpxzs.append(bernoulli_logpmf(X_var, x_mu_var))
    logpxz = T.stacklists(logpxzs)

    logqzxs = []
    z_log_covar_vars = T.exp(z_log_covar_vars)
    for i in range(p.num_components):
        logqzxs.append(mvn_logpdf_weighted(z_vars[i], z_mu_vars, z_log_covar_vars, z_weight_vars))
    logqzx = T.stacklists(logqzxs)
    logqzx = T.reshape(logqzx, [p.num_components, n_samples, batch_size])

    z_vars = T.stacklists(z_vars)
    logpz = mvn_std_logpdf(z_vars)
    logpz = T.reshape(logpz, [p.num_components, n_samples, batch_size])

    logw = (logpxz + logpz - logqzx)  # (n_comp, n_samples, batch)
    z_weight_vars = T.reshape(z_weight_vars, [p.num_components, n_samples, p.batch_size])
    if not p.importance_weighted:
        logw = logw.dimshuffle(1, 0, 2)
        logw = logsumexp(logw, axis=0)
        logw = logw - T.log(T.cast(n_samples, theano.config.floatX))
        weights = z_weight_vars.dimshuffle(1, 0, 2)
        logw = T.sum(T.mul(logw, weights[0]), axis=0)
    else:
        logw = logw + T.log(z_weight_vars)
        logw = T.reshape(logw, [p.num_components * n_samples, p.batch_size])

        logw = logw - T.log(T.cast(n_samples, theano.config.floatX))
        logw = logsumexp(logw, axis=0)

    return T.mean(logw)


def load_model(path):
    print("Building model and compiling functions...")
    rec_net = build_rec_model(Params.from_path(str(path)))
    gen_net = build_gen_model(Params.from_path(str(path)))
    layers = []
    layers.extend(rec_net["zs"])
    layers.append(rec_net["z_weights"])
    layers.append(gen_net["dec_output"])
    with path.open("rb") as handle:
        set_all_param_values(layers, pickle.load(handle))
    return rec_net, gen_net


def train_model(X_train, X_val, p, train_bias):
    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    srng = MRG_RandomStreams(seed=123)
    X_bin = T.cast(T.le(srng.uniform(T.shape(X_var)), X_var), 'float32') if p.dataset != 'gauss' \
                                                                            and p.dataset != 'potential' else X_var

    rec_net = build_rec_model(p)
    gen_net = build_gen_model(p, train_bias)

    elbo_train = elbo(X_bin, gen_net, rec_net, p, deterministic=False)
    elbo_val = elbo(X_bin, gen_net, rec_net, p, deterministic=False)
    # likelihood_val = likelihood(X_bin, gen_net, rec_net, p, 3000/p.num_components, deterministic=False)

    layers = []
    layers.extend(rec_net["zs"])
    layers.append(rec_net["z_weights"])
    layers.append(gen_net["dec_output"])
    params = get_all_params(layers, trainable=True)

    updates = theano.gradient.grad(-elbo_train, params, disconnected_inputs="warn")
    updates = adam(updates, params, learning_rate=1e-3, epsilon=1e-4, beta1=0.99)
    train_nelbo = theano.function([X_var], elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], elbo_val)
    # val_likelihood = theano.function([X_var], likelihood_val)

    print("Starting training...")
    monitor = Monitor(p.num_epochs, stop_early=False)
    sw = Stopwatch()
    x_weights = get_output(rec_net["z_weights"], X_bin, deterministic=False)
    weights_func = theano.function([X_var], x_weights)

    while monitor:
        with sw:
            train_err, train_batches = 0, 0
            np.random.shuffle(X_train)
            for Xb in iter_minibatches(X_train, p.batch_size):
                train_err += train_nelbo(Xb)
                train_batches += 1

            weights = weights_func(X_train)
            w_max = np.max(weights, axis=1)
            print(np.isclose(w_max, np.ones(w_max.shape[0]), 0.01, 0.01).sum())
            counter = Counter(np.argmax(weights, axis=1))
            print(counter)
            # if len(counter) < p.num_components and monitor.epoch == 0:
            #     return False

            # if monitor.epoch % 100 == 0:
            val_err, val_batches, lhood = 0, 0, 0
            for Xb in iter_minibatches(X_val, p.batch_size):
                val_err += val_nelbo(Xb)
                # lhood += val_likelihood(Xb)
                val_batches += 1

        snapshot = get_all_param_values(layers)

        monitor.report(snapshot, sw, train_err / train_batches,
                       val_err / val_batches)

    path = p.to_path()
    monitor.save(path.with_suffix(".csv"))
    with path.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(monitor.best, handle)
    return True


def fit_model(**kwargs):
    print("Loading data...")
    X_train, X_val = load_dataset(kwargs["dataset"], kwargs["continuous"])
    train_mean = np.mean(X_train, axis=0)
    train_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    num_features = X_train.shape[1]  # XXX abstraction leak.
    p = Params(num_features=num_features, **kwargs)

    print(p)
    success = False
    while not success:
        success = train_model(X_train, X_val, p, train_bias)


def print_weights(X_var, X_train, y_train, rec_net):
    x_weights = get_output(rec_net["z_weights"], X_var, deterministic=False)
    weights_func = theano.function([X_var], x_weights)
    for y in set(y_train):
        mask = y_train == y
        X_vali = X_train[mask, :]
        weights = weights_func(X_vali)
        print("y " + str(y) + " len= " + str(weights.shape[0]))
        # print(weights)
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
    fit_parser.add_argument("-E", dest="num_epochs", type=int, default=5000)
    fit_parser.add_argument("-B", dest="batch_size", type=int, default=200)
    fit_parser.add_argument("-N", dest="num_components", type=int, default=2)
    fit_parser.add_argument("-c", dest="continuous", action="store_true",
                            default=True)
    fit_parser.add_argument("-iw", dest="importance_weighted", action="store_true",
                            default=False)
    fit_parser.set_defaults(command=fit_model)

    args = vars(parser.parse_args())
    command = args.pop("command")
    command(**args)

    ###  plot samples ###
    import matplotlib.pyplot as plt
    X_train, X_test = load_dataset(args['dataset'], False)
    plt.scatter(X_train[:, 0], X_train[:, 1], color="lightsteelblue", lw=.3, s=3, cmap=plt.cool)

    p = Params(num_features=2, **args)
    path = p.to_path().with_suffix(".pickle")
    rec_net, gen_net = load_model(path)
    num_samples = 2000
    z_var = T.matrix()
    Z = as_floatX(np.random.normal(size=(num_samples, 2)))

    x_mu = theano.function([z_var], get_output(gen_net["x_mu"], z_var, deterministic=False))
    mu = x_mu(Z)

    if p.continuous:
        x_covar = theano.function([z_var], T.exp(get_output(gen_net["x_log_covar"], z_var, deterministic=False)))
        covar = x_covar(Z)
        X_decoded = np.random.normal(mu, covar)
    else:
        X_decoded = mu

    X_var = T.matrix()
    x_weights = get_output(rec_net["z_weights"], X_var, deterministic=False)
    weights_func = theano.function([X_var], x_weights)
    X_weights = weights_func(X_decoded)
    component_index = np.argmax(X_weights, axis=1)
    w_max = np.max(X_weights, axis=1)
    print(np.isclose(w_max, np.ones(w_max.shape[0]), 0.01, 0.01).sum())
    print(Counter(component_index))

    colors = ['r', 'g', 'c', 'y']
    for i in range(p.num_components):
        compi = X_decoded[component_index == i]
        plt.scatter(compi[:, 0], compi[:, 1], color=colors[i], label="component " + str(i), lw=.3, s=3, cmap=plt.cool)

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()
    plt.savefig("reconstructed.png")

    plt.clf()

    ### plot latent space ###

    X_var = T.matrix()
    zs = theano.function([X_var], get_output(rec_net["zs"], X_var, deterministic=False))
    z_mus = theano.function([X_var], get_output(rec_net["z_mus"], X_var, deterministic=False))
    z_covars = theano.function([X_var], T.exp(get_output(rec_net["z_log_covars"], X_var, deterministic=False)))
    mus = z_mus(X_train)
    covars = z_covars(X_train)
    zz = zs(X_train)

    ax = plt.subplot(111, aspect='equal')

    plt.scatter(X_train[:, 0], X_train[:, 1], color='lightsteelblue', lw=.3, s=3, cmap=plt.cool)

    for j in range(p.num_components):
        z_latent = zz[j]
        # plt.scatter(z_latent[:, 1], z_latent[:, 0], color=colors[j], label="component " + str(j), lw=.3, s=3, cmap=plt.cool)

        x = z_latent[:, 1]
        y = z_latent[:, 0]
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_color(colors[j])
        ell.set_facecolor('none')

        ax.add_artist(ell)

    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
    plt.savefig("latent.png")
