import argparse
import time
import pickle

import pandas as pd
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, concat
from lasagne.nonlinearities import rectify, identity
from lasagne.updates import adam

from .layers import PlanarFlowLayer, IndexLayer
from .datasets import load_mnist_dataset
from .layers import GaussianNoiseLayer
from .utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches


def build_model(batch_size, num_features, num_latent, num_hidden, num_flows):
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

    z = net["z"]
    logdet = []
    for _ in range(num_flows):
        flow_layer = PlanarFlowLayer(z)
        z = IndexLayer(flow_layer, 0)
        logdet.append(ListIndexLayer(flow_layer, 1))

    net["z_k"] = z
    net["logdet"] = logdet

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z_k"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                             nonlinearity=identity)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def elbo_nf(X_var, x_mu_var, x_log_covar_var, z_0_var, z_k_var, logdet_var):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return (
        mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
        + mvn_std_logpdf(z_k_var)
        - mvn_std_logpdf(z_0_var) + logdet_var
    ).mean()


def main(num_latent, num_hidden, num_flows, batch_size, num_epochs):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(batch_size, num_features, num_latent, num_hidden,
                      num_flows)

    vars = ["x_mu", "x_log_covar", "z", "z_k"]
    x_mu_var, x_log_covar_var, z_0_var, z_k_var, *logdet_var = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=False
    )
    elbo_train = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                         z_0_var, z_k_var, sum(logdet_var))

    x_mu_var, x_log_covar_var, z_0_var, z_k_var, *logdet_var = get_output(
        [net[var] for var in vars] + net["logdet"],
        X_var, deterministic=True
    )
    elbo_val = elbo_nf(X_var, x_mu_var, x_log_covar_var,
                       z_0_var, z_k_var, sum(logdet_var))

    params = get_all_params(concat([net["x_mu"], net["x_log_covar"]]),
                            trainable=True)
    updates = adam(-elbo_train, params)
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

    prefix = "nf_mnist_L{}_H{}_F{}".format(num_latent, num_hidden, num_flows)
    pd.DataFrame.from_dict({
        "train_err": train_errs,
        "val_err": val_errs
    }).to_csv(prefix + ".csv", index=False)

    all_param_values = get_all_param_values(
        concat([net["x_mu"], net["x_log_covar"]]))
    with open(prefix + ".pickle", "wb") as handle:
        pickle.dump(all_param_values, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn NF-VAE from MNIST data")
    parser.add_argument("-L", dest="num_latent", type=int, default=100)
    parser.add_argument("-H", dest="num_hidden", type=int, default=500)
    parser.add_argument("-F", dest="num_flows", type=int, default=2)
    parser.add_argument("-E", dest="num_epochs", type=int, default=1000)
    parser.add_argument("-B", dest="batch_size", type=int, default=500)

    args = parser.parse_args()
    main(**vars(args))
