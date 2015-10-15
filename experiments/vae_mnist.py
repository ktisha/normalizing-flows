import time

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, concat
from lasagne.nonlinearities import rectify, identity
from lasagne.updates import rmsprop

from .datasets import load_mnist_dataset
from .layers import GaussianLayer
from .utils import mvn_log_logpdf, mvn_std_logpdf


def build_model(batch_size, num_features, num_latent=50, num_hidden=500):
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

    net["z"] = GaussianLayer(net["z_mu"], net["z_log_covar"])

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=num_hidden,
                                    nonlinearity=rectify)
    net["x_mu"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                             nonlinearity=identity)
    net["x_log_covar"] = DenseLayer(net["dec_hidden2"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def iter_minibatches(X, y, batch_size):
    assert len(X) == len(y)
    for i in range(len(X) // batch_size + 1):
        indices = np.random.choice(len(X), replace=False, size=batch_size)
        yield X[indices], y[indices]


def main(batch_size=500, n_epochs=500):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(batch_size, num_features)

    vars = ["z", "z_mu", "z_log_covar", "x_mu", "x_log_covar"]
    z_var, z_mu_var, z_log_covar_var, x_mu_var, x_log_covar_var = get_output(
        [net[var] for var in vars],
        X_var, deterministic=False
    )

    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    log_likelihood = (
        mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
        + mvn_std_logpdf(z_var)
        - mvn_log_logpdf(z_var, z_mu_var, z_log_covar_var)
    ).mean()

    params = get_all_params(concat([net["x_mu"], net["x_log_covar"]]),
                            trainable=True)
    updates = rmsprop(-log_likelihood, params, learning_rate=1e-3)
    train_nll = theano.function([X_var], -log_likelihood, updates=updates)

    print("Starting training...")
    for epoch in range(n_epochs):
        start_time = time.perf_counter()

        train_err, train_batches = 0, 0
        for Xb, yb in iter_minibatches(X_train, y_train,
                                       batch_size=batch_size):
            train_err += train_nll(Xb)
            train_batches += 1

        # This won't work with stochastic layers. See MNIST example
        # in Lasagne sources.
        # val_err, val_batches = 0, 0
        # for Xb, yb in iter_minibatches(X_val, y_val, batch_size=500):
        #     val_err += train_nll(Xb, yb)
        #     val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.perf_counter() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


if __name__ == "__main__":
    main()
