import time

import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, \
    get_all_params, concat
from lasagne.nonlinearities import rectify, identity
from lasagne.updates import rmsprop

from .layers import PlanarFlowLayer, ListIndexLayer
from .datasets import load_mnist_dataset
from .layers import GaussianNoiseLayer
from .utils import mvn_log_logpdf, mvn_std_logpdf, iter_minibatches


def build_model(batch_size, num_features, num_latent=100, num_hidden=500, nflows=2):
    net = {}

    # q(z|x)
    net["enc_input"] = InputLayer((batch_size, num_features))
    net["enc_hidden1"] = DenseLayer(net["enc_input"], num_units=num_hidden,
                                    nonlinearity=rectify)
    # net["enc_hidden2"] = DenseLayer(net["enc_hidden1"], num_units=num_hidden,
    #                                 nonlinearity=rectify)
    net["z_mu"] = DenseLayer(net["enc_hidden1"], num_units=num_latent,
                             nonlinearity=identity)
    net["z_log_covar"] = DenseLayer(net["enc_hidden1"], num_units=num_latent,
                                    nonlinearity=identity)

    net["z"] = GaussianNoiseLayer(net["z_mu"], net["z_log_covar"])

    z = net["z"]
    logdet = []
    for nflow in range(nflows):
        flow_layer = PlanarFlowLayer(z)
        z = ListIndexLayer(flow_layer, 0)
        logdet.append(ListIndexLayer(flow_layer, 1))

    net["z_k"] = z
    net["logdet"] = logdet

    # q(x|z)
    net["dec_hidden1"] = DenseLayer(net["z_k"], num_units=num_hidden,
                                    nonlinearity=rectify)
    # net["dec_hidden2"] = DenseLayer(net["dec_hidden1"], num_units=num_hidden,
    #                                 nonlinearity=rectify)
    net["x_mu"] = DenseLayer(net["dec_hidden1"], num_units=num_features,
                             nonlinearity=identity)
    net["x_log_covar"] = DenseLayer(net["dec_hidden1"], num_units=num_features,
                                    nonlinearity=identity)
    return net


def elbo(X_var, x_mu_var, x_log_covar_var, z_var, z_mu_var, z_log_covar_var):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return (
        mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
        + mvn_std_logpdf(z_var)
        - mvn_log_logpdf(z_var, z_mu_var, z_log_covar_var)
    ).mean()


def elbo_nf(X_var, x_mu_var, x_log_covar_var, z_0_var, z_k_var, logdet_var):
    # L(x) = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    return ( mvn_log_logpdf(X_var, x_mu_var, x_log_covar_var)
        + mvn_std_logpdf(z_k_var)
        - mvn_std_logpdf(z_0_var) + logdet_var
    ).mean()



def main(batch_size=500, n_epochs=500):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()
    num_features = X_train.shape[1]

    print("Building model and compiling functions...")
    X_var = T.matrix("X")
    net = build_model(batch_size, num_features)

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
    updates = rmsprop(-elbo_train, params, learning_rate=1e-3)
    train_nelbo = theano.function([X_var], -elbo_train, updates=updates)
    val_nelbo = theano.function([X_var], -elbo_val)

    print("Starting training...")
    for epoch in range(n_epochs):
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
            epoch + 1, n_epochs, time.perf_counter() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


if __name__ == "__main__":
    main()
