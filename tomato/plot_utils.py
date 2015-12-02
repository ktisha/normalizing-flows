import re

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.utils import floatX as as_floatX
from matplotlib import cm
from matplotlib.gridspec import GridSpec


def plot_manifold(path, load_model, bounds=(-4, 4), num_steps=32):
    net = load_model(path)
    z_var = T.matrix()
    decoder = theano.function(
        [z_var],
        get_output(net["x_mu"], {net["z"]: z_var}, deterministic=True))

    Z01 = np.linspace(*bounds, num=num_steps)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    images = []
    for z_i in Zgrid:
        images.append(decoder(np.array([z_i])).reshape((28, -1)))

    _plot_grid(
        path.with_name("{}_manifold_{}.png".format(path.stem, num_steps)),
        images)


def plot_sample(path, load_model, load_params, num_samples):
    p = load_params(str(path))
    net = load_model(path)
    z_var = T.matrix()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}, deterministic=True))

    Z = as_floatX(np.random.normal(size=(num_samples, p.num_latent)))

    images = []
    if p.continuous:
        z_covar = theano.function(
            [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var},
                                      deterministic=True)))
        for z_i in Z:
            mu = z_mu(np.array([z_i]))
            covar = z_covar(np.array([z_i]))
            images.append(np.random.normal(mu, covar).reshape((28, -1)))
    else:
        for z_i in Z:
            mu = z_mu(np.array([z_i]))
            images.append(np.random.binomial(1, mu).reshape(28, -1))

    _plot_grid(
        path.with_name("{}_sample_{}.png".format(path.stem, num_samples)),
        images)


def _plot_grid(path, images):
    num_subplots = int(np.sqrt(len(images)))
    gs = GridSpec(num_subplots, num_subplots)
    gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0.4, bottom=0.1, top=0.9)
    for i, image in enumerate(images):
        plt.subplot(gs[i])
        plt.imshow(image, cmap=cm.Greys_r)
        plt.axis("off")

    plt.savefig(str(path), bbox_inches="tight")


def plot_errors(path):
    errors = np.genfromtxt(str(path), delimiter=',')
    epochs = np.arange(len(errors) - 1)
    plt.plot(epochs, errors[1:, 0], "b-", label="Train")
    plt.plot(epochs, errors[1:, 1], "r-", label="Test")
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(path.stem + "_errors.png")
