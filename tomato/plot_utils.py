import re
import theano.tensor as T
import theano
from lasagne.layers import get_output
import matplotlib.pyplot as plt
import numpy as np
from lasagne.utils import floatX as as_floatX
from matplotlib import cm


def plot_manifold(path, load_model, bounds=[-8, 8], num_steps=32):
    net = load_model(path)
    z_var = T.matrix()
    decoder = theano.function([z_var], get_output(net["x_mu"], {net["z"]: z_var}))

    figure = plt.figure()

    Z01 = np.linspace(*bounds, num=num_steps)
    Zgrid = as_floatX(np.dstack(np.meshgrid(Z01, Z01)).reshape(-1, 2))

    for (i, z_i) in enumerate(Zgrid, 1):
        figure.add_subplot(num_steps, num_steps, i)
        image = decoder(np.array([z_i])).reshape((28, 28))
        plt.axis('off')
        plt.imshow(image, cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name("{}_manifold_{}.png".format(path.stem, num_steps))))


def plot_sample(path, load_model, num_samples, prefix):
    net = load_model(path)
    z_var = T.matrix()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var}))
    z_covar = theano.function(
        [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var})))

    figure = plt.figure()
    [chunk] = re.findall(prefix + r"(\d+)_H(\d+)", str(path))
    num_latent, _ = map(int, chunk)

    num_subplots = int(np.sqrt(num_samples))
    z = as_floatX(np.random.normal(size=(num_samples, num_latent)))
    for i, z_i in enumerate(z, 1):
        mu = z_mu(np.array([z_i]))
        covar = z_covar(np.array([z_i]))
        x = np.random.normal(mu, covar)
        figure.add_subplot(num_subplots, num_subplots, i)
        plt.axis("off")
        plt.imshow(x.reshape((28, 28)), cmap=cm.Greys)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(path.with_name("{}_sample_{}.png".format(path.stem, num_samples))))


def plot_errors(path):
    plot_name=str(path.with_name(path.stem + "_train_val_errors.png"))
    errors = np.genfromtxt(str(path), delimiter=',')
    fig, ax = plt.subplots()
    ax.set_ylim([-100000, 20000])
    train_errors = errors[1:, 0]
    val_errors = errors[1:, 1]

    ax.scatter(range(len(train_errors)), train_errors, s=4, color="blue", label="train error")
    ax.scatter(range(len(train_errors)), val_errors, s=4, color="red", label="test error")
    plt.legend()
    fig.savefig(plot_name)
