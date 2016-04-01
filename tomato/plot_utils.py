import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.utils import floatX as as_floatX
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_manifold(path, load_model, load_params, num_steps):
    p = load_params(str(path))
    net = load_model(path)
    z_var = T.matrix()
    x_mu_function = get_output(net["x_mu"], {net["z"]: z_var}, deterministic=True)
    x_mu = theano.function([z_var], x_mu_function)

    images = []
    for i, j in np.ndindex(num_steps - 1, num_steps - 1):
        z_ij = as_floatX([[stats.norm.ppf((i + 1) / num_steps),
                           stats.norm.ppf((j + 1) / num_steps)]])
        mu = x_mu(z_ij)
        mu = np.where(mu <= 0.5, as_floatX(0), as_floatX(1))
        images.append(mu.reshape(28, -1))

    _plot_grid(
        path.with_name("{}_manifold_{}.png".format(path.stem, num_steps)),
        images, p.continuous)


def plot_sample(path, load_model, load_params, num_samples):
    p = load_params(str(path))
    net = load_model(path)
    z_var = T.matrix()
    x_mu_function = get_output(net["x_mu"], {net["z"]: z_var}, deterministic=True)
    x_mu = theano.function([z_var], x_mu_function)

    Z = as_floatX(np.random.normal(size=(num_samples, p.num_latent)))

    images = []
    if p.continuous:
        x_log_covar_function = get_output(net["x_log_covar"], {net["z"]: z_var}, deterministic=True)
        x_covar = theano.function([z_var], T.exp(x_log_covar_function))
        for z_i in Z:
            mu = x_mu(np.array([z_i]))
            covar = x_covar(np.array([z_i]))
            images.append(np.random.normal(mu, covar).reshape(28, -1))
    else:
        for z_i in Z:
            mu = x_mu(np.array([z_i]))
            mu = np.where(mu <= 0.5, as_floatX(0), as_floatX(1))
            images.append(mu.reshape(28, -1))

    _plot_grid(
        path.with_name("{}_sample_{}.png".format(path.stem, num_samples)),
        images, p.continuous)


def _image_from_array(data, continuous):
    return Image.fromarray((255 * data).clip(0, 255).astype(np.uint8))


def _plot_grid(path, images, continuous):
    num_subplots = int(np.sqrt(len(images)))
    height, width = images[0].shape
    im = Image.new("L", (width * num_subplots, height * num_subplots))
    for i, image in enumerate(images):
        x, y = divmod(i, num_subplots)
        im.paste(_image_from_array(image, continuous), (x * width, y * height))

    im.save(str(path))


def plot_components_mean_by_class(mus, covars, y_train, num_components):
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'orange', 'lightgreen', 'lightblue']
    colors = colors[:num_components]
    ax1 = None
    for y in set(y_train):
        mask = y_train == y
        ax1 = plt.subplot(1, len(set(y_train)), y + 1, sharex=ax1, sharey=ax1)
        plt.title("Class " + str(y))
        for n_comp, c in enumerate(colors):
            x1 = np.random.multivariate_normal(np.mean(mus[n_comp][mask], axis=0),
                                               np.diag(np.mean(covars[n_comp][mask], axis=0)), 1000)
            plt.scatter(x1[:, 0], x1[:, 1], c=c, label="comp " + str(n_comp))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_components_mean_by_components(mus, covars, y_train, num_components):
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'orange', 'lightgreen', 'lightblue']
    ax1 = None
    for n_comp in range(num_components):
        ax1 = plt.subplot(1, num_components, n_comp + 1, sharex=ax1, sharey=ax1)
        plt.title("Component " + str(n_comp))
        for i, y in enumerate(set(y_train)):
            mask = y_train == y
            x1 = np.random.multivariate_normal(np.mean(mus[n_comp][mask], axis=0),
                                               np.diag(np.mean(covars[n_comp][mask], axis=0)), 1000)
            plt.scatter(x1[:, 0], x1[:, 1], c=colors[i], label="class " + str(i))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_mu_by_class(mus, y_train, num_components):
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'orange', 'lightgreen', 'lightblue']
    colors = colors[:num_components]
    ax1 = None
    for y in set(y_train):
        ax1 = plt.subplot(1, len(set(y_train)), y + 1, sharex=ax1, sharey=ax1)
        plt.title("Class " + str(y))
        for n_comp, c in enumerate(colors):
            mask = y_train == y
            plt.scatter(mus[n_comp][mask][:, 0], mus[n_comp][mask][:, 1], c=c, label="comp " + str(n_comp))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_mu_by_components(mus, y_train, num_components):
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'orange', 'lightgreen', 'lightblue']
    index = 1
    ax1 = None
    for n_comp in range(num_components):
        ax1 = plt.subplot(1, num_components, index, sharex=ax1, sharey=ax1)
        plt.title("Component " + str(n_comp))
        for i, y in enumerate(set(y_train)):
            mask = y_train == y
            plt.scatter(mus[n_comp][mask][:, 0], mus[n_comp][mask][:, 1], c=colors[i], label="class " + str(i))
        index += 1
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_histogram_by_class(mus, covars, y_train, num_components):
    index = 1
    ax1 = None
    for n_component in range(num_components):
        xs = []
        for y in set(y_train):
            mask = y_train == y
            mus_ = mus[n_component][mask]
            covars_ = covars[n_component][mask]
            for i in range(mus_.shape[0]):
                x1 = np.random.multivariate_normal(mus_[i], np.diag(covars_[i]), 1000)
                xs.append(x1)

            xx = np.vstack(xs)
            H, xedges, yedges = np.histogram2d(xx[:, 0], xx[:, 1], bins=100)
            Hmasked = np.ma.masked_where(H == 0, H)
            ax1 = plt.subplot(num_components, len(set(y_train)), index, sharex=ax1, sharey=ax1)
            plt.title("Comp " + str(n_component) + "; class " + str(y))
            plt.pcolormesh(xedges, yedges, Hmasked)
            index += 1

    plt.show()


def plot_full_histogram(mus, covars, num_components):
    ax1 = None
    for k in range(num_components):
        xs = []
        for i in range(mus[k].shape[0]):
            x1 = np.random.multivariate_normal(mus[k][i], np.diag(covars[k][i]), 1000)
            xs.append(x1)

        xx = np.vstack(xs)
        H, xedges, yedges = np.histogram2d(xx[:, 0], xx[:, 1], bins=100)
        Hmasked = np.ma.masked_where(H == 0, H)
        ax1 = plt.subplot(1, num_components, k + 1, sharex=ax1, sharey=ax1)
        plt.title("Component " + str(k))
        plt.pcolormesh(xedges, yedges, Hmasked)

    plt.show()


def plot_object_by_components(mus, covars, y_train, num_components):
    ax1 = None
    object_number = 1
    mask = y_train == 1   # only first class
    for n_comp in range(num_components):
        ax1 = plt.subplot(1, num_components, n_comp + 1, sharex=ax1, sharey=ax1)
        plt.title("Component " + str(n_comp))
        musi = mus[n_comp][mask]
        covarsi = covars[n_comp][mask]
        x1 = np.random.multivariate_normal(musi[object_number],
                                           np.diag(covarsi[object_number]), 1000)
        plt.scatter(x1[:, 0], x1[:, 1])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
