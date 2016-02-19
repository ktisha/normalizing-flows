import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.utils import floatX as as_floatX
from PIL import Image
from scipy import stats


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
