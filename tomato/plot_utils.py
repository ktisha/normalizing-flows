import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.utils import floatX as as_floatX
from PIL import Image
from scipy import stats


def plot_manifold(path, load_model, num_steps=32):
    net = load_model(path)
    z_var = T.matrix()
    decoder = theano.function(
        [z_var],
        get_output(net["x_mu"], {net["z"]: z_var}, deterministic=True))

    eps = 1e-4
    images = []
    for i, j in np.ndindex(num_steps, num_steps):
        z_ij = stats.norm.ppf([[max(eps, i) / num_steps,
                                max(eps, j) / num_steps]])
        images.append(decoder(as_floatX(z_ij)).reshape(28, -1))

    _plot_grid(
        path.with_name("{}_manifold_{}.png".format(path.stem, num_steps)),
        images)


def plot_sample(path, load_model, load_params, num_samples):
    p = load_params(str(path))
    net = load_model(path)
    z_var = T.matrix()
    z_mu = theano.function(
        [z_var], get_output(net["x_mu"], {net["z"]: z_var},
                            deterministic=True))

    Z = as_floatX(np.random.normal(size=(num_samples, p.num_latent)))

    images = []
    if p.continuous:
        z_covar = theano.function(
            [z_var], T.exp(get_output(net["x_log_covar"], {net["z"]: z_var},
                                      deterministic=True)))
        for z_i in Z:
            mu = z_mu(np.array([z_i]))
            covar = z_covar(np.array([z_i]))
            images.append(np.random.normal(mu, covar).reshape(28, -1))
    else:
        for z_i in Z:
            mu = z_mu(np.array([z_i]))
            images.append(np.random.binomial(1, mu).reshape(28, -1))

    _plot_grid(
        path.with_name("{}_sample_{}.png".format(path.stem, num_samples)),
        images)


def _image_from_array(data):
    return Image.fromarray(
        (255.0 / data.max() * (data - data.min())).astype(np.uint8))


def _plot_grid(path, images):
    num_subplots = int(np.sqrt(len(images)))
    height, width = images[0].shape
    im = Image.new("L", (width * num_subplots, height * num_subplots))
    for i, image in enumerate(images):
        x, y = divmod(i, num_subplots)
        im.paste(_image_from_array(image), (x * width, y * height))

    im.save(str(path))
