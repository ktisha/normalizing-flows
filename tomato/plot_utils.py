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
