import gzip
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output, get_all_params
from lasagne.nonlinearities import sigmoid, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum


# Absolute path to the data directory.
DATA_ROOT = Path(__file__).parent / "data"

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)


def load_dataset():
    def download(path):
        print("Downloading {}".format(path))
        urlretrieve("http://yann.lecun.com/exdb/mnist/" + path.name, str(path))

    def load_mnist_images(path):
        if not path.exists():
            download(path)

        with gzip.open(str(path)) as handle:
            data = np.frombuffer(handle.read(), np.uint8, offset=16)

        # The inputs are vectors now, we reshape them to monochrome 28x28
        # images, following the shape convention:
        # (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)  # Convert to [0, 1].

    def load_mnist_labels(path):
        if not path.exists():
            download(path)

        with gzip.open(str(path)) as handle:
            return np.frombuffer(handle.read(), np.uint8, offset=8)

    X_train = load_mnist_images(DATA_ROOT / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(DATA_ROOT / "train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images(DATA_ROOT / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(DATA_ROOT / "t10k-labels-idx1-ubyte.gz")

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_var=None):
    net = {}
    net["input"] = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    net["hidden"] = DenseLayer(net["input"], num_units=800,
                               nonlinearity=sigmoid)
    net["output"] = DenseLayer(net["hidden"], num_units=10,
                               nonlinearity=softmax)
    return net["output"]


def iter_minibatches(X, y, *, batch_size):
    assert len(X) == len(y)
    for i in range(len(X) // batch_size + 1):
        indices = np.random.choice(len(X), replace=False, size=batch_size)
        yield X[indices], y[indices]


def main(n_epochs=500):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Building model and compiling functions...")
    X_var = T.tensor4("X")
    y_var = T.ivector("y")
    network = build_model(X_var)
    loss = categorical_crossentropy(get_output(network), y_var).mean()

    params = get_all_params(network, trainable=True)
    updates = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    train = theano.function([X_var, y_var], loss, updates=updates)

    print("Starting training...")
    for epoch in range(n_epochs):
        start_time = time.perf_counter()

        train_err, train_batches = 0, 0
        for Xb, yb in iter_minibatches(X_train, y_train, batch_size=500):
            train_err += train(Xb, yb)
            train_batches += 1

        val_err, val_batches = 0, 0
        for Xb, yb in iter_minibatches(X_val, y_val, batch_size=500):
            # This won't work with stochastic layers. See MNIST example
            # in Lasagne sources.
            val_err += train(Xb, yb)
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.perf_counter() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


if __name__ == "__main__":
    main()
