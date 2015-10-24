import gzip
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from lasagne.utils import floatX as as_floatX


# Absolute path to the data directory.
DATA_ROOT = Path(__file__).parent / "data"

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)


def load_mnist_dataset():
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
        data = data / as_floatX(256)  # Convert to [0, 1].
        return data.reshape(-1, 28 * 28)

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