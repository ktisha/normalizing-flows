import gzip
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from lasagne.utils import floatX as as_floatX
from scipy.io import loadmat
import pickle

# Absolute path to the data directory.
DATA_ROOT = Path(__file__).parent / "data"

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)


def load_mnist_binarized():
    dataset = 'mnist.pkl.gz'
    datasetfolder = DATA_ROOT / 'mnist_binarized'

    def download(datapath):
        datafiles = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
        }
        datasplits = {}
        for split in datafiles.keys():
            print("Downloading %s data..." % split)
            datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

        f = gzip.open(str(datapath / dataset), 'w')
        pickle.dump([datasplits['train'], datasplits['valid'], datasplits['test']], f)

    if not datasetfolder.exists():
        datasetfolder.mkdir(parents=True)

    if not (datasetfolder / dataset).exists():
        download(datasetfolder)

    with gzip.open(str(datasetfolder / dataset), 'rb') as f:
        X_train, X_val, X_test = pickle.load(f)
    return X_train.astype(np.float32), X_val.astype(np.float32)


def load_mnist_dataset(continuous, returnLabels=False):
    def download(path):
        print("Downloading {}".format(path))
        urlretrieve("http://yann.lecun.com/exdb/mnist/" + path.name, str(path))

    def load_mnist_images(path):
        if not path.exists():
            download(path)

        with gzip.open(str(path)) as handle:
            data = (np.frombuffer(handle.read(), np.uint8, offset=16)
                    .reshape(-1, 28 * 28))

        data = data / as_floatX(255)  # Convert to [0, 1).
        if continuous:
            return data
        else:
            return np.random.binomial(n=1, p=data, size=data.shape).astype(np.float32)
            # return np.where(data <= 127, as_floatX(0), as_floatX(1))

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

    # Don't use test data (we use val for this) and labels.
    if returnLabels:
        return X_train, X_val, y_train, y_val
    return X_train, X_val


def load_frey_dataset(continuous):
    path = DATA_ROOT / "frey_rawface.mat"
    if not path.exists():
        urlretrieve("http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat",
                    str(path))

    data = loadmat(str(path))["ff"].T
    if continuous:
        X = data / as_floatX(255)  # Convert to [0, 1).
    else:
        X = np.where(data <= 127, as_floatX(0), as_floatX(1))

    X_train, X_val = X[:-500], X[-500:]
    return X_train, X_val


def load_dataset(name, continuous, returnLabels=False):
    if name == "mnist":
        return load_mnist_dataset(continuous, returnLabels)
    elif name == "mnist_bin":
        return load_mnist_binarized()
    elif name == "frey":
        return load_frey_dataset(continuous)
    else:
        raise ValueError(name)
