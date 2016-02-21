import copy
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T


def logaddexp(X, Y):
    """Accurately computes ``log(exp(X) + exp(Y))``."""
    XY_max = T.maximum(X, Y)
    XY_min = T.minimum(X, Y)
    return XY_max + T.log1p(T.exp(XY_min - XY_max))


def logsumexp(X, axis=None):
    X_max = T.max(X, axis=axis)
    acc = T.log(T.sum(T.exp(X - X_max), axis=axis))
    return X_max + acc


def mvn_logpdf(X, mean, covar):
    """Computes log-pdf of the multivariate normal with diagonal covariance."""
    return -.5 * (T.log(2 * np.pi)
                  + T.log(covar)
                  + T.square((X - mean)) / covar).sum(axis=1)


def mvn_log_logpdf(X, mean, log_covar):
    return -.5 * (T.log(2 * np.pi)
                  + log_covar
                  + T.square((X - mean)) / T.exp(log_covar)).sum(axis=1)


def mvn_log_logpdf_weighted(X, mean, log_covar, weights):
    inner = -.5 * (T.log(2 * np.pi) + log_covar + T.square(X - mean) / T.exp(log_covar)).sum(axis=2)
    inner = inner + T.log(weights)
    return logsumexp(inner, axis=0)


def mvn_std_logpdf(X):
    return -.5 * (T.log(2 * np.pi) + T.square(X)).sum(axis=1)


def bernoulli_logpmf(X, p):
    """Computes log-pdf of the multivariate Bernoulli.

    >>> import theano
    >>> import theano.tensor as T
    >>> X = T.matrix("X")
    >>> p = T.vector("p")
    >>> f = theano.function([X, p], bernoulli_logpmf(X, p),
    ...                     allow_input_downcast=True)

    >>> from scipy.stats import bernoulli
    >>> X = np.array([[0, 1], [0, 1]])
    >>> p = np.array([.25, .42])
    >>> np.allclose(bernoulli.logpmf(X, p).sum(axis=1), f(X, p))
    True
    """
    return -T.nnet.binary_crossentropy(p, X).sum(axis=1)


def kl_mvn_log_mvn_std(mean, log_covar):
    # See Appendix A in the VAE paper.
    return -.5 * ((1 + log_covar - T.square(mean) - T.exp(log_covar))
                  .sum(axis=1))


def iter_minibatches(X, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(int(np.ceil(len(X) / batch_size))):
        lo = i * batch_size
        hi = (i + 1) * batch_size
        batch = indices[lo:hi]
        yield X[batch]


class Stopwatch:
    def __init__(self):
        self.result = None

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *exc_info):
        self.result = time.perf_counter() - self.start_time
        del self.start_time

    def __str__(self):
        if self.result is None:
            return "unknown"
        else:
            return "{:.3f}s".format(self.result)


class Monitor:
    def __init__(self, num_epochs, tolerance=10, stop_early=True):
        self.epoch = 0
        self.num_epochs = num_epochs
        self.tolerance = tolerance
        self.stop_early = stop_early
        self.snapshots = deque(maxlen=tolerance)
        self.train_errs = []
        self.val_errs = []

    def __bool__(self):
        if self.epoch == self.num_epochs:
            return False

        if self.epoch < self.tolerance:
            return True

        if not self.stop_early:
            return True

        mean_val_err = np.mean(self.val_errs[-self.tolerance:])
        eps = 1 + np.sign(mean_val_err) * 0.5
        good_to_go = self.val_errs[-1] < mean_val_err * eps
        if not good_to_go:
            print("Stopped early: {} > {}"
                  .format(self.val_errs[-1],  mean_val_err))
        return bool(good_to_go)

    @property
    def best(self):
        epoch = np.argmin(self.val_errs[-self.tolerance:])
        print("Best achieved validation loss: {:.6f}"
              .format(self.val_errs[epoch]))

        return self.snapshots[epoch]

    def report(self, snapshot, sw, train_err, val_err):
        print("Epoch {} of {} took {}"
              .format(self.epoch + 1, self.num_epochs, sw))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        assert not np.isnan(train_err) and not np.isnan(val_err)
        self.snapshots.append(copy.deepcopy(snapshot))
        self.train_errs.append(train_err)
        self.val_errs.append(val_err)
        self.epoch += 1

    def save(self, path):
        np.savetxt(str(path),
                   np.column_stack([self.train_errs, self.val_errs]),
                   delimiter=",")
        _plot_errors(path)


def _plot_errors(path):
    errors = np.genfromtxt(str(path), delimiter=',')
    epochs = np.arange(len(errors) - 1)
    plt.plot(epochs, errors[1:, 0], "b-", label="Train")
    plt.plot(epochs, errors[1:, 1], "r-", label="Test")
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(path.stem + "_errors.png")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
