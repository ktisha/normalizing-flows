import numpy as np
import theano.tensor as T

import time

from .plot_utils import plot_errors


def logaddexp(X, Y):
    """Accurately computes ``log(exp(X) + exp(Y))``."""
    XY_max = T.maximum(X, Y)
    XY_min = T.minimum(X, Y)
    return XY_max + T.log1p(T.exp(XY_min - XY_max))


def mvn_logpdf(X, mean, covar):
    """Returns a theano expression representing the values of the log
    probability density function of the multivariate normal with diagonal
    covariance.

    >>> import theano
    >>> import theano.tensor as T
    >>> X = T.matrix("X")
    >>> mean = T.vector("mean")
    >>> covar = T.vector("covar")
    >>> f = theano.function([X, mean, covar], mvn_logpdf(X, mean, covar))

    >>> from scipy.stats import multivariate_normal
    >>> X = np.array([[-2, 0], [1, -4]])
    >>> mean, covar = np.array([-1, 1]), np.array([.4, .2])
    >>> np.allclose(multivariate_normal.logpdf(X, mean, np.diag(covar)),
    ...             f(X, mean, covar))
    True
    """
    return -.5 * (T.log(2 * np.pi)
                  + T.log(covar)
                  + T.square((X - mean)) / covar).sum(axis=1)


def mvn_log_logpdf(X, mean, log_covar):
    return -.5 * (T.log(2 * np.pi)
                  + log_covar
                  + T.square((X - mean)) / T.exp(log_covar)).sum(axis=1)


def mvn_std_logpdf(X):
    return -.5 * (T.log(2 * np.pi) + T.square(X)).sum(axis=1)


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
    def __init__(self, num_epochs, tolerance=10):
        self.epoch = 0
        self.num_epochs = num_epochs
        self.tolerance = tolerance
        self.train_errs = []
        self.val_errs = []

    def __bool__(self):
        if self.epoch == self.num_epochs:
            return False

        if self.epoch < self.tolerance:
            return True

        # If the loss is decreasing, just continue, otherwise stop if
        # it'd increased too much.
        mean_val_err = np.mean(self.val_errs[-self.tolerance:])
        print(self.val_errs[-1], mean_val_err)
        return bool(self.val_errs[-2] >= self.val_errs[-1] or
                    self.val_errs[-1] / mean_val_err < 1.5)

    def report(self, sw, train_err, train_batches, val_err, val_batches):
        print("Epoch {} of {} took {}".format(self.epoch + 1, self.num_epochs,
                                              sw))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        assert not np.isnan(train_err) and not np.isnan(val_err)
        self.train_errs.append(train_err)
        self.val_errs.append(val_err)
        self.epoch += 1

    def save(self, path):
        np.savetxt(str(path),
                   np.column_stack([self.train_errs, self.val_errs]),
                   delimiter=",")
        plot_errors(path)
