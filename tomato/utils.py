import numpy as np
import theano.tensor as T

import time


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


def iter_minibatches(X, y, batch_size):
    assert len(X) == len(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(int(np.ceil(len(X) / batch_size))):
        lo = i * batch_size
        hi = (i + 1) * batch_size
        batch = indices[lo:hi]
        yield X[batch], y[batch]


class stopwatch:
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
