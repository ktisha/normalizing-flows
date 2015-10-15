import numpy as np
import theano.tensor as T


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
