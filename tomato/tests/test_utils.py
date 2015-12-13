import numpy as np
from scipy.stats import multivariate_normal

from tomato.utils import mvn_log_logpdf, mvn_logpdf


def test_mvn_log_logpdf(monkeypatch):
    monkeypatch.setattr("tomato.utils.T", np)

    X = np.array([[1.5, 0.7], [1.3, 1.5]])
    mean = np.array([0.3, -0.2])
    covar = np.array([2., 1.])

    expected = multivariate_normal.logpdf(X, mean, np.diag(covar))
    assert np.allclose(mvn_logpdf(X, mean, covar), expected)
    assert np.allclose(mvn_log_logpdf(X, mean, np.log(covar)), expected)
