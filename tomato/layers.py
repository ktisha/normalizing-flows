import numpy as np
import theano.tensor as T
from lasagne.init import Normal
from lasagne.layers import Layer, MergeLayer
from lasagne.random import get_rng
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import softplus

from tomato.utils import normalize


class PlanarFlowLayer(Layer):
    def __init__(self, incoming, W=Normal(), U=Normal(), b=Normal(),
                 **kwargs):
        super().__init__(incoming, **kwargs)

        n_inputs = self.input_shape[1]
        self.W = self.add_param(W, (n_inputs, ), "W")
        self.U = self.add_param(U, (n_inputs, ), "U")
        self.b = self.add_param(b, (), "b")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        Z = input

        wTu = self.W.dot(self.U)
        m_wTu = -1 + softplus(wTu)
        U_hat = self.U + (m_wTu - wTu) * (self.W / self.W.norm(L=2))
        tanh = T.tanh(Z.dot(self.W) + self.b)[:, np.newaxis]

        f_Z = Z + U_hat[np.newaxis, :] * tanh

        # Using tanh'(z) = 1 - [tanh(z)]^2.
        psi = (1 - T.square(tanh)).dot(self.W[np.newaxis, :])
        logdet = T.log(abs(1 + psi.dot(U_hat)))
        return f_Z, logdet


def planar_flow(Z_0, num_flows):
    logdet = []
    Z = Z_0
    for k in range(num_flows):
        flow_layer = PlanarFlowLayer(Z)
        Z = IndexLayer(flow_layer, 0)
        logdet.append(IndexLayer(flow_layer, 1))

    return Z, logdet


class GaussianNoiseLayer(MergeLayer):
    def __init__(self, mu, log_covar, **kwargs):
        super().__init__([mu, log_covar], **kwargs)

        # Shameless plug from ``lasagne.layers.GaussianNoiseLayer``.
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        mu, log_covar = input
        if deterministic:
            return mu
        else:
            eps = self._srng.normal(mu.shape)
            return mu + T.exp(log_covar) * eps


class IndexLayer(Layer):
    def __init__(self, incoming, index, **kwargs):
        super().__init__(incoming, **kwargs)

        self.index = index

    def get_output_for(self, input, **kwargs):
        return input[self.index]


class GMMNoiseLayer(MergeLayer):
    def __init__(self, comps, n_components, **kwargs):
        super().__init__(comps, **kwargs)
        self.n_components = n_components

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        mus = T.stacklists(input[:int(self.n_components)])   # (n_components, batch_size, latent)
        log_covars = T.stacklists(input[int(self.n_components):int(2*self.n_components)])
        weights = T.stacklists(input[int(2*self.n_components):])   # (n_components, batch_size, 1)

        weights = T.addbroadcast(weights, 2)
        weights = weights.dimshuffle(0, 1)     # (n_components, batch_size)
        weights = normalize(weights)

        idx = T.argmax(self._srng.multinomial(pvals=weights.T, n=1), axis=1)  # (batch_size, )

        range_array = T.arange(idx.shape[0])
        mu = mus[idx, range_array]
        log_covar = log_covars[idx, range_array]

        if deterministic:
            return mu
        else:
            eps = self._srng.normal(mu.shape)
            return mu + T.exp(log_covar) * eps
