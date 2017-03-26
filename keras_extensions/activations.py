from __future__ import absolute_import

from keras import activations
from keras import backend as K
from keras.layers import Layer


def euclidean_norm(x, axis=-1, keepdims=False):
    return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))


def gaussian_kernel(R, beta=1):
    output = K.exp(-beta * K.square(R))
    return output


def inverse_quadratic_kernel(R, beta=1):
    output = 1 / (1 + beta * K.square(R))
    return output


def inverse_multiquadratic_kernel(R, beta=1):
    output = 1 / K.sqrt(1 + beta * K.square(R))
    return output


def normalize(x):
    ndim = K.ndim(x)
    if ndim <= 3:
        # x = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(x, axis=-1, keepdims=True)
        return x / s
    else:
        raise Exception('Cannot apply row normalization to a tensor that is not 2D or 3D. ' + 'Here, ndim=' + str(ndim))


class ScaledTanh(Layer):
    def __init__(self, alpha=1, **kwargs):
        self.supports_masking = True
        self.alpha = alpha
        super(ScaledTanh, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.tanh(self.alpha * x)

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(ScaledTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Softmax(Layer):
    """Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Softmax, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ndim = K.ndim(x)
        if ndim <= 3:
            e = K.exp(x - K.max(x, axis=-1, keepdims=True))
            if mask is not None:
                print mask.ndim
                mask = K.expand_dims(mask, dim=-1)
                e = e * K.cast(mask, K.floatx())
            s = K.sum(e, axis=-1, keepdims=True)
            # return K.switch(s == 0, K.zeros_like(e), e / s)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor '
                             'that is not 2D or 3D. '
                             'Here, ndim=' + str(ndim))

    def get_config(self):
        config = {}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# def get(identifier):
# 	return get_from_module(identifier, globals(), 'activation function')
