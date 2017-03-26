__author__ = 'sjebbara'
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.models import Model
from keras.layers import Lambda, Layer, GRU, Merge, merge, Wrapper, Dense, TimeDistributed, Activation, Dropout, Input, \
    Embedding, Convolution1D
from keras import activations, initializations, regularizers, constraints
import theano
import theano.tensor as T
import keras_extensions.activations
from keras.utils.np_utils import conv_output_length
import numpy


def RMSELayer(**kwargs):
    return Lambda(lambda (target, context): K.sqrt(K.mean(K.square(target - context), axis=-1, keepdims=True)),
                  output_shape=lambda shapes: (1, 1), **kwargs)


class WeightedScore(Layer):
    def __init__(self, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(WeightedScore, self).__init__(**kwargs)

    def build(self, input_shapes):
        assert len(input_shapes) == 2
        input_dim = input_shapes[0][1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]
        self.W = self.init((input_dim, input_dim), name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((input_dim,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        left = x[0]
        right = x[1]
        output = K.dot(left, self.W)
        output = K.sum(output * right, axis=-1, keepdims=True)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shapes):
        assert input_shapes and len(input_shapes) == 2
        return (input_shapes[0][0], 1)

    def get_config(self):
        config = {'init': self.init.__name__, 'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None, 'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(WeightedScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ExtendedEmbedding(input_dim, additional_output_dim, weights, trainable=True, name=None, **kwargs):
    word_input = Input((None,), dtype="int32", name="word_input")
    output_dim = weights[0].shape[1]
    base_embeddings = Embedding(input_dim, output_dim, weights=weights, trainable=trainable, **kwargs)(word_input)
    if additional_output_dim > 0:
        additional_embeddings = Embedding(input_dim, additional_output_dim, **kwargs)(word_input)

        extended_embeddings = merge([base_embeddings, additional_embeddings], mode="concat")
    else:
        extended_embeddings = base_embeddings
    model = Model(word_input, extended_embeddings, name=name)
    return model


def OneHotEmbedding(n_vocab, input_length=None, **kwargs):
    W = initializations.identity((n_vocab, n_vocab))
    if input_length:
        return Lambda(lambda X: W[X], output_shape=lambda shape: (shape[0], input_length, n_vocab), **kwargs)
    else:
        return Lambda(lambda X: W[X], output_shape=lambda shape: (shape[0], shape[1], n_vocab), **kwargs)


def SumVectorElements(keepdims=False, **kwargs):
    def fn_shape(shape):
        if keepdims:
            return shape[:-1] + (1,)
        else:
            return shape[:-1]

    return Lambda(lambda v: K.sum(v, axis=-1, keepdims=keepdims), fn_shape, **kwargs)


def LogSumExp(axis=1, keepdims=True, **kwargs):
    def fn_shape(shape):
        if keepdims:
            return shape[:axis] + (1,) + shape[axis + 1:]
        else:
            return shape[:axis] + shape[axis + 1:]

    def fn(scores):
        score = K.log(K.sum(K.exp(scores), axis=axis, keepdims=keepdims))
        return score

    return Lambda(fn, fn_shape, **kwargs)


def ElementAt(index=0, **kwargs):
    return Lambda(lambda seq: seq[:, index], output_shape=lambda shape: shape[:1] + shape[2:], **kwargs)


def SqueezeSequence2Vector(**kwargs):
    return Lambda(function=lambda x: K.squeeze(x, axis=2), output_shape=lambda s: (s[0], s[1]), **kwargs)


def Squeeze(axis=2, **kwargs):
    return Lambda(function=lambda x: K.squeeze(x, axis=axis),
                  output_shape=lambda shape: shape[:axis] + shape[axis + 1:], **kwargs)


def MaskedMean(**kwargs):
    def fn((X, M)):
        M = K.expand_dims(M, -1)
        Xm = X * M
        Mm = K.sum(Xm, axis=1)
        return Mm / K.sum(M, axis=1)

    def fn_shape(shapes):
        X_shape, M_shape = shapes
        return [X_shape[0], X_shape[2]]

    return Lambda(fn, fn_shape, **kwargs)


def Expand(axis=-1, **kwargs):
    def fn(X):
        return K.expand_dims(X, axis)

    def fn_shape(shape):
        if axis == -1:
            return shape + (1,)
        else:
            return shape[:axis] + (1,) + shape[axis:]

    return Lambda(fn, fn_shape, **kwargs)


def ExpandLast(**kwargs):
    def fn(X):
        return K.expand_dims(X, dim=-1)

    def fn_shape(shape):
        return shape + (1,)

    return Lambda(fn, fn_shape, **kwargs)


def SliceLast(dim, **kwargs):
    def fn(X):
        X1 = X[:, :, dim]
        return X1

    def fn_shape(shape):
        return shape[:-1]

    return Lambda(fn, fn_shape, **kwargs)


def RemoveSlice(axis=1, index=-1, **kwargs):
    def fn(X):
        if index >= 0:
            _index = index
        elif index < 0:
            _index = X.shape[axis] - index

        if axis == 1:
            return K.concatenate((X[:, :_index], X[:, _index + 1:]), axis=axis)
        elif axis == 2:
            return K.concatenate((X[:, :, _index], X[:, :, _index + 1:]), axis=axis)

    def fn_shape(shape):
        sliced_shape = shape[:axis] + (shape[axis] - 1,) + shape[axis + 1:]
        return sliced_shape

    return Lambda(fn, fn_shape, **kwargs)


def ConcatBeyondLast(**kwargs):
    def fn((Xs)):
        Xs = [K.expand_dims(X, dim=-1) for X in Xs]
        X = K.concatenate(Xs, axis=-1)
        return X

    def fn_shape(shapes):
        shape = shapes[0]
        return shape + (len(shapes),)

    return Lambda(fn, fn_shape, **kwargs)


def BatchTimeDot(left_input_ndim, right_input_ndim, output_ndim, dot_axes=-1, **kwargs):
    n1 = left_input_ndim + 1
    n2 = right_input_ndim + 1
    if isinstance(dot_axes, int):
        if dot_axes < 0:
            dot_axes = [dot_axes % n1, dot_axes % n2]
        else:
            dot_axes = [dot_axes, ] * 2

    def call(inputs):
        left, right = inputs
        input_shape_left = K.shape(left)
        input_shape_right = K.shape(right)

        batch_size = input_shape_left[0]
        input_length = input_shape_left[1]

        # (nb_samples * timesteps, ...)
        reshaped_left_input_shape = (input_shape_left[0] * input_shape_left[1],) + tuple(input_shape_left[2:])
        reshaped_right_input_shape = (input_shape_left[0] * input_shape_left[1],) + tuple(input_shape_right[2:])
        print input_shape_left, "->", reshaped_left_input_shape
        print input_shape_right, "->", reshaped_right_input_shape
        left = K.reshape(left, reshaped_left_input_shape, ndim=left_input_ndim)
        right = K.reshape(right, reshaped_right_input_shape, ndim=right_input_ndim)

        shifted_dot_axes = tuple(d - 1 for d in dot_axes)  # shift to the left since we remove batch AND dot
        print "shifted_dot_axes", shifted_dot_axes
        output = merge([left, right], mode="dot", dot_axes=shifted_dot_axes)
        # output = K.batch_dot(left, right, axes=dot_axes)

        actual_output_shape = K.shape(output)
        # (nb_samples, timesteps, ...)
        reshaped_output_shape = (batch_size, input_length) + tuple(actual_output_shape[1:])
        print actual_output_shape, "->", reshaped_output_shape
        output = K.reshape(output, reshaped_output_shape, ndim=output_ndim + 1)

        return output

    def get_output_shape_for((left, right)):
        shape1 = list(left)
        shape2 = list(right)
        shape1.pop(dot_axes[0])
        shape2.pop(dot_axes[1])
        shape2.pop(0)  # remove batch dim
        shape2.pop(0)  # remove time dim
        print "left part", shape1
        print "right part", shape2
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        print output_shape
        return tuple(output_shape)

    return Lambda(call, get_output_shape_for, **kwargs)


def DynamicFlatten(**kwargs):
    def dynamic_flatten(X):
        return K.batch_flatten(X)

    def dynamic_flatten_shape(shape):
        d = shape[1]
        for d2 in shape[2:]:
            d = d * d2
        return (shape[0], d)

    return Lambda(dynamic_flatten, dynamic_flatten_shape, **kwargs)


def DynamicRepeat(other_tensor, axis=1, **kwargs):
    n_repeat = other_tensor.shape[axis]

    def dynamic_repeat(X):
        return K.repeat_elements(X, rep=n_repeat, axis=axis)

    def dynamic_repeat_shape(shape):
        return shape[:axis] + (None,) + shape[axis:]

    return Lambda(dynamic_repeat, dynamic_repeat_shape, **kwargs)


def RepeatToMatch(axis=1, **kwargs):
    def dynamic_repeat((X, other_tensor)):
        n_repeat = K.shape(other_tensor)[axis]
        # return K.repeat_elements(X, rep=n_repeat, axis=axis)
        return K.repeat(X, n_repeat)

    def dynamic_repeat_shape((shape, other_shape)):
        output_shape = shape[:axis] + (other_shape[axis],) + shape[axis:]
        return output_shape

    return Lambda(dynamic_repeat, dynamic_repeat_shape, **kwargs)


def DynamicReshape3D(n1_dynamic, n2_dynamic, n3_explicit, **kwargs):
    def dynamic_reshape(X):
        new_shape = (X.shape[0], n1_dynamic, n2_dynamic, n3_explicit)
        print "DS new reshape:", new_shape
        return K.reshape(X, new_shape)

    # def dynamic_shape(shape):
    # 	print "DS", shape
    new_shape = (None, None, n3_explicit)
    print "DS new", new_shape
    # return new_shape

    return Lambda(dynamic_reshape, new_shape, **kwargs)


def ReshapeLast(last_shape, **kwargs):
    def fn(X):
        X_shape = K.shape(X)
        X = K.reshape(X, tuple(X_shape[:-1]) + tuple(last_shape), ndim=X.ndim - 1 + len(last_shape))
        return X

    def fn_shape(X_shape):
        new_shape = X_shape[:-1] + tuple(last_shape)
        return new_shape

    return Lambda(fn, fn_shape, **kwargs)


def MultiConvolution1D(input_shape, conv_sizes, **kwargs):
    sequence_input = Input(shape=input_shape)
    outputs = []
    for nb_filter, filter_length in conv_sizes:
        output = Convolution1D(nb_filter, filter_length, border_mode="same", **kwargs)(sequence_input)
        outputs.append(output)

    if len(outputs) > 1:
        merged_outputs = merge(outputs, mode="concat")
    else:
        merged_outputs = outputs[1]

    model = Model(sequence_input, merged_outputs, **kwargs)
    return model


def TimeDistributedMerge(mode="sum", **kwargs):
    if mode == "sum":
        return Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:], **kwargs)
    elif mode == "mean" or mode == "avrg":
        return Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:],
                      **kwargs)
    elif mode == "max":
        return Lambda(function=lambda x: K.max(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:], **kwargs)
    elif mode == "min":
        return Lambda(function=lambda x: K.min(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:], **kwargs)
    elif mode == "mul":
        return Lambda(function=lambda x: K.prod(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:],
                      **kwargs)
    elif mode == "lse" or mode == "LogSumExp":
        def fn(x):
            x = K.exp(x)
            x = K.sum(x, axis=1, keepdims=False)
            x = K.log(x)
            return x

        return Lambda(function=fn, output_shape=lambda shape: (shape[0],) + shape[2:], **kwargs)
    return None


class RBF(Layer):
    def __init__(self, n_centroids, init='glorot_uniform', beta_init='one',
                 kernel=keras_extensions.activations.gaussian_kernel, norm=keras_extensions.activations.euclidean_norm,
                 weights=None, W_regularizer=None, beta_regularizer=None, activity_regularizer=None, W_constraint=None,
                 beta_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.beta_init = initializations.get(beta_init)
        self.kernel = activations.get(kernel)
        self.n_centroids = n_centroids
        self.input_dim = input_dim
        self.norm = norm

        self.W_regularizer = regularizers.get(W_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.beta_constraint = constraints.get(beta_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(RBF, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]

        self.W = self.init((input_dim, self.n_centroids), name='{}_W'.format(self.name))
        self.beta = self.beta_init((self.n_centroids,), name='{}_beta'.format(self.name))

        self.trainable_weights = [self.W, self.beta]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.beta_regularizer:
            self.beta_regularizer.set_param(self.beta)
            self.regularizers.append(self.beta_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.beta_constraint:
            self.constraints[self.beta] = self.beta_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        X = K.expand_dims(x, dim=-1)
        C = K.expand_dims(self.W, dim=0)
        Z = X - C
        R = self.norm(Z, axis=1, keepdims=False)
        return self.kernel(R, self.beta)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.n_centroids)

    def get_config(self):
        config = {'n_centroids': self.n_centroids, 'init': self.init.__name__, 'beta_init': self.beta_init.__name__,
                  'kernel': self.kernel.__name__, 'norm': self.norm.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'beta_constraint': self.beta_constraint.get_config() if self.beta_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(RBF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def TimeDistributedBilinearProduct(left_size, right_size, activation="tanh", init="glorot_uniform", dropout=None,
                                   **kwargs):
    left = Input(shape=(None, left_size))
    right = Input(shape=(None, right_size))
    tmp = TimeDistributed(Dense(right_size, activation="linear", init=init))(left)
    tmp = Dropout(dropout)(tmp)
    tmp = merge([tmp, right], mode="mul")
    product = SumVectorElements()(tmp)
    product = Activation(activation)(product)
    model = Model([left, right], product, **kwargs)
    return model


def TimeDistributedBilinearTensorProduct(left_size, right_size, output_size, activation="linear", init="glorot_uniform",
                                         dropout=None, **kwargs):
    left = Input(shape=(None, left_size))
    right = Input(shape=(None, right_size))

    tensor = LearnedTensor(output_tensor_shape=(left_size, right_size, output_size))
    outputs = []
    for i in range(output_size):
        tmp = TimeDistributed(Dense(right_size, activation="linear", init=init))(left)
        tmp = Dropout(dropout)(tmp)
        tmp = merge([tmp, right], mode="mul")
        product = SumVectorElements()(tmp)
        product = Activation(activation)(product)
        outputs.append(product)
    model = Model([left, right], outputs, **kwargs)
    return model


class BilinearProduct2(Layer):
    def __init__(self, left_dim=None, right_dim=None, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None,
                 b_constraint=None, bias=True, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.left_dim = left_dim
        self.right_dim = right_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        super(BilinearProduct2, self).__init__(**kwargs)

    def build(self, input_shapes):
        assert len(input_shapes) == 2
        self.left_dim = input_shapes[1][1]
        self.right_dim = input_shapes[1][1]

        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, self.left_dim)),
                           InputSpec(dtype=K.floatx(), shape=(None, self.right_dim))]

        self.W = self.init((self.left_dim, self.right_dim), name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((1,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, (left, right), mask=None):
        output = K.dot(left, self.W)
        output = K.sum(output * right, axis=1, keepdims=True)

        # output = K.batch_dot(K.batch_dot(left, self.W), right)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
        config = {'init': self.init.__name__, 'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None, 'bias': self.bias,
                  'left_dim': self.left_dim, 'right_dim': self.right_dim}
        base_config = super(BilinearProduct2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level)  # compensate for the scaling by the dropout
    return x


class QRNN(Layer):
    '''Qausi RNN

    from: https://github.com/DingKe/qrnn

    # Arguments
        output_dim: dimension of the internal projections and the final output.

    # References
        - [Qausi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    '''

    def __init__(self, output_dim, window_size=2, return_sequences=False, go_backwards=False, stateful=False,
                 unroll=False, subsample_length=1, init='uniform', activation='tanh', W_regularizer=None,
                 b_regularizer=None, W_constraint=None, b_constraint=None, dropout=0, weights=None, bias=True,
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.output_dim = output_dim
        self.window_size = window_size
        self.subsample = (subsample_length, 1)

        self.bias = bias
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.dropout = dropout
        if self.dropout is not None and 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        input_dim = input_shape[2]
        self.input_spec = [InputSpec(shape=input_shape)]
        self.W_shape = (self.window_size, 1, input_dim, self.output_dim)

        self.W_z = self.init(self.W_shape, name='{}_W_z'.format(self.name))
        self.W_f = self.init(self.W_shape, name='{}_W_f'.format(self.name))
        self.W_o = self.init(self.W_shape, name='{}_W_o'.format(self.name))
        self.trainable_weights = [self.W_z, self.W_f, self.W_o]
        self.W = K.concatenate([self.W_z, self.W_f, self.W_o], 1)

        if self.bias:
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))
            self.b_f = K.zeros((self.output_dim,), name='{}_b_f'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))
            self.trainable_weights += [self.b_z, self.b_f, self.b_o]
            self.b = K.concatenate([self.b_z, self.b_f, self.b_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1, self.window_size, 'valid', self.subsample[0])
        if self.return_sequences:
            return (input_shape[0], length, self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception(
                'If a RNN is stateful, a complete ' + 'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], numpy.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input, initial_states,
                                             go_backwards=self.go_backwards, mask=mask, constants=constants)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, x):
        if self.bias:
            weights = zip(self.trainable_weights[0:3], self.trainable_weights[3:])
        else:
            weights = self.trainable_weights

        if self.window_size > 1:
            x = K.asymmetric_temporal_padding(x, self.window_size - 1, 0)
        x = K.expand_dims(x, 2)  # add a dummy dimension

        # z, f, o
        outputs = []
        for param in weights:
            if self.bias:
                W, b = param
            else:
                W = param
            output = K.conv2d(x, W, strides=self.subsample, border_mode='valid', dim_ordering='tf')
            output = K.squeeze(output, 2)  # remove the dummy dimension
            if self.bias:
                output += K.reshape(b, (1, 1, self.output_dim))

            outputs.append(output)

        if self.dropout is not None and 0. < self.dropout < 1.:
            f = K.sigmoid(outputs[1])
            outputs[1] = K.in_train_phase(1 - _dropout(1 - f, self.dropout), f)

        return K.concatenate(outputs, 2)

    def step(self, input, states):
        prev_output = states[0]

        z = input[:, :self.output_dim]
        f = input[:, self.output_dim:2 * self.output_dim]
        o = input[:, 2 * self.output_dim:]

        z = self.activation(z)
        f = f if self.dropout is not None and 0. < self.dropout < 1. else K.sigmoid(f)
        o = K.sigmoid(o)

        output = f * prev_output + (1 - f) * z
        output = o * output

        return output, [output]

    def get_constants(self, x):
        constants = []
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim, 'init': self.init.__name__, 'window_size': self.window_size,
                  'subsample_length': self.subsample[0], 'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None, 'bias': self.bias,
                  'input_dim': self.input_dim, 'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LearnedTensor(Layer):
    def __init__(self, output_tensor_shape, init='glorot_uniform', W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, mask_zero=False, weights=None, **kwargs):
        self.output_tensor_shape = output_tensor_shape
        self.init = initializations.get(init)
        self.mask_zero = mask_zero

        self.W_constraint = constraints.get(W_constraint)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        super(LearnedTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(self.output_tensor_shape, initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer, constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + self.output_tensor_shape

    def call(self, x, mask=None):
        input_shape = K.shape(x)
        W = self.W
        W = K.expand_dims(W, dim=0)
        W = K.repeat_elements(W, input_shape[0], 0)
        return W

    def get_config(self):
        config = {'output_tensor_shape': self.output_tensor_shape, 'init': self.init.__name__,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(LearnedTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TDBilinearLowRankTensorProduct(Layer):
    def __init__(self, output_dim, rank, init='glorot_uniform', V_regularizer=None, activity_regularizer=None,
                 activation="linear", V_constraint=None, mask_zero=False, weights=None, bias=True, **kwargs):
        self.output_dim = output_dim
        self.rank = rank
        self.init = initializations.get(init)
        self.mask_zero = mask_zero
        self.activation = activations.get(activation)
        self.V_constraint = constraints.get(V_constraint)

        self.V_regularizer = regularizers.get(V_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

        super(TDBilinearLowRankTensorProduct, self).__init__(**kwargs)

    def build(self, input_shapes):

        batch_size, sequence_length, left_input_dim = input_shapes[0]
        batch_size, sequence_length, right_input_dim = input_shapes[1]

        self.Q1 = self.init((self.output_dim, left_input_dim, self.rank),
                            name='{}_Q1'.format(self.name))  # output_dim, left_input_dim, rank      p,m,q
        self.Q2 = self.init((self.output_dim, right_input_dim, self.rank),
                            name='{}_Q2'.format(self.name))  # output_dim, right_input_dim, rank     p,m,q
        self.trainable_weights += [self.Q1, self.Q2]

        self.V = K.batch_dot(self.Q1, self.Q2, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m

        if self.bias:
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights += [self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], self.output_dim)

    def call(self, inputs, mask=None):
        left, right = inputs
        input_shape_left = K.shape(left)
        input_shape_right = K.shape(right)

        batch_size, input_length, left_input_dim = input_shape_left
        batch_size, input_length, right_input_dim = input_shape_right

        # (nb_samples * timesteps, ...)
        left = K.reshape(left, (batch_size * input_length, left_input_dim), ndim=2)
        right = K.reshape(right, (batch_size * input_length, right_input_dim), ndim=2)

        tmp1 = K.dot(left, self.V)  # n,m + p,m,m = n,p,m
        output = K.batch_dot(right, tmp1, axes=[[1], [2]])  # n,m + n,p,m = n,p
        if self.bias:
            output += self.b

        # (nb_samples, timesteps, ...)
        output = K.reshape(output, (batch_size, input_length, self.output_dim), ndim=3)

        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim, 'init': self.init.__name__, 'rank': self.rank,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'V_constraint': self.V_constraint.get_config() if self.V_constraint else None}
        base_config = super(TDBilinearLowRankTensorProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseTensor(Layer):
    '''
    From: https://github.com/bstriner/dense_tensor
    '''

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None,
                 V_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None,
                 b_constraint=None, bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseTensor, self).__init__(**kwargs)

    def build_V(self, input_dim):
        self.V = self.init((input_dim, self.output_dim, input_dim), name='{}_V'.format(self.name))
        return [self.V]

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]

        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))

        self.trainable_weights = [self.W]
        self.trainable_weights += self.build_V(input_dim=input_dim)
        if self.bias:
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights += [self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        tmp1 = K.dot(x, self.V)  # n,m + p,m,m = n,p,m
        tmp2 = K.batch_dot(x, tmp1, axes=[[1], [2]])  # n,m + n,p,m = n,p
        output += tmp2
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim, 'init': self.init.__name__, 'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None, 'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(DenseTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseTensorLowRank(DenseTensor):
    '''
    From: https://github.com/bstriner/dense_tensor
    '''

    def __init__(self, q=10, **kwargs):
        self.q = q
        super(DenseTensorLowRank, self).__init__(**kwargs)

    def build_V(self, input_dim):
        self.Q1 = self.init((self.output_dim, input_dim, self.q), name='{}_Q1'.format(self.name))  # p,m,q
        self.Q2 = self.init((self.output_dim, input_dim, self.q), name='{}_Q2'.format(self.name))  # p,m,q
        self.V = K.batch_dot(self.Q1, self.Q2, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m
        return self.Q1, self.Q2

    def get_config(self):
        config = {'q': self.q}
        base_config = super(DenseTensorLowRank, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
