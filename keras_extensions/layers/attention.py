from keras.layers import SimpleRNN, Recurrent, time_distributed_dense, Wrapper, initializations, activations, \
    regularizers
from keras.layers.core import Lambda
from keras.engine.topology import Layer, InputSpec
import theano.tensor as T
import keras.backend as K
import numpy

__author__ = 'sjebbara'


def CombineSequences(**kwargs):
    return Lambda(combine_sequences, combine_sequences_shape, **kwargs)


def combine_sequences(x):
    sequence1 = x[0]
    sequence2 = x[1]

    batch_dim, sequence1_seq_dim, sequence1_element_dim = sequence1.shape
    __, sequence2_seq_dim, sequence2_element_dim = sequence2.shape

    sequence1_tile = T.tile(sequence1, sequence2_seq_dim)
    sequence1_repeat = T.reshape(sequence1_tile,
                                 (batch_dim, sequence1_seq_dim * sequence2_seq_dim, sequence1_element_dim))

    sequence2_flat = T.reshape(sequence2, (batch_dim, 1, sequence2_seq_dim * sequence2_element_dim))
    sequence2_flat_repeat = T.repeat(sequence2_flat, sequence1_seq_dim, axis=1)
    sequence2_repeat = T.reshape(sequence2_flat_repeat,
                                 (batch_dim, sequence1_seq_dim * sequence2_seq_dim, sequence2_element_dim))

    Z_repeat = T.concatenate((sequence1_repeat, sequence2_repeat), axis=2)

    return Z_repeat


def combine_sequences_shape(input_shape):
    batch_dim, sequence1_seq_dim, sequence1_element_dim = input_shape[0]
    batch_dim, sequence2_seq_dim, sequence2_element_dim = input_shape[1]
    if sequence1_seq_dim is None or sequence2_seq_dim is None:
        return (batch_dim, None, sequence1_element_dim + sequence2_element_dim)
    else:
        return (batch_dim, sequence1_seq_dim * sequence2_seq_dim, sequence1_element_dim + sequence2_element_dim)


def CombineVectorAndSequence(**kwargs):
    return Lambda(combine_vector_and_sequence, combine_vector_and_sequence_shape, **kwargs)


def combine_vector_and_sequence(x):
    context = x[0]
    input = x[1]

    __, input_seq_dim, input_dim = input.shape

    context_expand = context.dimshuffle((0, 'x', 1))
    context_repeat = T.repeat(context_expand, input_seq_dim, axis=1)

    Z_repeat = T.concatenate((context_repeat, input), axis=2)
    return Z_repeat


def combine_vector_and_sequence_shape(input_shape):
    batch_dim, context_dim = input_shape[0]
    __, input_seq_dim, input_dim = input_shape[1]
    return (batch_dim, input_seq_dim, context_dim + input_dim)


def SqueezeSequence2Vector(**kwargs):
    return Lambda(function=lambda x: T.squeeze(x), output_shape=lambda s: (s[0], s[1]), **kwargs)


#
# def WeightedSum(**kwargs):
#     def weighted_sum((input_sequence, weights)):
#         W_expand = K.expand_dims(weights, -1)
#
#         M = input_sequence * W_expand
#         avrg = K.sum(M, axis=1)
#         return avrg
#
#     def weighted_sum_shape(input_shapes):
#         batch_dim, X_seq_dim, X_dim = input_shapes[0]
#         _, _ = input_shapes[1]
#         return (batch_dim, X_dim)
#
#     return Lambda(weighted_sum, weighted_sum_shape, **kwargs)

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = False
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(WeightedSum, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, (X, W), mask=None):
        W_expand = K.expand_dims(W, -1)

        M = X * W_expand
        if mask is not None:
            mask = mask[0]  # mask of X
            if mask is not None:
                print mask.ndim
                M = M * K.cast(mask, K.floatx())

        avrg = K.sum(M, axis=1)
        return avrg

    def get_output_shape_for(self, input_shapes):
        batch_dim, X_seq_dim, X_dim = input_shapes[0]
        _, _ = input_shapes[1]
        return (batch_dim, X_dim)


def MultiplyAttentionVector():
    def fn((input_sequence, weights)):
        W_expand = K.expand_dims(weights, -1)

        M = input_sequence * W_expand
        return M

    def fn_shape((input_shape, weight_shape)):
        return input_shape

    return Lambda(fn, fn_shape)


def ReshapeSequences(sequence1_shape=None, sequence2_shape=None, **kwargs):
    return Lambda(reshape_sequence, reshape_sequence_shape,
                  arguments={"sequence1_seq_dim": sequence1_shape[1], "sequence2_seq_dim": sequence2_shape[1]},
                  **kwargs)


def reshape_sequence(sequence, sequence1_seq_dim=None, sequence2_seq_dim=None):
    batch_dim = sequence.shape[0]

    reshaped = T.reshape(sequence, (batch_dim, sequence1_seq_dim, sequence2_seq_dim, 1))
    reshaped = T.squeeze(reshaped)
    return reshaped


def reshape_sequence_shape(shape):
    return (None, None, None)


class ApplyAttentionMatrix(Layer):
    def __init__(self, aggregate="rows", **kwargs):
        self.aggregate = aggregate
        super(ApplyAttentionMatrix, self).__init__(**kwargs)

    def call(self, x, mask=None):
        sequence = x[0]
        attention_matrix = x[1]
        other_seq_dim = attention_matrix.shape[1]
        A_expand = attention_matrix.dimshuffle((0, 1, 2, 'x'))

        sequence_expand = sequence.dimshuffle((0, 'x', 1, 2))
        sequence_repeat = T.repeat(sequence_expand, other_seq_dim, axis=1)
        M = sequence_repeat * A_expand
        avrg = T.sum(M, axis=2)
        return avrg

    def get_output_shape_for(self, input_shape):
        batch_dim, seq_dim, seq_element_dim = input_shape[0]
        batch_dim, other_seq_dim, __ = input_shape[1]
        return (batch_dim, other_seq_dim, seq_element_dim)


class SimpleAttention(Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, attention_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh',
                 W_regularizer=None, w_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.,
                 dropout_U=0., **kwargs):
        self.attention_dim = attention_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.R_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_R = dropout_U

        if self.dropout_W or self.dropout_R:
            self.uses_learning_phase = True
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.add_weight((input_dim, self.attention_dim), initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
        self.w = self.add_weight((self.attention_dim,), initializer=self.init, name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer)
        self.R = self.add_weight((self.attention_dim, self.attention_dim), initializer=self.inner_init,
                                 name='{}_R'.format(self.name), regularizer=self.R_regularizer)
        self.b = self.add_weight((self.attention_dim,), initializer='zero', name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], numpy.zeros((input_shape[0], self.attention_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.attention_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W, input_dim, self.attention_dim, timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_R = states[1]
        B_W = states[2]

        if self.consume_less == 'cpu':
            Wx = x
        else:
            Wx = K.dot(x * B_W, self.W) + self.b

        h_intermediate = self.activation(Wx + K.dot(prev_output * B_R, self.R))
        score = K.dot(h_intermediate, self.w)
        return h_intermediate, [h_intermediate]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_R < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.attention_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_R), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.attention_dim, 'init': self.init.__name__, 'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'R_regularizer': self.R_regularizer.get_config() if self.R_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W, 'dropout_R': self.dropout_R}
        base_config = super(SimpleAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Wrapper):
    """
    This wrapper will provide an attention layer to a recurrent layer.

    # Arguments:
        layer: `Recurrent` instance with consume_less='gpu' or 'mem'

    # Examples:

    ```python
    model = Sequential()
    model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```

    # References
    - [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)


    """

    def __init__(self, layer, **kwargs):
        assert isinstance(layer, Recurrent)
        if layer.get_config()['consume_less'] == 'cpu':
            raise Exception("AttentionLSTMWrapper doesn't support RNN's with consume_less='cpu'")
        self.supports_masking = True
        super(Attention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(Attention, self).build()

        self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W2 = self.layer.init((self.layer.output_dim, input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim * 2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))

        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        # This is based on [tensorflows implementation](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506).
        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * X +b2 + W1 * h))
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).

        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h = states[0]
        X = states[-1]
        xW1 = states[-2]

        Xr = K.reshape(X, (-1, nb_time, 1, input_dim))
        hW2 = K.dot(h, self.W2) + self.b2
        hW2 = K.reshape(hW2, (-1, 1, 1, input_dim))
        u = K.tanh(xW1 + hW2)
        a = K.sum(self.V * u, [2, 3])
        a = K.softmax(a)
        a = K.reshape(a, (-1, nb_time, 1, 1))

        # Weight attention vector by attention
        Xa = K.sum(a * Xr, [1, 2])
        Xa = K.reshape(Xa, (-1, input_dim))

        # Merge input and attention weighted inputs into one vector of the right size.
        x = K.dot(K.concatenate([x, Xa], 1), self.W3) + self.b3

        h, new_states = self.layer.step(x, states)
        return h, new_states

    def get_constants(self, x):
        constants = self.layer.get_constants(x)

        # Calculate K.dot(x, W2) only once per sequence by making it a constant
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        Xr = K.reshape(x, (-1, nb_time, input_dim, 1))
        Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
        xW1t = K.conv2d(Xrt, self.W1, border_mode='same')
        xW1 = K.permute_dimensions(xW1t, (0, 2, 3, 1))
        constants.append(xW1)

        # we need to supply the full sequence of inputs to step (as the attention_vector)
        constants.append(x)

        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name + ': ' + str(input_shape))

        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input, initial_states,
                                             go_backwards=self.layer.go_backwards, mask=mask, constants=constants,
                                             unroll=self.layer.unroll, input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output
