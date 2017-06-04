import numpy

from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Recurrent, time_distributed_dense, ELU
from keras import activations, initializations, regularizers, constraints
from keras_extensions.layers import crf_theano


class TaggingGRU(Recurrent):
    """Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, pre_output_dim, output_dim, mode="concat", init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', tag_activation="softmax", inner_activation='hard_sigmoid', W_regularizer=None,
                 U_regularizer=None, b_regularizer=None, W_t_regularizer=None, b_t_regularizer=None, dropout_W=0.,
                 dropout_U=0., dropout_W_t=0., **kwargs):
        self.pre_output_dim = pre_output_dim
        self.output_dim = output_dim
        self.mode = mode
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.tag_activation = activations.get(tag_activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_t_regularizer = regularizers.get(W_t_regularizer)
        self.b_t_regularizer = regularizers.get(b_t_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.dropout_W_t = dropout_W_t

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(TaggingGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None, None]

        self.W_z = self.add_weight((self.input_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_W_z'.format(self.name), regularizer=self.W_regularizer)
        self.U_z = self.add_weight((self.pre_output_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_U_z'.format(self.name), regularizer=self.W_regularizer)
        self.b_z = self.add_weight((self.pre_output_dim,), initializer='zero', name='{}_b_z'.format(self.name),
                                   regularizer=self.b_regularizer)
        self.W_r = self.add_weight((self.input_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_W_r'.format(self.name), regularizer=self.W_regularizer)
        self.U_r = self.add_weight((self.pre_output_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_U_r'.format(self.name), regularizer=self.W_regularizer)
        self.b_r = self.add_weight((self.pre_output_dim,), initializer='zero', name='{}_b_r'.format(self.name),
                                   regularizer=self.b_regularizer)
        self.W_h = self.add_weight((self.input_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_W_h'.format(self.name), regularizer=self.W_regularizer)
        self.U_h = self.add_weight((self.pre_output_dim, self.pre_output_dim), initializer=self.init,
                                   name='{}_U_h'.format(self.name), regularizer=self.W_regularizer)
        self.b_h = self.add_weight((self.pre_output_dim,), initializer='zero', name='{}_b_h'.format(self.name),
                                   regularizer=self.b_regularizer)
        self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
        self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
        self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        if self.mode == "concat":
            self.W_t = self.add_weight((self.pre_output_dim + self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_t'.format(self.name), regularizer=self.W_t_regularizer)
            self.b_t = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_t'.format(self.name),
                                       regularizer=self.b_regularizer)
        if self.mode == "tensor":
            self.W_t = self.add_weight((self.pre_output_dim, self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_t'.format(self.name), regularizer=self.W_t_regularizer)
            self.b_t = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_t'.format(self.name),
                                       regularizer=self.b_regularizer)
        if self.mode == "transition":
            self.A_t = self.add_weight((self.output_dim, self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_A_t'.format(self.name), regularizer=self.W_t_regularizer)
            self.W_t = self.add_weight((self.pre_output_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_t'.format(self.name), regularizer=self.W_t_regularizer)
            self.b_at = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_at'.format(self.name),
                                        regularizer=self.b_regularizer)
            self.b_wt = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_wt'.format(self.name),
                                        regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, pre_output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_h_state = K.tile(initial_state, [1, self.pre_output_dim])  # (samples, pre_output_dim)
        initial_t_state = K.tile(initial_state, [1, self.output_dim])  # (samples, tag_size)
        initial_states = [initial_h_state, initial_t_state]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], numpy.zeros((input_shape[0], self.pre_output_dim)))
            K.set_value(self.states[1], numpy.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.pre_output_dim)), K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        input_shape = K.int_shape(x)
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W, input_dim, self.pre_output_dim, timesteps)
        x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W, input_dim, self.pre_output_dim, timesteps)
        x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W, input_dim, self.pre_output_dim, timesteps)
        return K.concatenate([x_z, x_r, x_h], axis=2)

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        t_tm1 = states[1]  # previous tag
        B_U = states[2]  # dropout matrices for recurrent units
        B_W = states[3]

        x_z = x[:, :self.pre_output_dim]
        x_r = x[:, self.pre_output_dim: 2 * self.pre_output_dim]
        x_h = x[:, 2 * self.pre_output_dim:]
        z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh

        #### Project to tags ####
        if self.mode == "concat":
            ht = K.concatenate([h, t_tm1])
            t = self.tag_activation(K.dot(ht, self.W_t) + self.b_t)
        elif self.mode == "tensor":
            tmp1 = K.dot(h, self.W_t)  # n,m + p,m,m = n,p,m
            t = K.batch_dot(t_tm1, tmp1, axes=[[1], [2]]) + self.b_t  # n,m + n,p,m = n,p
        elif self.mode == "transition":
            t = K.dot(h, self.W_t) + self.b_wt  # n,m + p,m,m = n,p,m
            tmp1 = K.dot(t, self.A_t)  # n,m + p,m,m = n,p,m
            t = K.batch_dot(t_tm1, tmp1, axes=[[1], [2]]) + self.b_at  # n,m + n,p,m = n,p

        return t, [h, t]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.pre_output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'pre_output_dim': self.pre_output_dim, "output_dim": self.output_dim, "mode": self.mode,
                  'init': self.init.__name__, 'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__, 'inner_activation': self.inner_activation.__name__,
                  "tag_activation": self.tag_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_t_regularizer': self.W_t_regularizer.get_config() if self.W_t_regularizer else None,
                  'b_t_regularizer': self.b_t_regularizer.get_config() if self.b_t_regularizer else None,
                  'dropout_W': self.dropout_W, 'dropout_U': self.dropout_U, 'dropout_W_t': self.dropout_W_t}
        base_config = super(TaggingGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CRFLayer(Layer):
    def __init__(self, init='glorot_uniform', T_regularizer=None, activity_regularizer=None, T_constraint=None,
                 mask_zero=False, weights=None, **kwargs):
        self.init = initializations.get(init)
        self.mask_zero = mask_zero

        self.T_constraint = constraints.get(T_constraint)

        self.T_regularizer = regularizers.get(T_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        super(CRFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        n_tags = input_shape[2]
        self.T = self.add_weight((n_tags, n_tags, n_tags), initializer=self.init, name='{}_T'.format(self.name),
                                 regularizer=self.T_regularizer, constraint=self.T_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        def _step(x, _):
            x = crf_theano.forward_dynamic(x, self.T, log_space=False, return_alpha=True)
            return x, []

        _, X, _ = K.rnn(_step, X, initial_states=[], unroll=False)

        return X

    def get_config(self):
        config = {'init': self.init.__name__, 'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'T_regularizer': self.T_regularizer.get_config() if self.T_regularizer else None,
                  'T_constraint': self.T_constraint.get_config() if self.T_constraint else None}
        base_config = super(CRFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ELUGRU(Recurrent):
    """Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.inner_init = initializations.get(inner_init)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        self.init = "he_normal"
        self.activation = lambda x: K.elu(x, 1.0)

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(ELUGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':
            self.W = self.add_weight((self.input_dim, 3 * self.output_dim), initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer)
            self.U = self.add_weight((self.output_dim, 3 * self.output_dim), initializer=self.inner_init,
                                     name='{}_U'.format(self.name), regularizer=self.U_regularizer)
            self.b = self.add_weight((self.output_dim * 3,), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)
        else:
            self.W_z = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_z'.format(self.name), regularizer=self.W_regularizer)
            self.U_z = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_z'.format(self.name), regularizer=self.W_regularizer)
            self.b_z = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_z'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_r = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_r'.format(self.name), regularizer=self.W_regularizer)
            self.U_r = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_r'.format(self.name), regularizer=self.W_regularizer)
            self.b_r = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_r'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_h = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_h'.format(self.name), regularizer=self.W_regularizer)
            self.U_h = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_h'.format(self.name), regularizer=self.W_regularizer)
            self.b_h = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_h'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W, input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W, input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W, input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim, 'inner_init': self.inner_init.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W, 'dropout_U': self.dropout_U}
        base_config = super(ELUGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
