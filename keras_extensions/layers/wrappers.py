from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.models import Model
from keras.layers import Lambda, Layer, GRU, Merge, merge, Wrapper, Dense, TimeDistributed, Activation, Dropout, Input, \
    Embedding
from keras_extensions import objectives


def ranking_model_wrapper(scoring_model, pos_inputs, neg_inputs, optimizer=None, **kwargs):
    pos_score = scoring_model(pos_inputs)
    neg_score = scoring_model(neg_inputs)

    diff_score = Lambda(lambda (pos, neg): pos - neg, output_shape=lambda (pos, neg): pos, name="diff_score")(
        [pos_score, neg_score])
    ranking_model = Model(input=pos_inputs + neg_inputs, output=diff_score)
    if (optimizer):
        ranking_model.compile(optimizer, objectives.margin_ranking_loss)
    return ranking_model


class BetterTimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every temporal slice of an input.

    The input should be at least 3D, and the dimension of index one
    will be considered to be the temporal dimension.

    Consider a batch of 32 samples,
    where each sample is a sequence of 10 vectors of 16 dimensions.
    The batch input shape of the layer is then `(32, 10, 16)`,
    and the `input_shape`, not including the samples dimension, is `(10, 16)`.

    You can then use `TimeDistributed` to apply a `Dense` layer
    to each of the 10 timesteps, independently:

    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)

        # subsequent layers: no need for input_shape
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 8)`.

    `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Convolution2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(64, 3, 3),
                                  input_shape=(10, 3, 299, 299)))
    ```

    # Arguments
        layer: a layer instance.
    """

    def __init__(self, layer, input_ndim=2, output_ndim=2, n_distribution_axes=1, **kwargs):
        self.supports_masking = True
        self.n_distribution_axes = n_distribution_axes
        super(BetterTimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if not self.layer.built:
            child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
            # print "build: child_input_shape", child_input_shape
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(BetterTimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
        # print "get_shape: child_input_shape", child_input_shape
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        # print "get_shape: child_output_shape", child_output_shape
        distribution_axes = tuple(input_shape[1:self.n_distribution_axes + 1])
        # print "get_shape: distribution_axes", distribution_axes
        output_shape = (child_output_shape[0],) + distribution_axes + child_output_shape[1:]
        # print "get_shape: output_shape", output_shape
        return output_shape

    def call(self, inputs, mask=None):
        # no batch size specified, therefore the layer will be able
        # to process batches of any size
        # we can go with reshape-based implementation for performance
        actual_input_shape = K.shape(inputs)

        # reshaped_input_shape = (actual_input_shape[0] * actual_input_shape[1], actual_input_shape[2])
        reshaped_axis_size = actual_input_shape[0]
        for i in range(self.n_distribution_axes):
            reshaped_axis_size *= actual_input_shape[i + 1]
        reshaped_input_shape = (reshaped_axis_size,) + tuple(actual_input_shape[self.n_distribution_axes + 1:])

        # reshaped_input_shape = (actual_input_shape[0] * actual_input_shape[1],) + tuple(actual_input_shape[2:])
        # print "reshaped_input_shape", len(reshaped_input_shape), reshaped_input_shape
        # print "reshape input ..."
        inputs = K.reshape(inputs, reshaped_input_shape, ndim=len(reshaped_input_shape))
        # print "call inner model ..."
        y = self.layer.call(inputs)  # (nb_samples * timesteps, ...)
        actual_output_shape = K.shape(y)

        reshaped_output_shape = tuple(actual_input_shape[:self.n_distribution_axes + 1]) + tuple(
            actual_output_shape[1:])

        # reshaped_output_shape = (actual_input_shape[0], actual_input_shape[1]) + tuple(actual_output_shape[1:])
        # print "reshaped_output_shape", len(reshaped_output_shape), reshaped_output_shape

        # print "reshape output ..."
        y = K.reshape(y, reshaped_output_shape, ndim=len(reshaped_output_shape))
        # print "done!"

        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        # print "all done!"
        return y


# class BetterTimeDistributed(Wrapper):
#     """This wrapper allows to apply a layer to every temporal slice of an input.
#
#     The input should be at least 3D, and the dimension of index one
#     will be considered to be the temporal dimension.
#
#     Consider a batch of 32 samples,
#     where each sample is a sequence of 10 vectors of 16 dimensions.
#     The batch input shape of the layer is then `(32, 10, 16)`,
#     and the `input_shape`, not including the samples dimension, is `(10, 16)`.
#
#     You can then use `TimeDistributed` to apply a `Dense` layer
#     to each of the 10 timesteps, independently:
#
#     ```python
#         # as the first layer in a model
#         model = Sequential()
#         model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
#         # now model.output_shape == (None, 10, 8)
#
#         # subsequent layers: no need for input_shape
#         model.add(TimeDistributed(Dense(32)))
#         # now model.output_shape == (None, 10, 32)
#     ```
#
#     The output will then have shape `(32, 10, 8)`.
#
#     `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
#     for instance with a `Convolution2D` layer:
#
#     ```python
#         model = Sequential()
#         model.add(TimeDistributed(Convolution2D(64, 3, 3),
#                                   input_shape=(10, 3, 299, 299)))
#     ```
#
#     # Arguments
#         layer: a layer instance.
#     """
#
#     def __init__(self, layer, input_ndim=2, output_ndim=2, n_distribution_axes=1, **kwargs):
#         self.supports_masking = True
#         self.n_distribution_axes = n_distribution_axes
#         super(BetterTimeDistributed, self).__init__(layer, **kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 3
#         self.input_spec = [InputSpec(shape=input_shape)]
#         if not self.layer.built:
#             child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
#             # print "build: child_input_shape", child_input_shape
#             self.layer.build(child_input_shape)
#             self.layer.built = True
#         super(BetterTimeDistributed, self).build()
#
#     def get_output_shape_for(self, input_shape):
#         child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
#         # print "get_shape: child_input_shape", child_input_shape
#         child_output_shape = self.layer.get_output_shape_for(child_input_shape)
#         # print "get_shape: child_output_shape", child_output_shape
#         distribution_axes = tuple(input_shape[1:self.n_distribution_axes+1])
#         # print "get_shape: distribution_axes", distribution_axes
#         output_shape = (child_output_shape[0],) + distribution_axes + child_output_shape[1:]
#         # print "get_shape: output_shape", output_shape
#         return output_shape
#
#     def call(self, inputs, mask=None):
#         input_shape = K.int_shape(inputs)
#         if input_shape[0]:
#             # batch size matters, use rnn-based implementation
#             def step(x, _):
#                 output = self.layer.call(x)
#                 return output, []
#
#             _, outputs, _ = K.rnn(step, inputs, initial_states=[], input_length=input_shape[1], unroll=False)
#             y = outputs
#         else:
#             # no batch size specified, therefore the layer will be able
#             # to process batches of any size
#             # we can go with reshape-based implementation for performance
#             actual_input_shape = K.shape(inputs)
#
#             # reshaped_input_shape = (actual_input_shape[0] * actual_input_shape[1], actual_input_shape[2])
#             reshaped_axis_size = actual_input_shape[0]
#             for i in range(self.n_distribution_axes):
#                 reshaped_axis_size *= actual_input_shape[i + 1]
#             reshaped_input_shape = (reshaped_axis_size,) + tuple(actual_input_shape[self.n_distribution_axes + 1:])
#
#             # reshaped_input_shape = (actual_input_shape[0] * actual_input_shape[1],) + tuple(actual_input_shape[2:])
#             # print "reshaped_input_shape", len(reshaped_input_shape), reshaped_input_shape
#             # print "reshape input ..."
#             inputs = K.reshape(inputs, reshaped_input_shape, ndim=len(reshaped_input_shape))
#             # print "call inner model ..."
#             y = self.layer.call(inputs)  # (nb_samples * timesteps, ...)
#             actual_output_shape = K.shape(y)
#
#             reshaped_output_shape = tuple(actual_input_shape[:self.n_distribution_axes + 1]) + tuple(
#                 actual_output_shape[1:])
#
#             # reshaped_output_shape = (actual_input_shape[0], actual_input_shape[1]) + tuple(actual_output_shape[1:])
#             # print "reshaped_output_shape", len(reshaped_output_shape), reshaped_output_shape
#
#             # print "reshape output ..."
#             y = K.reshape(y, reshaped_output_shape, ndim=len(reshaped_output_shape))
#             # print "done!"
#         # Apply activity regularizer if any:
#         if (hasattr(self.layer, 'activity_regularizer') and self.layer.activity_regularizer is not None):
#             regularization_loss = self.layer.activity_regularizer(y)
#             self.add_loss(regularization_loss, inputs)
#
#         # print "all done!"
#         return y


class TimeDistributed3Dto2D(Wrapper):
    # excluding Batch
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(TimeDistributed3Dto2D, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed3Dto2D, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []

            input_length = input_shape[1]
            if K.backend() == 'tensorflow' and len(input_shape) > 3:
                if input_length is None:
                    raise Exception('When using TensorFlow, you should define '
                                    'explicitly the number of timesteps of '
                                    'your sequences.\n'
                                    'If your first layer is an Embedding, '
                                    'make sure to pass it an "input_length" '
                                    'argument. Otherwise, make sure '
                                    'the first layer has '
                                    'an "input_shape" or "batch_input_shape" '
                                    'argument, including the time axis.')
                unroll = True
            else:
                unroll = False
            last_output, outputs, states = K.rnn(step, X, initial_states=[], input_length=input_length, unroll=unroll)
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            actual_input_shape = K.shape(X)
            # print "actual_input_shape", actual_input_shape
            reshaped_input_shape = (
                actual_input_shape[0] * actual_input_shape[1], actual_input_shape[2], actual_input_shape[3])

            # print "reshaped_shape", reshaped_input_shape

            X = K.reshape(X, reshaped_input_shape, ndim=3)  # (nb_samples * timesteps, ...)
            y = self.layer.call(X)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            actual_output_shape = K.shape(y)
            # output_shape = self.get_output_shape_for(input_shape)
            reshaped_output_shape = (actual_input_shape[0], actual_input_shape[1], actual_output_shape[1])
            y = K.reshape(y, reshaped_output_shape, ndim=3)
        return y


class TimeDistributed2Dto3D(Wrapper):
    # excluding Batch
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(TimeDistributed2Dto3D, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed2Dto3D, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []

            input_length = input_shape[1]
            if K.backend() == 'tensorflow' and len(input_shape) > 3:
                if input_length is None:
                    raise Exception('When using TensorFlow, you should define '
                                    'explicitly the number of timesteps of '
                                    'your sequences.\n'
                                    'If your first layer is an Embedding, '
                                    'make sure to pass it an "input_length" '
                                    'argument. Otherwise, make sure '
                                    'the first layer has '
                                    'an "input_shape" or "batch_input_shape" '
                                    'argument, including the time axis.')
                unroll = True
            else:
                unroll = False
            last_output, outputs, states = K.rnn(step, X, initial_states=[], input_length=input_length, unroll=unroll)
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            actual_input_shape = K.shape(X)
            # print "actual_input_shape", actual_input_shape
            reshaped_input_shape = (actual_input_shape[0] * actual_input_shape[1], actual_input_shape[2])

            # print "reshaped_shape", reshaped_input_shape

            X = K.reshape(X, reshaped_input_shape, ndim=2)  # (nb_samples * timesteps, ...) ndim of target shape
            y = self.layer.call(X)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            actual_output_shape = K.shape(y)
            # output_shape = self.get_output_shape_for(input_shape)
            reshaped_output_shape = (
                actual_input_shape[0], actual_input_shape[1], actual_output_shape[1], actual_output_shape[2])
            y = K.reshape(y, reshaped_output_shape, ndim=4)
        return y
