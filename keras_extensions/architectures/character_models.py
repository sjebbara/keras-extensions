from keras.engine import Input
from keras.engine import Layer
from keras.engine import Model
from keras.engine import merge
from keras.layers import Embedding, GRU, TimeDistributed, Dense, Dropout, Convolution1D, Lambda, ELU, Activation, \
    Bidirectional, RepeatVector, Masking, Flatten, Reshape, Permute, LocallyConnected1D, ParametricSoftplus, LeakyReLU, \
    BatchNormalization, GlobalMaxPooling1D
import keras.backend as K
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.regularizers import WeightRegularizer
from keras_extensions.activations import Softmax, normalize, inverse_quadratic_kernel
from keras_extensions.layers.attention import WeightedSum
from keras_extensions.layers.core import OneHotEmbedding, LearnedTensor, SqueezeSequence2Vector, BilinearProduct2, \
    TimeDistributedBilinearProduct, RepeatToMatch, TimeDistributedMerge, TimeDistributedBilinearTensorProduct, \
    TDBilinearLowRankTensorProduct, LogSumExp, ReshapeLast, Expand, BatchTimeDot, DynamicFlatten, Squeeze, RBF, \
    MultiConvolution1D
from keras_extensions.layers.wrappers import TimeDistributed2Dto3D, BetterTimeDistributed


def character_embedding_model(char_embedding_weights, char_word_embedding_size, char_word_embedding_filter_length,
                              dropout, character_dropout=0, l2=0, **kwargs):
    char_input_size, char_embedding_size = char_embedding_weights.shape

    default_initialization = "he_normal"
    default_activation = ELU()

    l2 = WeightRegularizer(l2) if l2 else None
    ######### Define Network Inputs ##########
    char_input = Input(shape=(None,), dtype='int32', name='char_input')

    ### Embed input
    char_embeddings = Embedding(char_input_size, char_embedding_size, dropout=character_dropout, W_regularizer=l2)(
        char_input)

    ### Compute word embeddings from character embeddings
    char_word_embeddings = Convolution1D(char_word_embedding_size, char_word_embedding_filter_length,
                                         activation="linear", init=default_initialization, border_mode="same")(
        char_embeddings)
    char_word_embeddings = default_activation(char_word_embeddings)
    char_word_embeddings = GlobalMaxPooling1D(name="char_word_embeddings")(char_word_embeddings)

    model = Model(input=[char_input], output=[char_word_embeddings])

    return model


def residual_character_embedding_model(char_input_size, char_embedding_size, char_embedding_weights,
                                       char_word_embedding_size, n_residuals, residual_filter_length, dropout,
                                       character_dropout=0, l2=0, batch_normalization=False, **kwargs):
    if char_embedding_weights:
        char_input_size, char_embedding_size = char_embedding_weights.shape

    default_initialization = "he_normal"
    default_activation = ELU()

    l2 = WeightRegularizer(l2) if l2 else None

    ######### Define Network Inputs ##########
    char_input = Input(shape=(None,), dtype='int32', name='char_input')
    ### Embed input
    char_embeddings = Embedding(char_input_size, char_embedding_size,
                                weights=char_embedding_weights if char_embedding_weights else None,
                                dropout=character_dropout)(char_input)

    residual_input = Convolution1D(char_word_embedding_size, residual_filter_length, activation="linear",
                                   init=default_initialization, border_mode="same")(char_embeddings)
    # residual_input = default_activation(residual_input)

    for r in range(n_residuals):
        residual = residual_input
        if batch_normalization:
            residual = BatchNormalization()(residual)
        residual = ELU()(residual)
        residual = Convolution1D(char_word_embedding_size, residual_filter_length, activation="linear",
                                 border_mode="same", W_regularizer=l2)(residual)
        if batch_normalization:
            residual = BatchNormalization()(residual)
        residual = ELU()(residual)
        residual = Dropout(dropout)(residual)
        residual = Convolution1D(char_word_embedding_size, residual_filter_length, activation="linear",
                                 border_mode="same", W_regularizer=l2)(residual)
        residual_input = merge([residual_input, residual], mode="sum")

    char_word_embeddings = GlobalMaxPooling1D()(residual_input)
    char_word_embeddings = default_activation(char_word_embeddings)

    model = Model(input=[char_input], output=[char_word_embeddings])

    return model


def residual_multiconv_character_embedding_model(char_input_size, char_embedding_size, char_embedding_weights,
                                                 conv_sizes, n_residuals, residual_filter_length, dropout,
                                                 character_dropout=0, l2=0, batch_normalization=False, **kwargs):
    if char_embedding_weights:
        char_input_size, char_embedding_size = char_embedding_weights.shape

    default_initialization = "he_normal"
    default_activation = ELU()

    l2 = WeightRegularizer(l2) if l2 else None

    ######### Define Network Inputs ##########
    char_input = Input(shape=(None,), dtype='int32', name='char_input')
    ### Embed input
    char_embeddings = Embedding(char_input_size, char_embedding_size,
                                weights=char_embedding_weights if char_embedding_weights else None,
                                dropout=character_dropout, W_regularizer=l2)(char_input)

    residual_input = MultiConvolution1D((None, char_embedding_size), conv_sizes, activation="linear")(char_embeddings)
    # residual_input = default_activation(residual_input)

    char_word_embedding_size = sum(s[0] for s in conv_sizes)

    for r in range(n_residuals):
        residual = residual_input
        if batch_normalization:
            residual = BatchNormalization()(residual)
        residual = ELU()(residual)
        residual = Convolution1D(char_word_embedding_size, residual_filter_length, activation="linear",
                                 border_mode="same")(residual)
        if batch_normalization:
            residual = BatchNormalization()(residual)
        residual = ELU()(residual)
        residual = Dropout(dropout)(residual)
        residual = Convolution1D(char_word_embedding_size, residual_filter_length, activation="linear",
                                 border_mode="same")(residual)
        residual_input = merge([residual_input, residual], mode="sum")

    char_word_embeddings = GlobalMaxPooling1D()(residual_input)
    char_word_embeddings = default_activation(char_word_embeddings)

    model = Model(input=[char_input], output=[char_word_embeddings])

    return model


def residual_block(input_shape, residual_filter_length, dropout, batch_normalization):
    residual_input = Input(input_shape, name="input_sequence")
    sequence_length, element_embedding_size = input_shape

    residual = residual_input
    if batch_normalization:
        residual = BatchNormalization()(residual)
    residual = ELU()(residual)
    residual = Convolution1D(element_embedding_size, residual_filter_length, activation="linear", border_mode="same")(
        residual)
    if batch_normalization:
        residual = BatchNormalization()(residual)
    residual = ELU()(residual)
    residual = Dropout(dropout)(residual)
    residual = Convolution1D(element_embedding_size, residual_filter_length, activation="linear", border_mode="same")(
        residual)
    residual_output = merge([residual_input, residual], mode="sum")
    model = Model(residual_input, residual_output)
    return model
