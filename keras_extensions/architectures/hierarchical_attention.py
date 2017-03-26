from keras.engine import Input
from keras.engine import Layer
from keras.engine import Model
from keras.engine import merge
from keras.layers import Embedding, GRU, TimeDistributed, Dense, Dropout, Convolution1D, Lambda, ELU, Activation, \
    Bidirectional, RepeatVector, Masking, Flatten, Reshape, Permute, LocallyConnected1D, ParametricSoftplus, LeakyReLU, \
    BatchNormalization
import keras.backend as K
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.regularizers import WeightRegularizer
from keras_extensions.activations import Softmax, normalize, inverse_quadratic_kernel
from keras_extensions.layers.attention import WeightedSum
from keras_extensions.layers.core import OneHotEmbedding, LearnedTensor, SqueezeSequence2Vector, BilinearProduct2, \
    TimeDistributedBilinearProduct, RepeatToMatch, TimeDistributedMerge, TimeDistributedBilinearTensorProduct, \
    TDBilinearLowRankTensorProduct, LogSumExp, ReshapeLast, Expand, BatchTimeDot, DynamicFlatten, Squeeze, RBF
from keras_extensions.layers.wrappers import TimeDistributed2Dto3D, BetterTimeDistributed


###############################
######## OTHER MODEL ##########
###############################

def hierarchical_attention_classification_model(word_embedding_weights, trainable_word_embeddings,
                                                sentence_embedding_size, word_prototype_size, document_embedding_size,
                                                sentence_prototype_size, n_classes, attention_activation,
                                                attention_normalization, input_dropout, rnn_dropout_W, rnn_dropout_U,
                                                dropout, **kwargs):
    word_input_size, word_embedding_size = word_embedding_weights.shape
    RNN = GRU

    word_embedding_layer = Embedding(input_dim=word_input_size, output_dim=word_embedding_size,
                                     weights=[word_embedding_weights], trainable=trainable_word_embeddings,
                                     dropout=input_dropout)
    word_rnn_layer = Bidirectional(
        RNN(sentence_embedding_size / 2, dropout_U=rnn_dropout_U, dropout_W=rnn_dropout_W, return_sequences=True,
            activation="tanh"), merge_mode="concat")
    sentence_rnn_layer = Bidirectional(
        RNN(document_embedding_size / 2, dropout_U=rnn_dropout_U, dropout_W=rnn_dropout_W, return_sequences=True,
            activation="tanh"), merge_mode="concat")

    ######### Define Network Inputs ##########
    text_input = Input(shape=(None, None), dtype='int32', name='text_input')

    print "build sentence model..."
    sentence_embedding_model, word_attention_model = single_sequence_attention_model(word_rnn_layer,
                                                                                     word_embedding_size,
                                                                                     word_prototype_size,
                                                                                     attention_activation,
                                                                                     attention_normalization,
                                                                                     dropout=dropout, **kwargs)

    print "build document model..."
    review_embedding_model, sentence_attention_model = single_sequence_attention_model(sentence_rnn_layer,
                                                                                       sentence_embedding_size,
                                                                                       sentence_prototype_size,
                                                                                       attention_activation,
                                                                                       attention_normalization,
                                                                                       dropout=dropout, **kwargs)
    ### Lookup word embeddings ###
    word_embedding_sequences = BetterTimeDistributed(word_embedding_layer)(text_input)  # B, S, W -> B, S, W, Dw

    ### Embed each sentence using attention model ###
    sentence_embeddings = BetterTimeDistributed(sentence_embedding_model)(
        word_embedding_sequences)  # B, S, W, Dw -> B, S, Ds
    sentence_embeddings = Dropout(dropout)(sentence_embeddings)
    aspect_sentence_embeddings = sentence_embeddings

    aspect_word_attentions = BetterTimeDistributed(word_attention_model)(
        word_embedding_sequences)  # B, S, W, Dw -> B, S, W

    ### Embed review using attention model ###
    aspect_review_embeddings = review_embedding_model(aspect_sentence_embeddings)  # B, S, Ds -> B, Dr
    aspect_review_embeddings = Dropout(dropout)(aspect_review_embeddings)
    aspect_sentence_attentions = sentence_attention_model(aspect_sentence_embeddings)  # B, S, Ds -> B, S

    aspect_polarities = Dense(n_classes, activation="softmax", name="class_output")(
        aspect_review_embeddings)  # B, Dr -> B, Av
    ###############################

    model = Model(input=[text_input], output=[aspect_polarities])
    print "compile model ..."
    model.compile(optimizer="adam", loss={"class_output": "categorical_crossentropy"})
    print "build review model functions..."
    model._make_train_function()
    model._make_predict_function()

    ### Permute attentions to comply with dimension order convention

    attention_model = Model(input=text_input,
                            output=[aspect_polarities, aspect_word_attentions, aspect_sentence_attentions])
    attention_model._make_predict_function()

    return model, attention_model


def single_sequence_attention_model(rnn_layer, sequence_input_size, prototype_size, attention_activation,
                                    attention_normalization, dropout=None, **kwargs):
    ######### Define Network Inputs ##########
    sequence_input = Input(shape=(None, sequence_input_size), dtype='float32', name='text_input')

    sequence_element_embedding = rnn_layer(sequence_input)  # (B, S, Ds)

    tmp_sequence_element_embedding = TimeDistributed(Dense(prototype_size, activation="linear"))(
        sequence_element_embedding)  # (B, S, Da)
    tmp_sequence_element_embedding = attention_activation_layer(attention_activation, (1,))(
        tmp_sequence_element_embedding)
    tmp_sequence_element_embedding = Dropout(dropout)(tmp_sequence_element_embedding)

    attention_scores = TimeDistributed(Dense(1))(tmp_sequence_element_embedding)  # B,S,Da -> B,S,1
    attention_scores = Squeeze(2)(attention_scores)
    attention_scores = attention_normalization_layer(attention_normalization)(
        attention_scores)  # (B, S) (last dim sums to 1)

    sequence_embedding = WeightedSum()([sequence_element_embedding, attention_scores])  # (B, S, Ds), (B, S) -> B, Ds

    embedding_model = Model(input=sequence_input, output=sequence_embedding)
    attention_model = Model(input=sequence_input, output=attention_scores)
    return embedding_model, attention_model


def single_sequence_multi_prototype_attention_model(rnn_layer, sequence_input_size, prototype_size, n_prototypes,
                                                    attention_activation, attention_normalization,
                                                    prototype_score_aggregation="max", dropout=None,
                                                    batch_normalization=False, **kwargs):
    ######### Define Network Inputs ##########
    sequence_input = Input(shape=(None, sequence_input_size), dtype='float32', name='text_input')

    if batch_normalization:
        sequence_element_embedding = BatchNormalization()(sequence_input)
    else:
        sequence_element_embedding = sequence_input

    sequence_element_embedding = rnn_layer(sequence_element_embedding)  # (B, S, Ds)
    ### project to tmp space
    tmp_sequence_element_embedding = TimeDistributed(Dense(prototype_size, activation="linear"))(
        sequence_element_embedding)  # (B, S, Da)

    if batch_normalization:
        tmp_sequence_element_embedding = BatchNormalization()(tmp_sequence_element_embedding)
    else:
        tmp_sequence_element_embedding = tmp_sequence_element_embedding

    tmp_sequence_element_embedding = attention_activation_layer(attention_activation, (1,))(
        tmp_sequence_element_embedding)
    tmp_sequence_element_embedding = Dropout(dropout)(tmp_sequence_element_embedding)

    ### compare to prototyps (dot product)
    attention_scores = TimeDistributed(Dense(n_prototypes))(tmp_sequence_element_embedding)  # B,S,Da -> B,S,P

    ### aggregate prototype similarities
    if prototype_score_aggregation == "max":
        attention_scores = Lambda(lambda X: K.max(X, axis=2), lambda shape: shape[:2])(attention_scores)  # B,S
    elif prototype_score_aggregation == "mean":
        attention_scores = Lambda(lambda X: K.mean(X, axis=2), lambda shape: shape[:2])(attention_scores)  # B,S
    elif prototype_score_aggregation == "lse":
        attention_scores = LogSumExp(axis=2, keepdims=False)(attention_scores)

    ### normalize attention scores
    attention_scores = attention_normalization_layer(attention_normalization)(
        attention_scores)  # (B, S) (last dim sums to 1)

    sequence_embedding = WeightedSum()([sequence_element_embedding, attention_scores])  # (B, S, Ds), (B, S) -> B, Ds

    embedding_model = Model(input=sequence_input, output=sequence_embedding)
    attention_model = Model(input=sequence_input, output=attention_scores)
    return embedding_model, attention_model


def single_sequence_rbf_attention_model(rnn_layer, sequence_input_size, prototype_size, n_prototypes,
                                        attention_activation, attention_normalization,
                                        prototype_score_aggregation="max", dropout=None, prototype_weights=None,
                                        **kwargs):
    ######### Define Network Inputs ##########
    sequence_input = Input(shape=(None, sequence_input_size), dtype='float32', name='text_input')

    sequence_element_embedding = rnn_layer(sequence_input)  # (B, S, Ds)
    ### project to tmp space
    tmp_sequence_element_embedding = TimeDistributed(Dense(prototype_size, activation="linear"))(
        sequence_element_embedding)  # (B, S, Da)
    tmp_sequence_element_embedding = attention_activation_layer(attention_activation, (1,))(
        tmp_sequence_element_embedding)
    tmp_sequence_element_embedding = Dropout(dropout)(tmp_sequence_element_embedding)

    ### compare to prototyps (dot product)
    attention_scores = BetterTimeDistributed(
        RBF(n_prototypes, kernel=inverse_quadratic_kernel, weights=[prototype_weights] if prototype_weights else None))(
        tmp_sequence_element_embedding)
    attention_scores = Dropout(dropout)(attention_scores)

    ### aggregate prototype similarities
    if prototype_score_aggregation == "max":
        attention_scores = Lambda(lambda X: K.max(X, axis=2), lambda shape: shape[:2])(attention_scores)  # B,S
    elif prototype_score_aggregation == "mean":
        attention_scores = Lambda(lambda X: K.mean(X, axis=2), lambda shape: shape[:2])(attention_scores)  # B,S
    elif prototype_score_aggregation == "lse":
        attention_scores = LogSumExp(axis=2, keepdims=False)(attention_scores)

    ### normalize attention scores
    attention_scores = attention_normalization_layer(attention_normalization)(
        attention_scores)  # (B, S) (last dim sums to 1)

    sequence_embedding = WeightedSum()([sequence_element_embedding, attention_scores])  # (B, S, Ds), (B, S) -> B, Ds

    embedding_model = Model(input=sequence_input, output=sequence_embedding)
    attention_model = Model(input=sequence_input, output=attention_scores)
    return embedding_model, attention_model


###############################

def attention_normalization_layer(attention_normalization):
    if attention_normalization == "normalize":
        act = normalize
    else:
        act = attention_normalization

    return Activation(act)


def attention_activation_layer(attention_activation, shared_axes):
    if attention_activation == "psoftplus":
        act = ParametricSoftplus(shared_axes=shared_axes)
    elif attention_activation == "elu":
        act = ELU()
    elif attention_activation == "lrelu":
        act = LeakyReLU()
    else:
        act = Activation(attention_activation)

    return act
