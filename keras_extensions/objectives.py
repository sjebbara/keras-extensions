import keras.backend as K


def margin_mse(y_true, y_pred, margin=0):
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.switch(err < margin, 0, err)


def margin_ranking_loss(label, diff):
    ranking_loss = K.maximum(0, 1 - diff)

    return ranking_loss
