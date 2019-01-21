from keras import Input, Model
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    concatenate, Dense, Dropout
from sklearn.metrics import f1_score

from keras.engine import Layer
from keras import backend as K, initializers, regularizers, constraints

from tuple_utils import TrainTuple


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def bmdl_0(max_len, max_features, embedding_matrix):
    def build_fn(max_len, max_features, embedding_matrix):
        embedding_dim = embedding_matrix.shape[1]
        inp = Input(shape=(max_len,))
        x = Embedding(input_dim=max_features, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(
            inp)
        x = Bidirectional(CuDNNLSTM(5, return_sequences=True))(x)
        y = Bidirectional(CuDNNGRU(2, return_sequences=True))(x)
        atten_1 = Attention(max_len)(x)
        atten_2 = Attention(max_len)(y)
        avg_pool = GlobalAveragePooling1D()(y)
        max_pool = GlobalMaxPooling1D()(y)

        conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
        conc = Dense(128, activation='relu')(conc)
        conc = Dropout(0.1)(conc)
        outp = Dense(1, activation='sigmoid')(conc)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

        return model

    param_grid = dict(max_len=[max_len], max_features=[max_features], embedding_matrix=[embedding_matrix])

    return TrainTuple(index_name=0, build_fn=build_fn, param_grid=param_grid)


def bmdl_1(max_len, max_features, embedding_matrix):
    def build_fn(max_len, max_features, embedding_matrix):
        embedding_dim = embedding_matrix[1]
        inp = Input(shape=(max_len,))
        x = Embedding(input_dim=max_features, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(
            inp)

        x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(units=128))(x)

        x = Dense(units=56)(x)
        x = Dropout(rate=0.1)(x)
        x = Dense(units=1)(x)

        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
        return model

    param_grid = dict(max_len=[max_len], max_features=[max_features], embedding_matrix=[embedding_matrix])

    return TrainTuple(index_name=1, build_fn=build_fn, param_grid=param_grid)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
