

from keras import Input, Model
from keras.engine import Layer
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, initializers, regularizers, constraints, Dense
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

MAX_LEN = 100
MAX_FEATURES = 10 ** 5
EMBEDDING_SIZE = 300

TRAIN_FILE_PATH = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/src/input/train.csv'
TEST_FILE_PATH = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/src/input/test.csv'

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


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_SIZE, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(num_units=128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(num_units=64, return_sequences=True))(x)
    x = Attention(MAX_LEN)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentroy', optimizer='adam', metrics=['accuracy'])

    return model


X_train, y_train, X_val, y_val, X_test= None, None, None, None, None


def train_pred(model, epochs, batch_size=64):
    for idx in range(epochs):
        model = Model()
        model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))

        # Evaluation on Validation
        best_threshold, best_score = 0.0, 0.0

        y_val_pred = model.predict(x=X_val, batch_size=batch_size, verbose=0)

        for threshold in np.arange(0, 1, 0.01):
            score = f1_score(y_true=y_val, y_pred=(y_val_pred > threshold).astype(dtype=int))
            if score > best_score:
                best_threshold = threshold
                best_score = score

        y_test_pred = model.predict(x=X_test, batch_size=batch_size, verbose=0)

        return best_score, best_threshold, y_val_pred, y_test_pred



def load_and_preprocessing():
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)


    tokenizer = Tokenizer(num_words=MAX_FEATURES,lower=True,split=' ',)
