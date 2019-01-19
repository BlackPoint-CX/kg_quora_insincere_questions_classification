import gc
import os
import sklearn
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 每个GPU现存上届控制在60%以内
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import nltk as nltk
from keras import Input, Model
from keras.engine import Layer
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, initializers, regularizers, constraints, Dense, GRU, LSTM, \
    CuDNNGRU
from keras import backend as K
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))

from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, make_scorer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from attention_layer import Attention
from common_utils import TRAIN_FILE_PATH, TEST_FILE_PATH, PROJECT_DIR, SRC_DIR
from embedding_utils import new_load_glove
from preprocessing_utils import clean_text, clean_numbers, replace_typical_misspell
from vocab_utils import load_vocab

MAX_LEN = 100
MAX_FEATURES = 2 * 10 ** 5
EMBEDDING_SIZE = 300

# def model_lstm_atten(embedding_matrix):
#     inp = Input(shape=(MAX_LEN,))
#     x = Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_SIZE, weights=[embedding_matrix], trainable=False)(inp)
#     x = Bidirectional(CuDNNLSTM(num_units=128, return_sequences=True))(x)
#     x = Bidirectional(CuDNNLSTM(num_units=64, return_sequences=True))(x)
#     x = Attention(MAX_LEN)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=inp, outputs=x)
#     model.compile(loss='binary_crossentroy', optimizer='adam', metrics=['accuracy'])
#
#     return model


X_train, y_train, X_val, y_val, X_test = None, None, None, None, None


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


def rmv_symbol(sequence, symbol):
    if sequence.startswith(symbol):
        sequence = sequence[1:]
    if sequence.endswith(symbol):
        sequence = sequence[:-1]
    return sequence


def preprocessing(row):
    question_text = row['question_text']
    for symbol in ['"', '\'']:
        rmv_symbol(sequence=question_text, symbol=symbol)
    row['question_text'] = question_text


def load_and_preprocessing(dataframe):
    dataframe = pd.DataFrame()
    dataframe['question_text'] = dataframe['question_text'].str.lower()  # Operation : Lower


def model_lstm_atten(embedding_matrix, maxlen, max_features, embed_size=300):
    max_features, embed_size = embedding_matrix.shape
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print('Process Step 0')
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df['question_text'] = test_df['question_text'].str.lower()

    # train_df['question_text'] = train_df['question_text'].apply(lambda x: replace_typical_misspell(x))
    # test_df['question_text'] = test_df['question_text'].apply(lambda x: replace_typical_misspell(x))
    # 
    # train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))
    # test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text(x))
    # 
    # train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_numbers(x))
    # test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_numbers(x))

    print('Process Step 1')
    embedding_file_path = os.path.join(SRC_DIR, 'input/embeddings/glove.840B.300d/glove.840B.300d.txt')
    vocab_file_path = os.path.join(SRC_DIR, 'input/embeddings/glove.840B.300d/glove.vocab.txt')
    word_idx_table, idx_word_table = load_vocab(vocab_file_path)
    embedding_matrix = new_load_glove(word_idx_table, embedding_file_path)

    tokenizer = Tokenizer()
    tokenizer.word_index = word_idx_table
    tokenizer.index_word = idx_word_table

    maxlen = 100
    max_features = len(word_idx_table)
    X = train_df['question_text']
    X_test = test_df['question_text']

    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad the sentences
    X = pad_sequences(X, maxlen=maxlen)
    y = train_df['target'].values

    X_test = pad_sequences(X_test, maxlen=maxlen)
    sub = test_df[['qid']]

    del train_df, test_df
    gc.collect()

    print('Process Step 2')
    # cv = StratifiedShuffleSplit(test_size=0.1, random_state=2019, n_splits=1)
    #
    # model = KerasClassifier(build_fn=model_lstm_atten)
    #
    # param_grid = dict(embedding_matrix=[embedding_matrix], maxlen=[maxlen], max_features=[max_features + 1])
    # scorer = make_scorer(f1_score)
    #
    # gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer)
    # gs.fit(X=X, y=y)
    # print(gs.best_score_)
    #
    # y_test_pred = gs.predict(X_test)
    # sub['prediction'] = y_test_pred
    # sub.to_csv("submission.csv", index=False)

    model = model_lstm_atten(embedding_matrix, maxlen, max_features + 1)
    model.fit(x=X, y=y, batch_size=64, epochs=5,validation_data=(X))
    y_test_pred = model.predict(X_test)
    sub['prediction'] = y_test_pred
    sub.to_csv("submission.csv", index=False)

    # pd.read_csv('~/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/src/own_solutions/self_solution_1/submission.csv')['prediction'].apply(lambda x : 1 if x > 0.5 else 0 ).tolist()
