"""
FROM : https://www.kaggle.com/shujian/mix-of-nn-models-based-on-meta-embeddings/notebook
"""
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

import numpy as np  # linear algebra
import pandas as pd  # input processing, CSV file I/O (e.g. pd.read_csv)
import os

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input input files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# print(os.listdir("../input"))

from attention_layer import Attention
from embeddinga_utils import load_glove

embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embeddings vector)
maxlen = 70  # max number of words in a question to use



def load_and_prec():
    train_df = pd.read_csv(os.path.join(SOURCE_DIR, "train_clean.csv"))
    test_df = pd.read_csv(os.path.join(SOURCE_DIR, "test_clean.csv"))
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

    # fill up the missing values
    train_X = train_df["question_text"].fillna("_#_").values
    val_X = val_df["question_text"].fillna("_#_").values
    test_X = test_df["question_text"].fillna("_#_").values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    # Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    # shuffling the input
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))

    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]

    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index





# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)  # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, epochs=2):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y, best_score


train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()
embedding_matrix_1 = load_glove(word_index)
# embedding_matrix_2 = load_fasttext(word_index)
# embedding_matrix_3 = load_para(word_index)

# Simple average: http://aclweb.org/anthology/N18-2031

# We have presented an argument for averaging as
# a valid meta-embeddings technique, and found experimental
# performance to be close to, or in some cases
# better than that of concatenation, with the
# additional benefit of reduced dimensionality


# Unweighted DME in https://arxiv.org/pdf/1804.07983.pdf

# “The downside of concatenating embeddings and
#  giving that as input to an RNN encoder, however,
#  is that the network then quickly becomes inefficient
#  as we combine more and more embeddings.”

# embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 0)
# embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis=0)
embedding_matrix = np.mean([embedding_matrix_1], axis=0)
np.shape(embedding_matrix)

outputs = []
pred_val_y, pred_test_y, best_score = train_pred(model_gru_srk_atten(embedding_matrix), epochs=2)
outputs.append([pred_val_y, pred_test_y, best_score, 'gru atten srk'])
#
# pred_val_y, pred_test_y, best_score = train_pred(model_cnn(embedding_matrix), epochs=2)
# outputs.append([pred_val_y, pred_test_y, best_score, '2d CNN'])
#
# pred_val_y, pred_test_y, best_score = train_pred(model_cnn(embedding_matrix_1), epochs=2)  # GloVe only
# outputs.append([pred_val_y, pred_test_y, best_score, '2d CNN GloVe'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_du(embedding_matrix), epochs=2)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix), epochs=3)
outputs.append([pred_val_y, pred_test_y, best_score, '2 LSTM w/ attention'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix_1), epochs=3)  # Only GloVe
outputs.append([pred_val_y, pred_test_y, best_score, '2 LSTM w/ attention GloVe'])
#
# pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix_3), epochs=3)  # Only Para
# outputs.append([pred_val_y, pred_test_y, best_score, '2 LSTM w/ attention Para'])

outputs.sort(key=lambda x: x[2])  # Sort the output by val f1 score
weights = [i for i in range(1, len(outputs) + 1)]
weights = [float(i) / sum(weights) for i in weights]
print(weights)

for output in outputs:
    print(output[2], output[3])

# pred_val_y = np.sum([outputs[i][0] * weights[i] for i in range(len(outputs))], axis = 0)
pred_val_y = np.mean([outputs[i][0] for i in range(len(outputs))], axis=0)  # to avoid overfitting, just take average

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))

thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

# pred_test_y = np.sum([outputs[i][1] * weights[i] for i in range(len(outputs))], axis = 0)
pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis=0)

pred_test_y = (pred_test_y > best_thresh).astype(int)
test_df = pd.read_csv(os.path.join(SOURCE_DIR,"test_clean.csv"), usecols=["qid"])
out_df = pd.DataFrame({"qid": test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

import pandas as pd
test_df = pd.read_csv("/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/src/public_solutions/submission.csv", usecols=["prediction"])
list(test_df['prediction'])
type(test_df)
