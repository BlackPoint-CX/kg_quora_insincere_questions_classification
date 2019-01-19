import gc
import os
from collections import defaultdict
import sklearn
import tensorflow as tf
from keras import Input, Model
from keras.engine import Layer
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, initializers, regularizers, constraints, Dense, GRU, LSTM
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from functools import partial

from common_utils import INPUT_DIR

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 每个GPU现存上届控制在60%以内
# session = tf.Session(config=config)


# Preprocessing Functions Definition
def check_symbol_mid(text, symbol):
    l = text.split(symbol)
    if len(l) == 2 and l[0].isalpha() and l[1].isalpha():
        return True
    else:
        return False


def check_contain_numeric(text):
    for single_char in list(text):
        if single_char.isnumeric():
            return True
        else:
            pass

    return False


check_hyphen_mid = partial(check_symbol_mid, symbol='-')
check_slash_mid = partial(check_symbol_mid, symbol='/')
check_underline_mid = partial(check_symbol_mid, symbol='_')
check_full_stop_mid = partial(check_symbol_mid, symbol='.')


def clean_word(word, nest_flag=False):
    if nest_flag:
        word = rmv_uncessary_mark(word)

    if word.startswith('?') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('?') and word[:-1].isalpha():
        result = word[:-1]
    elif word.endswith("'?") and word[:-2].isalpha():
        result = word[:-2]
    elif word.endswith("'") and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith('.') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('.') and word[:-1].isalpha():
        result = word[:-1]
    elif word.endswith('.?') and word[:-2].isalpha():
        result = word[:-2]
    elif word.startswith(',') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith(',') and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith('!') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('!') and word[:-1].isalpha():
        result = word[:-1]
    elif word.isnumeric():
        result = word
    elif word.endswith('\'s') and word[:-2].isalpha():
        result = word
    elif word.endswith('\'s?') and word[:-3].isalpha():
        result = word[:-2]
    elif word.endswith('’s') and word[:-2].isalpha():
        result = word[:-2] + "'s"
    elif word.startswith(':') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith(':') and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith('"') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('"') and word[:-1].isalpha():
        result = word[:-1]
    elif word.endswith('"?') and word[:-2].isalpha():
        result = word[:-2]
    elif word.startswith('"') and word.endswith('"') and word[1:-1].isalpha():
        result = word[1:-1]
    elif word.startswith(';') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith(';') and word[:-1].isalpha():
        result = word[:-1]
    elif check_hyphen_mid(word):
        result = word
    elif word.endswith('?') and check_hyphen_mid(word[:-1]):
        result = word[:-1]
    elif check_slash_mid(word):
        word_list = word.split('/')
        result = ' '.join(word_list)
    elif word.startswith('(') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('(') and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith(')') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith(')') and word[:-1].isalpha():
        result = word[:-1]
    elif word.endswith(')?') and word[:-2].isalpha():
        result = word[:-2]
    elif word.startswith('(') and word.endswith(')') and word[1:-1].isalpha():
        result = word[1:-1]
    elif word.startswith('[') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith('[') and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith(']') and word[1:].isalpha():
        result = word[1:]
    elif word.endswith(']') and word[:-1].isalpha():
        result = word[:-1]
    elif word.startswith(']') and word.endswith(']') and word[1:-1].isalpha():
        result = word[1:-1]
    elif check_underline_mid(word):
        result = word
    elif check_full_stop_mid(word):
        result = word
    elif check_contain_numeric(word):
        result = word
    else:
        if nest_flag:
            result = word
        else:
            nest_flag = True
            result = clean_word(word, nest_flag)

    return result


def rmv_uncessary_mark(seq):
    """
    Used for removing uncessary mark at beginning and end.
    :return:
    """

    def _rmv_uncessary_mark(seq_list):
        start_idx = 0
        for idx, c in enumerate(seq_list):
            if not c.isalpha():
                pass
            else:
                start_idx = idx
                break
        seq_list = seq_list[start_idx:]
        return seq_list

    seq = ''.join(_rmv_uncessary_mark(list(seq)))
    seq_reverse = list(seq)
    seq_reverse.reverse()
    seq_reverse = _rmv_uncessary_mark(seq_reverse)
    seq_reverse.reverse()
    seq_result = ''.join(seq_reverse)
    return seq_result


def clean_question_text(row):
    question_text = row['question_text']
    word_list = question_text.split(' ')
    clean_word_list = []
    for ele in word_list:
        clean_word_list.append(clean_word(ele))
    clean_question_text = ' '.join(clean_word_list)
    row['question_text'] = clean_question_text
    return row


mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                'bigdata': 'big input', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def load_vocab(vocab_file_path):
    word_idx_table = defaultdict(int)
    with open(vocab_file_path, 'r') as r_file:
        for word in r_file:
            word = word.strip()
            word_idx_table[word] = len(word_idx_table) + 1

    idx_word_table = {v: k for k, v in word_idx_table.items()}
    return word_idx_table, idx_word_table


def build_word_count(train_df, test_df):
    union_ser = pd.concat([train_df['question_text'], test_df['question_text']], axis=0)
    split_ser = union_ser.apply(lambda text: text.split(' '))

    word_count = defaultdict(int)

    for word_list in split_ser:
        for word in word_list:
            word_count[word] += 1

    word_count = {t[0]: t[1] for t in sorted(word_count.items(), key=lambda t: t[1], reverse=True)}

    return word_count


def build_vocab(word_count, embedding_file_path, vocab_file_path):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(embedding_file_path) if len(o) > 100 and o.split(" ")[0] in word_count)

    vocab_list = list(embeddings_index.keys())
    with open(vocab_file_path, 'w+') as w_file:
        for word in vocab_list:
            w_file.write('%s\n' % word)


def new_load_glove(word_idx_table, embedding_file_path):
    max_features = len(word_idx_table)

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(embedding_file_path) if o.split(" ")[0] in word_idx_table)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features + 1, embed_size))
    for word, i in word_idx_table.items():
        # if i >= MAX_FEATURES:
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# Preprocessing Stage
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
# test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

train_df['question_text'] = train_df['question_text'].str.lower()
test_df['question_text'] = test_df['question_text'].str.lower()

train_df = train_df.apply(lambda row: clean_question_text(row), axis=1)
test_df = test_df.apply(lambda row: clean_question_text(row), axis=1)

train_df['question_text'] = train_df['question_text'].apply(lambda x: replace_typical_misspell(x))
test_df['question_text'] = test_df['question_text'].apply(lambda x: replace_typical_misspell(x))

embedding_file_path = '../input/embedding/glove.840B.300d.txt'
vocab_file_path = '../input/embedding/glove.vocab.txt'
# embedding_file_path = os.path.join(INPUT_DIR, 'embedding/glove.840B.300d.txt')
# vocab_file_path = os.path.join(INPUT_DIR, 'embedding/glove.vocab.txt')

word_count = build_word_count(train_df, test_df)
build_vocab(word_count, embedding_file_path, vocab_file_path)
word_idx_table, idx_word_table = load_vocab(vocab_file_path)

embedding_matrix = new_load_glove(word_idx_table, embedding_file_path)


# Modeling Functions Definition

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


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=EMBEDDING_SIZE, weights=[embedding_matrix],
                  trainable=False)(inp)
    # x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
    # x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(units=64, return_sequences=True))(x)
    x = Attention(MAX_LEN)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_pred(model, epochs=2):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = sklearn.metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)

    return pred_val_y, pred_test_y, best_score, best_thresh


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


# ===== Modeling =====
MAX_LEN = 90
MAX_FEATURES = int(130000)
EMBEDDING_SIZE = 300

np.random.seed(2019)
train_part_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)

# Tokenize the sentences
train_X = train_part_df['question_text'].values
val_X = val_df['question_text'].values
test_X = test_df['question_text'].values

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

# Pad the sentences
train_X = pad_sequences(train_X, maxlen=MAX_LEN)
val_X = pad_sequences(val_X, maxlen=MAX_LEN)
test_X = pad_sequences(test_X, maxlen=MAX_LEN)

# Get the target values
train_y = train_part_df['target'].values
val_y = val_df['target'].values

# Shuffling the input
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))
train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]

del train_df
del test_df
gc.collect()
pred_val_y, pred_test_y, best_score, best_thresh = train_pred(model_lstm_atten(embedding_matrix), epochs=3)

pred_test_y = (pred_test_y > best_thresh).astype(int)

sub = pd.read_csv('../input/sample_submission.csv')
# sub = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
out_df = pd.DataFrame({"qid": sub["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
