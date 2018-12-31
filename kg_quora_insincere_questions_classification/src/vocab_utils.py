import os
from collections import defaultdict
import re

import codecs
import logging
import pandas as pd
import numpy as np
from tensorflow.python.ops.lookup_ops import index_table_from_file

from config_utils import LOG_DIR, SOURCE_DIR, EMBEDDING_DIR

logging.basicConfig(filename=os.path.join(LOG_DIR, 'vocab_utils.log'),
                    filemode='w+')

VOCAB_SIZE_THRESHOLD_CPU = 50000
UNK_IND = 0


# question_text =
# """"What is~!@#$%^&*()_+}{|":<>?`[]\;',./' the book ""Nothing Succeeds Like Success"" about?"""""

def load_vocab(vocab_file):
    """
    Load word from vocab file.
    :param vocab_file:
    :return:
    """
    word_list = []
    with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'rb')) as r_file:
        for word in r_file:
            word_list.append(word.strip())
    num_word = len(word_list)
    return num_word, word_list


def _load_pretrained_embedding(embedding_file):
    """
    Loading embedding from Glove / word2vec formatted text file.
    For word2vec format, the first line will be : <num_words> <embedding_dim>
    :param embedding_file:
    :return:
        num_words : number of words.
        embedding_dim : dimension of embedding
        word_embedding_dict : dict(word : embedding)

    """
    word_embedding_dict = dict()
    num_words, embedding_dim = None, None
    is_first_line = True
    with codecs.getreader('utf-8')(tf.gfile.GFile(embedding_file, 'rb')) as r_file:
        for line in r_file:

            eles = line.rstrip().split(' ')
            if is_first_line:
                if len(eles) == 2:  # header line
                    [num_words, embedding_dim] = map(int, eles)
                    continue

            word = eles[0]
            embedding = list(map(float, eles[1:]))
            if embedding_dim:
                if len(embedding) != embedding_dim:
                    logging.warning('Ignoring %s since embedding size is inconsistent.' % word)
            else:
                embedding_dim = len(embedding)
            word_embedding_dict[word] = embedding

    return num_words, embedding_dim, word_embedding_dict


def load_embedding(vocab_file, embedding_file, num_trainable_words=0):
    """
    According vocabulary to load relate embedding;
    Assign initial value for trainable words.
    :param vocab_file:
    :param embedding_file:
    :param num_trainable_words:
    :return:
    """
    num_word, word_list = load_vocab(vocab_file)
    trainable_words = word_list[:num_trainable_words]

    num_words, embedding_dim, word_embedding_dict = _load_pretrained_embedding(embedding_file)

    for word in trainable_words:
        if word not in word_embedding_dict:
            word_embedding_dict[word] = [0.0] * embedding_dim

    embedding_mat = tf.constant(np.array([word_embedding_dict[word] for word in word_list], dtype=tf.float32))
    embedding_mat_const = tf.slice(embedding_mat, [num_trainable_words, 0], [-1, -1])
    embedding_mat_var = tf.get_variable('embedding_mat_var', [num_trainable_words, embedding_dim], dtype=tf.float32)

    return tf.concat([embedding_mat_var, embedding_mat_const], axis=0)


def _get_embedding_device(vocab_size):
    """
    Choose proper device for vocab embedding.
    :param vocab_size:
    :return:
    """
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        device = '/cpu:0'
    else:
        device = '/gpu:0'
    return device


def build_or_load_embedding(name, vocab_file, embedding_file, vocab_size, embedding_dim):
    if vocab_file and embedding_file:
        embedding = load_embedding(vocab_file=vocab_file, embedding_file=embedding_file)
    else:
        with tf.device(_get_embedding_device(vocab_size)):
            embedding = tf.get_variable(name=name, shape=[vocab_size, embedding_dim], dtype=tf.float32)

    return embedding


def rmv_end_with(sequence, symbol):
    if sequence.endswith(symbol):
        sequence = sequence[:-1]
    return sequence


def preprocessing_sentence(sequence):
    sequence = sequence.lower().strip()
    if sequence.startswith('\'') or sequence.startswith('"'):
        sequence = sequence[1:]
    if sequence.endswith('\'') or sequence.endswith('"'):
        sequence = sequence[:-1]

    symbol_list = ['?', '!', '.']
    for symbol in symbol_list:
        sequence = rmv_end_with(sequence, symbol)

    word_list = sequence.split()
    for idx, word in enumerate(word_list):
        for symbol in symbol_list:
            word_list[idx] = rmv_end_with(word, symbol)
        if word_list[idx].startswith('http'):
            word_list[idx] = '<url>'

    sequence = ' '.join(word_list)

    r = '[^A-Za-z0-9- ]+'
    sequence = re.sub(r, ' ', sequence)
    word_list = [ele for ele in sequence.split() if ele]
    return word_list


# TODO : filter out those valid word and relate embedding from whole embeddings.
def build_word_count_dict(files):
    word_count_dict = defaultdict(int)
    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            question_text = row['question_text']
            question_text = question_text.lower()
            words_list = preprocessing_sentence(question_text)
            for word in words_list:
                word_count_dict[word] += 1
    return word_count_dict


def build_vocab_file():
    files = [os.path.join(SOURCE_DIR, file_name) for file_name in ['train.csv', 'test.csv']]
    word_count_dict = build_word_count_dict(files)

    with open(os.path.join(SOURCE_DIR, 'vocab_file.txt'), 'w+') as w_file:
        for word in word_count_dict.keys():
            w_file.write('%s\n' % word)


def build_vocab_table(vocab_file):
    """
    Create vocab table from vocabulary file.
    :param vocab_file:
    :return:
    """
    vocab_table = index_table_from_file(vocab_file, default_value=UNK_IND)
    return vocab_table
