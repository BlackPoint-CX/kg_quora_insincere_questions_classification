import os
from collections import defaultdict
import re
import tensorflow as tf
import codecs
import logging
import pandas as pd
import numpy as np
from tensorflow.python.ops.lookup_ops import index_table_from_file

from commons_utils import UNK_IND
from config_utils import LOG_DIR, SOURCE_DIR, EMBEDDING_DIR

logging.basicConfig(filename=os.path.join(LOG_DIR, 'vocab_utils.log'),
                    filemode='w+')


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


def preprocessing_csv():
    for file in ['train.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(SOURCE_DIR, file))
        df['question_text_new'] = df.apply(lambda row: ' '.join(preprocessing_sentence(row['question_text'])), axis=1)

        df['question_text'] = df['question_text_new']
        df = df.drop(['question_text'], axis=1)
        df.to_csv(os.path.join(SOURCE_DIR, 'new_' + file), index=False)


# TODO : filter out those valid word and relate embeddings from whole embeddings.
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


def build_vocab_file_from_csv():
    files = [os.path.join(SOURCE_DIR, file_name) for file_name in ['train.csv', 'test.csv']]
    word_count_dict = build_word_count_dict(files)  # 220822

    word_list = []  # 2195896
    with codecs.getreader('utf-8')(open(os.path.join(SOURCE_DIR, 'embedding_gen_vocab.txt'), 'rb')) as r_file:
        for line in r_file:
            word_list.append(line.strip())

    word_dict = {}
    for word in word_list:
        word_dict[word] = 1

    new_word_list = []
    for word in word_dict.keys():
        if word in word_count_dict:
            new_word_list.append(word)

    with codecs.getreader('utf-8')(open(os.path.join(SOURCE_DIR, 'vocab.txt'), 'w+')) as w_file:
        for word in new_word_list:
            w_file.write('%s\n' % word)


def build_vocab_file_from_glove(glove_file=os.path.join(EMBEDDING_DIR, 'glove.840B.300d.txt')):
    with codecs.getreader('utf-8')(open(glove_file, 'rb')) as r_file, \
            open(os.path.join(SOURCE_DIR, 'embedding_gen_vocab.txt'), 'w+') as w_file:
        for line in r_file:
            word = line.split(' ')[0].strip()
            w_file.write('%s\n' % word)


def build_vocab_table(vocab_file):
    """
    Create vocab table from vocabulary file.
    :param vocab_file:
    :return:
    """
    vocab_table = index_table_from_file(vocab_file, default_value=UNK_IND)
    return vocab_table
