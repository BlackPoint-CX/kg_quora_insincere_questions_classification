import codecs
import logging
import os
import numpy as np
import tensorflow as tf

from commons_utils import VOCAB_SIZE_THRESHOLD_CPU
from config_utils import LOG_DIR, EMBEDDING_DIR, SOURCE_DIR
from vocab_utils import load_vocab

logging.basicConfig(filename=os.path.join(LOG_DIR, 'embedding_utils.log'),
                    filemode='w+')


def _get_embedding_device(vocab_size):
    """
    Choose proper device for vocab embeddings.
    :param vocab_size:
    :return:
    """
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        device = '/cpu:0'
    else:
        device = '/gpu:0'
    return device


def _load_pretrained_embedding(embedding_file):
    """
    Loading embeddings from Glove / word2vec formatted text file.
    For word2vec format, the first line will be : <num_words> <embedding_dim>
    :param embedding_file:
    :return:
        num_words : number of words.
        embedding_dim : dimension of embeddings
        word_embedding_dict : dict(word : embeddings)

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
                    logging.warning('Ignoring %s since embeddings size is inconsistent.' % word)
            else:
                embedding_dim = len(embedding)
            word_embedding_dict[word] = embedding

    return num_words, embedding_dim, word_embedding_dict


def load_embedding(vocab_file, embedding_file, num_trainable_words=0):
    """
    According vocabulary to load relate embeddings;
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

    embedding_mat = tf.constant(np.array([word_embedding_dict[word] for word in word_list]), dtype=tf.float32)
    embedding_mat_const = tf.slice(embedding_mat, [num_trainable_words, 0], [-1, -1])
    embedding_mat_var = tf.get_variable('embedding_mat_var', [num_trainable_words, embedding_dim], dtype=tf.float32)

    return tf.concat([embedding_mat_var, embedding_mat_const], axis=0)


def build_or_load_embedding(name, vocab_file, embedding_file, vocab_size, embedding_dim):
    if vocab_file and embedding_file:
        embedding = load_embedding(vocab_file=vocab_file, embedding_file=embedding_file)
    else:
        with tf.device(_get_embedding_device(vocab_size)):
            embedding = tf.get_variable(name=name, shape=[vocab_size, embedding_dim], dtype=tf.float32)

    return embedding


def get_embedding_from_ori():
    num_words, embedding_dim, word_embedding_dict = _load_pretrained_embedding(
        os.path.join(EMBEDDING_DIR, 'glove.840B.300d.txt'))

    new_word_embedding_dict = {}

    num_word, word_list = load_vocab(os.path.join(SOURCE_DIR, 'vocab.txt'))

    with open(os.path.join(EMBEDDING_DIR, 'embeddings.txt'), 'w+') as w_file:
        for word in word_list:
            if word in word_embedding_dict:
                new_word_embedding_dict[word] = word_embedding_dict[word]

        for word, embedding in new_word_embedding_dict.items():
            w_file.write('%s %s\n' % (word, ' '.join(map(str, embedding))))
