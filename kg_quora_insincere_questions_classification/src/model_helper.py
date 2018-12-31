from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, DropoutWrapper, DeviceWrapper, MultiRNNCell
import logging
import os

import iterator_utils
from config_utils import LOG_DIR
from model import Model
from vocab_utils import build_vocab_table

logging.basicConfig(filename=os.path.join(LOG_DIR, 'model_helper.log'))


class TrainTuple(namedtuple('TrainTuple', ['graph', 'model', 'iterator'])):
    pass


def build_single_cell(cell_type, learning_rate, num_units, dropout=0.0, device=None):
    """
    Build single cell.
    :param cell_type:
    :param learning_rate:
    :param num_units:
    :param dropout:
    :param device:
    :return:
    """
    cell_type = cell_type.lower()
    if cell_type == 'lstm':
        single_cell = tf.nn.rnn_cell.LSTMCell(learning_rate=learning_rate)
    elif cell_type == 'gru':
        single_cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
    elif cell_type == '':
        single_cell = LayerNormBasicLSTMCell(num_units=num_units)
    else:
        raise ValueError('Unknown cell_type : %s' % cell_type)

    if dropout:
        single_cell = DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))

    if device:
        single_cell = DeviceWrapper(cell=single_cell, device=device)

    return single_cell


def build_cell_list(num_layers, cell_type, learning_rate, num_units, dropout, device=None):
    """
    Build cell list.
    :param num_layers:
    :param cell_type:
    :param learning_rate:
    :param num_units:
    :param dropout:
    :param device:
    :return:
    """
    cell_list = []
    for i in range(num_layers):
        single_cell = build_single_cell(cell_type, learning_rate, num_units, dropout, device)
        cell_list.append(single_cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return MultiRNNCell(cells=cell_list)


def build_train_tuple(hparams):
    """
    Build train tuple <graph, model, iterator>
    :param hparams:
    :return:
    """
    graph = tf.Graph()
    with graph.as_default():
        vocab_table = build_vocab_table(hparams.vocab_file)
        input_dataset = CsvDataset(filenames=hparams.train_file,
                                   record_defaults=[[0.0], ['']])
        iterator = iterator_utils.build_train_iterator(train_dataset=input_dataset,
                                                       vocab_table=vocab_table,
                                                       batch_size=hparams.batch_size)
        model = Model(hparams, iterator)

    return TrainTuple(graph=graph, model=model, iterator=iterator)
