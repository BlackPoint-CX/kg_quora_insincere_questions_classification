import os
import sys
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.training import HParams

import train
from config_utils import LOG_DIR, MODEL_DIR, EMBEDDING_DIR, SOURCE_DIR

FLAGS = None


def build_argparser(argument_parser):
    argument_parser.register('type', 'bool', lambda v: v.lower() == 'true')
    argument_parser.add_argument('--vocab_size', type=int, default=30000, help='size of vocabulary.')
    argument_parser.add_argument('--embedding_dim', type=int, default=300, help='Di')
    argument_parser.add_argument('--embedding_trainable', type='bool', nargs='?', const=True, help='')
    argument_parser.add_argument('--embedding_file', type=str,
                                 default=os.path.join(EMBEDDING_DIR, 'embeddings.txt'))
    argument_parser.add_argument('--max_to_keep', type=int, default=5, help='max number of kept models.')
    argument_parser.add_argument('--vocab_file', type=str, default=os.path.join(SOURCE_DIR, 'vocab.txt'))
    argument_parser.add_argument('--train_file', type=str, default=os.path.join(SOURCE_DIR, 'train_clean.csv'))
    argument_parser.add_argument('--test_file', type=str, default=os.path.join(SOURCE_DIR, 'test_clean.csv'))
    argument_parser.add_argument('--num_layers', type=int, default=1, help='number of layers.')
    argument_parser.add_argument('--cell_type', type=str, default='lstm', help='type of rnn cell',
                                 choices=['lstm', 'gru'])
    argument_parser.add_argument('--learning_rate', type=int, default=0.001, help='learning rate.')
    argument_parser.add_argument('--num_units', type=int, default=128, help='num of hidden units.')
    argument_parser.add_argument('--dropout', type=int, default=0.3, help='dropout rate.')
    argument_parser.add_argument('--time_major', type='bool', nargs='?', const=False, help='')
    argument_parser.add_argument('--direction_type', type=str, default='uni', help='direction of encoder.',
                                 choices=['uni', 'bi'])
    argument_parser.add_argument('--num_labels', type=int, default=2, help='num of labels.')
    argument_parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer.',
                                 choices=['sgd', 'adam'])
    argument_parser.add_argument('--mode', type=str, default='train', help='choose mode in [train | eval | infer]',
                                 choices=['train', 'infer', 'eval'])
    argument_parser.add_argument('--decay_steps', type=int, default=1000,
                                 help='Begin decay learning rate after # num of steps ')
    argument_parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    argument_parser.add_argument('--clip_norm', type=float, default=5.0, help='clip norm')
    argument_parser.add_argument('--batch_size', type=int, default=64, help='num of batch size.')
    argument_parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='directory of logs.')
    argument_parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='directory of models.')
    argument_parser.add_argument('--num_train_steps', type=int, default=10000, help='num of train steps.')


def build_hparams(flags):
    return HParams(
        vocab_size=flags.vocab_size,
        embedding_dim=flags.embedding_dim,
        embedding_trainable=flags.embedding_trainable,
        embedding_file=flags.embedding_file,
        max_to_keep=flags.max_to_keep,
        vocab_file=flags.vocab_file,
        train_file=flags.train_file,
        test_file=flags.test_file,
        num_layers=flags.num_layers,
        cell_type=flags.cell_type,
        learning_rate=flags.learning_rate,
        num_units=flags.num_units,
        dropout=flags.dropout,
        time_major=flags.time_major,
        direction_type=flags.direction_type,
        num_labels=flags.num_labels,
        optimizer=flags.optimizer,
        mode=flags.mode,
        decay_steps=flags.decay_steps,
        decay_rate=flags.decay_rate,
        clip_norm=flags.clip_norm,
        batch_size=flags.batch_size,
        log_dir=flags.log_dir,
        model_dir=flags.model_dir,
        num_train_steps=flags.num_train_steps
    )


def run_main(default_hparams, train_fn):
    train_fn(default_hparams)


def main(unused_argv):
    default_hparams = build_hparams(FLAGS)
    train_fn = train.train(hparams=default_hparams)
    run_main(default_hparams, train_fn)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    build_argparser(argument_parser)
    FLAGS, unparsed = argument_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
