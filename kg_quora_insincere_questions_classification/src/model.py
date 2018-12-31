import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

import model_helper
import vocab_utils
from collections import namedtuple


class TrainOutputTuple(
    namedtuple('TrainOutputTuple', ['train_summary', 'train_loss', 'global_step', 'batch_size', 'grad_norm'])):
    pass


class Model(object):
    def __init__(self, hparams, iterator):
        self.hparams = hparams
        self.iterator = iterator
        self.init()
        self.build_graph_forward()
        self.build_graph_backward()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.hparams.max_to_keep)

    def init(self):
        self.global_step = tf.Variable(initial_value=0, trainable=False)

    def build_embedding_encoder(self):
        with tf.variable_scope('encoder_embedding') as scope:
            self.encoder_embedding = vocab_utils.build_or_load_embedding(name='embedding_encoder',
                                                                         vocab_file=self.hparams.vocab_file,
                                                                         embedding_file=self.hparams.embedding_file,
                                                                         vocab_size=self.hparams.vocab_size,
                                                                         embedding_dim=self.hparams.embedding_dim
                                                                         )

    def build_encoder_cell(self):
        self.encoder_cell = model_helper.build_cell_list(num_layers=self.hparams.num_layers,
                                                         cell_type=self.hparams.cell_type,
                                                         learning_rate=self.hparams.learning_rate,
                                                         num_units=self.hparams.num_units,
                                                         dropout=self.hparams.dropout,
                                                         device=None)

    def build_encoder(self, sequence, sequence_length):
        if self.hparams.time_major:
            sequence = tf.transpose(sequence)

        with tf.variable_scope('encoder') as scope:
            self.encoder_embedding_input = tf.nn.embedding_lookup(self.encoder_embedding, sequence)

            if self.hparams.direction_type == 'uni':
                encoder_outputs, encoder_state = dynamic_rnn(cell=self.encoder_cell,
                                                             inputs=sequence,
                                                             sequence_length=sequence_length,
                                                             time_major=self.hparams.time_major)

            # elif self.hparams.direction_type == 'bi':
            # TODO Bi-directional
            # pass
            else:
                raise ValueError('Unknown direction_type : %s' % self.hparams.direction_type)

        return encoder_outputs, encoder_state

    def build_dense_layer(self):
        self.output_layer = tf.layers.Dense(units=self.hparams.num_labels)

    def build_logits(self):
        logits = self.output_layer(inputs=self.encoder_outputs)
        predicts = tf.argmax(logits, 1)
        return logits, predicts

    def build_loss(self):
        cross_ent = sparse_softmax_cross_entropy_with_logits(self.logits, self.iterator.label)
        loss = tf.reduce_mean(cross_ent)
        return loss

    def build_graph_forward(self):
        with tf.variable_scope('build_graph_forward') as scope:
            self.encoder_outputs, self.encoder_state = self.build_encoder(sequence=self.iterator.sequence,
                                                                          sequence_length=self.iterator.sequence_length)

            self.logits, self.predicts = self.build_logits()
            if self.hparams.mode == ModeKeys.TRAIN:

                self.train_loss = self.build_loss()

            elif self.hparams.mode == ModeKeys.EVAL:

                self.valid_losss = self.build_loss()

            elif self.hparams.mode == ModeKeys.INFER:
                pass
            else:
                raise ValueError('Unknown mode : %s' % self.hparams.mode)

    def build_optimizer(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = GradientDescentOptimizer(self.learning_rate)
        elif self.hparams.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError('Unknown optimizer : %s' % self.hparams.optimizer)
        return optimizer

    def build_train_summary(self):
        train_summary = tf.summary.merge([tf.summary.scalar('learning_rate', self.learning_rate),
                                          tf.summary.scalar('train_loss',
                                                            self.train_loss)] + self.gradient_norm_summary)
        return train_summary

    def build_graph_backward(self):
        with tf.variable_scope('build_graph_backward') as scope:
            if self.hparams.mode == ModeKeys.TRAIN:
                params = tf.trainable_variables()
                self.learning_rate = tf.constant(self.hparams.learning_rate)
                # TODO : 实现不同的decay策略.
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                global_step=self.global_step,
                                                                decay_steps=self.hparams.decay_steps,
                                                                decay_rate=self.hparams.decay_rate,
                                                                staircase=False)

                gradients = tf.gradients(ys=self.iterator.label, xs=params)
                self.clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(t_list=gradients,
                                                                                    clip_norm=self.hparams.clip_norm)
                self.gradient_norm_summary = [tf.summary.scalar('grad_norm', self.gradient_norm),
                                              tf.summary.scalar('clipped_gradient',
                                                                tf.global_norm(self.clipped_gradients))]

                self.optimizer = self.build_optimizer()
                self.update = self.optimizer.apply_gradients(grads_and_vars=zip(self.clipped_gradients, params),
                                                             global_step=self.global_step)

                self.train_summary = self.build_train_summary()

            elif self.hparams.mode in [ModeKeys.EVAL, ModeKeys.INFER]:

                pass
            else:
                raise ValueError('Unknown mode : %s' % self.hparams.mode)

    def train(self, sess):
        assert self.hparams.mode == ModeKeys.TRAIN
        train_output_tuple = TrainOutputTuple(train_summary=self.train_summary,
                                              train_loss=self.train_loss,
                                              global_step=self.global_step,
                                              batch_size=self.hparams.batch_size,
                                              grad_norm=self.gradient_norm)
        return sess.run([self.update, train_output_tuple])
