import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
import os

import iterator_utils
from config_utils import SOURCE_DIR, DATA_DIR
from iterator_utils import build_train_iterator
from vocab_utils import build_vocab_table


class IteratorUtilsTest(tf.test.TestCase):
    def test_build_iterator(self):
        input_dataset = CsvDataset(filenames=os.path.join(SOURCE_DIR, 'train_ori.csv'),
                                   record_defaults=[['NULL'], [''], [0]])
        vocab_table = None
        batch_size = 64
        buffer_size = None
        num_parallel_calls = 4

        iterator = build_train_iterator(input_dataset, vocab_table, batch_size, buffer_size, num_parallel_calls)

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())

    def test_build_train_iterator(self):
        vocab_file = os.path.join(DATA_DIR, 'vocab.txt')
        train_file = os.path.join(DATA_DIR, 'train.csv')
        batch_size = 64
        vocab_table = build_vocab_table(vocab_file)
        input_dataset = tf.data.experimental.CsvDataset(filenames=train_file,
                                                        record_defaults=[tf.string, tf.int32, tf.string],
                                                        header=True)
        iterator = iterator_utils.build_train_iterator(train_dataset=input_dataset,
                                                       vocab_table=vocab_table,
                                                       batch_size=batch_size)


train_file = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/data/source/train_ori.csv'
input_dataset = CsvDataset(filenames=train_file, record_defaults=[tf.string, tf.string, tf.int32], header=True)

input_dataset = input_dataset.map(
    map_func=lambda idx, question_text, target: (idx, tf.string_split([question_text]).values, target))
input_dataset = input_dataset.map(map_func=lambda idx, tokens, target: (idx, tokens, tf.size(tokens), target))

batched_input_dataset = input_dataset.padded_batch(batch_size=64,
                                                   padded_shapes=(
                                                       tf.TensorShape([]),
                                                       tf.TensorShape([None]),
                                                       tf.TensorShape([]),
                                                       tf.TensorShape([])
                                                   ),
                                                   padding_values=(
                                                       '',
                                                       '',
                                                       0,
                                                       0
                                                   ))

iterator = batched_input_dataset.make_initializable_iterator()
(idx, question_text, question_len, label) = (iterator.get_next())

sess = tf.InteractiveSession()
sess.run(iterator.initializer)
(idx_val, question_text_val, question_len, label_val) = sess.run([idx, question_text, question_len, label])
idx_val, question_text_val, label_val
