import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
import os

from config_utils import SOURCE_DIR
from iterator_utils import build_train_iterator


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


train_file = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/data/source/train_ori.csv'
input_dataset = CsvDataset(filenames=train_file, record_defaults=[tf.string, tf.string, tf.int32], header=True)
input_dataset = input_dataset.map(map_func=lambda idx, question_text, target: (idx, tf.string_split([question_text]).values, target))
iterator = input_dataset.make_initializable_iterator()
(idx, question_text, label) = (iterator.get_next())

sess = tf.InteractiveSession()
sess.run(iterator.initializer)
(idx_val, question_text_val, label_val) = sess.run([idx, question_text, label])
idx_val, question_text_val, label_val
