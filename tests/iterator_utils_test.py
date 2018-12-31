import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
import os

from config_utils import SOURCE_DIR
from iterator_utils import build_train_iterator


class IteratorUtilsTest(tf.test.TestCase):
    def test_build_iterator(self):
        input_dataset = CsvDataset(filenames = os.path.join(SOURCE_DIR,'train.csv'),record_defaults=[['NULL'],[''],[0]])
        vocab_table = None
        batch_size = 64
        buffer_size = None
        num_parallel_calls = 4

        iterator = build_train_iterator(input_dataset, vocab_table, batch_size, buffer_size, num_parallel_calls)


        with self.test_session() as sess:
            sess.run(tf.tables_initializer())




