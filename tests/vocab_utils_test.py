import os
import unittest
import tensorflow as tf
from config_utils import EMBEDDING_DIR
from vocab_utils import _load_pretrained_embedding


class VocabUtilsTest(unittest.TestCase):
    def test_load_embedding(self):
        embedding_file = os.path.join(EMBEDDING_DIR, 'glove.840B.300d.txt')
        num_words, embedding_dim, _ = _load_pretrained_embedding(embedding_file)
        self.assertEqual(num_words, 2196017)
        self.assertEqual(embedding_dim, 300)


class VocabUtilsTFTest(tf.test.TestCase):
    def test_build_or_load_embedding_case_0(self):
        pass

    def test_build_or_load_embedding_case_1(self):
        pass



def main():
    suite = unittest.TestSuite()
    suite.addTest(VocabUtilsTest('test_load_embedding'))


if __name__ == '__main__':
    unittest.main('main')
