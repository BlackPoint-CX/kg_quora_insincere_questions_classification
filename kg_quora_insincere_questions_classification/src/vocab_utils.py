import pandas as pd
from collections import defaultdict
import numpy as np
from keras_preprocessing.text import Tokenizer

from common_utils import EMBED_DIM, TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding_utils import get_coefs
from preprocessing_utils import replace_typical_misspell, clean_text, clean_numbers


def build_word_count(train_df, test_df):
    union_ser = pd.concat([train_df['question_text'], test_df['question_text']], axis=0)
    split_ser = union_ser.apply(lambda text: text.split(' '))

    word_count = defaultdict(int)

    for word_list in split_ser:
        for word in word_list:
            word_count[word] += 1

    word_count = {t[0]: t[1] for t in sorted(word_count.items(), key=lambda t: t[1], reverse=True)}

    return word_count


def build_vocab(word_count, embedding_file_path, vocab_file_path):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(embedding_file_path) if len(o) > 100 and o.split(" ")[0] in word_count)

    vocab_list = list(embeddings_index.keys())
    with open(vocab_file_path, 'w+') as w_file:
        for word in vocab_list:
            w_file.write('%s\n' % word)


def load_vocab(vocab_file_path):
    word_idx_table = defaultdict(int)
    with open(vocab_file_path, 'r') as r_file:
        for word in r_file:
            word = word.strip()
            word_idx_table[word] = len(word_idx_table) + 1

    idx_word_table = {v:k for k,v in word_idx_table.items()}
    return word_idx_table, idx_word_table



if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df['question_text'] = test_df['question_text'].str.lower()

    train_df['question_text'] = train_df['question_text'].apply(lambda x: replace_typical_misspell(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: replace_typical_misspell(x))

    train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text(x))

    train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_numbers(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_numbers(x))

    word_count = build_word_count(train_df, test_df)
    embedding_file_path = '/Users/alfredchen/Develop/PyRepos/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/input/embeddings/glove.840B.300d.txt'
    vocab_file_path = '/Users/alfredchen/Develop/PyRepos/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification/input/embeddings/glove.vocab.txt'
    build_vocab(word_count, embedding_file_path, vocab_file_path)

    word_idx_table, idx_word_table = load_vocab(vocab_file_path)
