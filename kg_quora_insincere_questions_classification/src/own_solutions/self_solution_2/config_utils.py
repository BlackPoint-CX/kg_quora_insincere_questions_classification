import os
import platform
import sys

if platform.system() == 'Darwin':  # OS
    PROJECT_DIR = '/Users/alfredchen/Develop/PyRepos/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification'
elif platform.system() == 'Linux':
    PROJECT_DIR = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification'
else:
    raise ValueError('Wrong Project Directory')

LOG_DIR = os.path.join(PROJECT_DIR, 'log')
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
INPUT_DIR = os.path.join(SRC_DIR, 'input')
TRAIN_FILE_PATH = os.path.join(INPUT_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(INPUT_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE_PATH = os.path.join(INPUT_DIR, 'sample_submission.csv')

EMBEDDINGS_DIR = os.path.join(INPUT_DIR,'embeddings')
EMBEDDINGS_GLOVE_PATH = os.path.join(EMBEDDINGS_DIR,'glove.840B.300d/glove.840B.300d.txt')
EMBEDDINGS_FASETEXT_PATH = os.path.join(EMBEDDINGS_DIR,'wiki-news-300d-1M/wiki-news-300d-1M.vec')
EMBEDDINGS_PARA_PATH = os.path.join(EMBEDDINGS_DIR,'paragram_300_sl999/paragram_300_sl999.txt')

MAX_FEATURES = 10000
MAX_LEN = 100
EMBED_DIM = 300
