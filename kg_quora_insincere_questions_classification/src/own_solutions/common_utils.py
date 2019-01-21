import os
import platform
import sys

if platform.system() == 'Darwin':  # OS
    PROJECT_DIR = '/Users/alfredchen/Develop/PyRepos/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification'
elif platform.system() == 'Linux':
    PROJECT_DIR = '/home/bp/PycharmProjects/kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification'
else:
    raise ValueError('Wrong Project Directory')

SRC_DIR = os.path.join(PROJECT_DIR, 'src')
INPUT_DIR = os.path.join(SRC_DIR, 'input')
TRAIN_FILE_PATH = os.path.join(INPUT_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(INPUT_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE_PATH = os.path.join(INPUT_DIR, 'sample_submission.csv')

MAX_FEATURES = 10000
MAX_LEN = 100
EMBED_DIM = 300
