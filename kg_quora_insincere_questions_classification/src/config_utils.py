import os
import platform

system = platform.system()

if system == 'Linux':
    REPO_DIR = '/home/bp/PycharmProjects/'
elif system == 'Darwin':
    REPO_DIR = '/Users/alfredchen/Develop/PyRepos/'
else:
    raise ValueError('Unknown System : %s' % system)

PROJECT_DIR = os.path.join(REPO_DIR,
                           'kg_quora_insincere_questions_classification/kg_quora_insincere_questions_classification')

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
LOG_DIR = os.path.join(PROJECT_DIR,'log')
MODEL_DIR = os.path.join(PROJECT_DIR,'model')


SOURCE_DIR = os.path.join(DATA_DIR, 'source')
EMBEDDING_DIR = os.path.join(DATA_DIR, 'embedding')
PREDICTION_DIR = os.path.join(DATA_DIR, 'prediction')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')

TRAIN_FILE_PATH = os.path.join(SOURCE_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(SOURCE_DIR, 'test.csv')
