import os

import pandas as pd
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
import numpy as np

from config_utils import TRAIN_FILE_PATH, TEST_FILE_PATH, LOG_DIR
from embedding_utils import global_all_embeddings_init, EMBEDDINGS_METHOD_LIST, load_embeddings_factory
from model_utils import bmdl_0, bmdl_1
from tuple_utils import GridSearchResultTuple


def gs_func(train_tuple, X, y, n_splits=1, test_size=0.1, epochs=3):
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    scoring = make_scorer(f1_score)
    model = KerasClassifier(build_fn=train_tuple.build_fn, epochs=epochs, batch_size=64, verbose=2)
    gs = GridSearchCV(estimator=model, param_grid=train_tuple.param_grid, scoring=scoring, cv=cv)
    gs.fit(X=X, y=y)

    best_score = gs.best_score_
    best_params = gs.best_params_
    best_estimator = gs.best_estimator_
    gs_result = GridSearchResultTuple(best_score=best_score, best_estimator=best_estimator, best_params=best_params)

    return gs_result


def load_data_cv(max_features, max_len):
    # Load and preprocessing input
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    # Fill up the missing values
    train_X = train_df['question_text'].values
    test_X = test_df['question_text'].values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=max_len)
    test_X = pad_sequences(test_X, maxlen=max_len)

    # Get the target values
    train_y = train_df['target'].values

    return train_X, train_y, test_X, tokenizer.word_index


def load_data(max_features, max_len):
    # Load and preprocessing input
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)
    # Fill up the missing values
    train_X = train_df['question_text'].values
    val_X = val_df['question_text'].values
    test_X = test_df['question_text'].values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=max_len)
    val_X = pad_sequences(val_X, maxlen=max_len)
    test_X = pad_sequences(test_X, maxlen=max_len)

    # Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    return train_X, train_y, val_X, val_y, test_X, tokenizer.word_index


if __name__ == '__main__':

    seed = 2019
    np.random.seed(seed=seed)

    max_features_range = [100000,200000]
    max_len_range = [90]
    train_tuple_list = [bmdl_0]
    # gs_result_list = []
    history_list = []
    history_list_2 = []
    with open(os.path.join(LOG_DIR, 'history_log.txt'), 'w+') as w_file:

        for max_features in max_features_range:
            for max_len in max_len_range:
                # train_X, train_y, test_X, word_index = load_data_cv(max_features=max_features, max_len=max_len)
                train_X, train_y, val_X, val_y, test_X, word_index = load_data(max_features=max_features,
                                                                               max_len=max_len)

                global_all_embeddings_init(word_index=word_index, max_features=max_features)

                for embedding_method in EMBEDDINGS_METHOD_LIST:
                    embedding_matrix = load_embeddings_factory(embedding_method=embedding_method)
                    # embedding_dim = embedding_matrix.shape[1]  # load_embeddings_factory has different embedding_dim

                    for train_tuple_func in train_tuple_list:
                        train_tuple = train_tuple_func(max_len=max_len, max_features=max_features,
                                                       embedding_matrix=embedding_matrix)
                        # gs_result = gs_func(train_tuple=train_tuple, X=train_X, y=train_y)
                        #
                        # gs_result.append('{} <- {}'.format(gs_result.best_score, gs_result.best_params))
                        model = train_tuple.build_fn(max_len, max_features, embedding_matrix)

                        history = model.fit(x=train_X, y=train_y, batch_size=128, epochs=5, verbose=2)

                        # history_list.append(history)
                        print(history.history)
                        w_file.write('%s\n' % history.history)

    # with open(os.path.join(LOG_DIR, 'gs_result_log.txt'), 'w+') as w_file:
    #     for result in gs_result_list:
    #         w_file.write('{}\n'.format(result))
