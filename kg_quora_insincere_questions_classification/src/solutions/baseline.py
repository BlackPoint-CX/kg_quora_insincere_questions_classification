import os
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import re
from project_config import TEST_FILE_PATH, TRAIN_FILE_PATH, PREDICTION_DIR, FEATURES_DIR
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from nltk.stem import WordNetLemmatizer


# train_df.shape -> (1306122, 3)
# test_df.shape -> (56370, 2)
# col-target distribution
# 0    1225312
# 1      80810

def _step_cleaning(df):
    lemmatizer = WordNetLemmatizer()

    def _preprocessing(row):
        question_text = row['question_text']

        # Step 0 : Make lower
        question_text = question_text.lower()
        # Step 1 : Remove unnecessary symbols
        question_text = re.sub('[^a-zA-z \']', ' ', question_text)

        tokens = [token for token in question_text.split(' ') if token]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        question_text = ' '.join(tokens)

        row['f_qes_txt'] = question_text
        return row

    return df.apply(lambda row: _preprocessing(row), axis=1)


def build_tfidf_vectoizer(input_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=3000)
    tfidf_vectorizer.fit(input_df['f_qes_txt'])
    return tfidf_vectorizer


def reload_or_generate_df(file_path, func=None, input_df=None):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        result_df = func(input_df)
        result_df.to_csv(file_path, index=False)
        return result_df


def baseline(train_df, test_df):
    union_ques_text_df = pd.concat([train_df['question_text'], test_df['question_text']]).to_frame()

    # Preprocessing
    print('Preprocessing')
    step_cleaning_df_path = os.path.join(FEATURES_DIR, 'step_cleaning.csv')
    step_cleaning_df = reload_or_generate_df(step_cleaning_df_path, step_cleaning, union_ques_text_df)
    # step_cleaning_df = step_cleaning(union_ques_text_df)

    # Building tf-idf vectorizer
    print('Building tf-idf vectorizer')
    tfidf_vectorizer = build_tfidf_vectoizer(step_cleaning_df)

    # Building features
    print('Building features')
    train_df = _step_cleaning(train_df)
    f_tfidf_train = tfidf_vectorizer.transform(train_df['f_qes_txt'])
    t_train = train_df['target']

    # Building models
    print('Building models')
    svc = SVC()

    # Split train and valid dataset
    print('Split train and valid dataset')
    X_train, X_val, y_train, y_val = train_test_split(f_tfidf_train, t_train)

    # Fit classify model
    print('Fit classify model')
    svc.fit(X=X_train, y=y_train)

    # Evaluation on validate dataset
    print('Evaluation on validate dataset')
    y_val_pred = svc.predict(X_val)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    print('Precision : {}'.format(precision))
    print('Recall : {}'.format(recall))

    # Prediction on test
    f_tfidf_test = tfidf_vectorizer.transform(test_df['f_qes_txt'])
    test_df['target'] = svc.predict(f_tfidf_test)
    test_df.to_csv(os.path.join(PREDICTION_DIR, 'test_df.csv'), index=False)


def p1(train_df, test_df):
    train_df = _step_cleaning(train_df)
    train_df.to_csv(os.path.join(FEATURES_DIR, 'cleaning_train_df.csv'), index=False)
    # test_df = _step_cleaning(test_df)


if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)
    # baseline(train_df, test_df)
    p1(train_df, test_df)
