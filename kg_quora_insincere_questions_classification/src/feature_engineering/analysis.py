from project_config import SOURCE_DIR
import os
import pandas as pd



def analysis(train_file_path,test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)




    pass


def main():

    source_train_file_path = os.path.join(SOURCE_DIR,'train.csv')
    source_test_file_path = os.path.join(SOURCE_DIR,'test.csv')

    analysis(source_train_file_path,source_test_file_path)

if __name__ == '__main__':
    main()
