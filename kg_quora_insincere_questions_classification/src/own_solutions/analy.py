import os
from collections import defaultdict, Counter

import pandas as pd

from common_utils import TRAIN_FILE_PATH, TEST_FILE_PATH, INPUT_DIR

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df['question_text'] = test_df['question_text'].str.lower()


    def clean_question_text(row):
        question_text = row['question_text']
        word_list = question_text.split(' ')
        clean_word_list = []
        for ele in word_list:
            clean_word_list.append(clean_word(ele))
        clean_question_text = ' '.join(clean_word_list)
        row['question_text'] = clean_question_text
        return row

    train_df = train_df.apply(lambda row: clean_question_text(row),axis=1)
    test_df = test_df.apply(lambda row: clean_question_text(row),axis=1)

    train_df.to_csv(os.path.join(INPUT_DIR,'train.csv'),index=False)
    test_df.to_csv(os.path.join(INPUT_DIR,'test.csv'),index=False)

    train_df.shape  # (1306122, 3)
    test_df.shape  # (56370, 2)

    all_df = pd.concat([train_df['question_text'], test_df['question_text']], axis=0)
    all_df.shape  # (1362492,)
    split_ser = all_df.apply(lambda text: text.split(' '))

    alpha_word_count = defaultdict(int)
    no_alpha_word_count = defaultdict(int)

    for word_list in split_ser:
        for word in word_list:
            if word.isalpha():
                alpha_word_count[word] += 1
            else:
                no_alpha_word_count[word] += 1

    len(alpha_word_count)  # 145435 # 178469
    # pd.Series(list(alpha_word_count.values())).describe()
    # count
    # 178469.000000
    # mean
    # 95.234052
    # std
    # 3637.738980
    # min
    # 1.000000
    # 25 % 1.000000
    # 50 % 1.000000
    # 75 % 5.000000
    # max
    # 694302.000000
    # dtype: float64
    len(no_alpha_word_count)  # 317241 # 89785
    # pd.Series(list(no_alpha_word_count.values())).describe()
    # count
    # 89785.000000
    # mean
    # 5.269911
    # std
    # 121.723258
    # min
    # 1.000000
    # 25 % 1.000000
    # 50 % 1.000000
    # 75 % 1.000000
    # max
    # 16856.000000
    # dtype: float64


    # alpha_word_count = {t[0]: t[1] for t in sorted(alpha_word_count.items(), key=lambda t: t[1], reverse=True)}
    # no_alpha_word_count = {t[0]: t[1] for t in sorted(no_alpha_word_count.items(), key=lambda t: t[1], reverse=True)}

    import re
    from functools import partial

    question_mark_start = set()  # ?swing
    question_mark_end = set()  # swings?
    slice_question_mark_end = set()  # swing'?
    slice_end = set()  # swing'
    full_stop_start = set()  # .end
    full_stop_end = set()  # end.
    full_stop_question_mark_end = set()  # end.?
    comma_start = set()  # ,here
    comma_end = set()  # here,
    exclamatory_mark_start = set()  # !You
    exclamatory_mark_end = set()  # You!
    pure_number = set()  # 123123
    whose_end = set()  # lucifer's
    whose_question_mark_end = set()  # lucifer's?
    special_whose_end = set()  # melkor’s
    colon_start = set()  # :temp
    colon_end = set()  # temp:
    double_quotation_mask_start = set()  # "who
    double_quotation_mask_end = set()  # who"
    double_quotation_mask_question_mask_end = set()  # who"?
    double_quotation_mask_start_end = set()  # "who"
    semicolon_start = set()  # ;who
    semicolon_end = set()  # who;
    hyphen_mid = set()  # three-dollar
    hyphen_mid_question_mark_end = set()
    slash_mid = set()  # three/dollar
    left_bracket_start = set()  # (three
    left_bracket_end = set()  # three(
    right_bracket_start = set()  # )three
    right_bracket_end = set()  # three)
    right_bracket_question_mark_end = set()  # three)?
    bracket_start_end = set()  # (three)
    left_square_bracket_start = set()  # [three
    left_square_bracket_end = set()  # three[
    right_square_bracket_start = set()  # ]three
    right_square_bracket_end = set()  # three]
    square_bracket_start_end = set()  # [three]
    underline_mid = set()  # there_is
    full_stop_mid = set()  # there.is
    contain_numeric = set()

    remain = set()


    def check_symbol_mid(text, symbol):
        l = text.split(symbol)
        if len(l) == 2 and l[0].isalpha() and l[1].isalpha():
            return True
        else:
            return False


    def check_contain_numeric(text):
        for single_char in list(text):
            if single_char.isnumeric():
                return True
            else:
                pass

        return False


    check_hyphen_mid = partial(check_symbol_mid, symbol='-')
    check_slash_mid = partial(check_symbol_mid, symbol='/')
    check_underline_mid = partial(check_symbol_mid, symbol='_')
    check_full_stop_mid = partial(check_symbol_mid, symbol='.')


    def rmv_uncessary_mark(seq):
        """
        Used for removing uncessary mark at beginning and end.
        :return:
        """

        def _rmv_uncessary_mark(seq_list):
            start_idx = 0
            for idx, c in enumerate(seq_list):
                if not c.isalpha():
                    pass
                else:
                    start_idx = idx
                    break
            seq_list = seq_list[start_idx:]
            return seq_list

        seq = ''.join(_rmv_uncessary_mark(list(seq)))
        seq_reverse = list(seq)
        seq_reverse.reverse()
        seq_reverse = _rmv_uncessary_mark(seq_reverse)
        seq_reverse.reverse()
        seq_result = ''.join(seq_reverse)
        return seq_result


    def clean_word(word, nest_flag=False):

        if nest_flag:
            word = rmv_uncessary_mark(word)

        if word.startswith('?') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('?') and word[:-1].isalpha():
            result = word[:-1]
        elif word.endswith("'?") and word[:-2].isalpha():
            result = word[:-2]
        elif word.endswith("'") and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith('.') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('.') and word[:-1].isalpha():
            result = word[:-1]
        elif word.endswith('.?') and word[:-2].isalpha():
            result = word[:-2]
        elif word.startswith(',') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith(',') and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith('!') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('!') and word[:-1].isalpha():
            result = word[:-1]
        elif word.isnumeric():
            result = word
        elif word.endswith('\'s') and word[:-2].isalpha():
            result = word
        elif word.endswith('\'s?') and word[:-3].isalpha():
            result = word[:-2]
        elif word.endswith('’s') and word[:-2].isalpha():
            result = word[:-2] + "'s"
        elif word.startswith(':') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith(':') and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith('"') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('"') and word[:-1].isalpha():
            result = word[:-1]
        elif word.endswith('"?') and word[:-2].isalpha():
            result = word[:-2]
        elif word.startswith('"') and word.endswith('"') and word[1:-1].isalpha():
            result = word[1:-1]
        elif word.startswith(';') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith(';') and word[:-1].isalpha():
            result = word[:-1]
        elif check_hyphen_mid(word):
            result = word
        elif word.endswith('?') and check_hyphen_mid(word[:-1]):
            result = word[:-1]
        elif check_slash_mid(word):
            word_list = word.split('/')
            result = ' '.join(word_list)
        elif word.startswith('(') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('(') and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith(')') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith(')') and word[:-1].isalpha():
            result = word[:-1]
        elif word.endswith(')?') and word[:-2].isalpha():
            result = word[:-2]
        elif word.startswith('(') and word.endswith(')') and word[1:-1].isalpha():
            result = word[1:-1]
        elif word.startswith('[') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith('[') and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith(']') and word[1:].isalpha():
            result = word[1:]
        elif word.endswith(']') and word[:-1].isalpha():
            result = word[:-1]
        elif word.startswith(']') and word.endswith(']') and word[1:-1].isalpha():
            result = word[1:-1]
        elif check_underline_mid(word):
            result = word
        elif check_full_stop_mid(word):
            result = word
        elif check_contain_numeric(word):
            result = word
        else:
            if nest_flag:
                result = word
            else:
                nest_flag = True
                result = clean_word(word, nest_flag)

        return result

    # for k, v in no_alpha_word_count.items():
    #
    #     if k.startswith('?') and k[1:].isalpha():
    #         question_mark_start.add(k)
    #     elif k.endswith('?') and k[:-1].isalpha():
    #         question_mark_end.add(k)
    #     elif k.endswith("'?") and k[:-2].isalpha():
    #         slice_question_mark_end.add(k)
    #     elif k.endswith("'") and k[:-1].isalpha():
    #         slice_end.add(k)
    #     elif k.startswith('.') and k[1:].isalpha():
    #         full_stop_start.add(k)
    #     elif k.endswith('.') and k[:-1].isalpha():
    #         full_stop_end.add(k)
    #     elif k.endswith('.?') and k[:-2].isalpha():
    #         full_stop_question_mark_end.add(k)
    #     elif k.startswith(',') and k[1:].isalpha():
    #         comma_start.add(k)
    #     elif k.endswith(',') and k[:-1].isalpha():
    #         comma_end.add(k)
    #     elif k.startswith('!') and k[1:].isalpha():
    #         comma_start.add(k)
    #     elif k.endswith('!') and k[:-1].isalpha():
    #         comma_end.add(k)
    #     elif k.isnumeric():
    #         pure_number.add(k)
    #     elif k.endswith('\'s') and k[:-2].isalpha():
    #         whose_end.add(k)
    #     elif k.endswith('\'s?') and k[:-3].isalpha():
    #         whose_question_mark_end.add(k)
    #     elif k.endswith('’s') and k[:-2].isalpha():
    #         special_whose_end.add(k)
    #     elif k.startswith(':') and k[1:].isalpha():
    #         colon_start.add(k)
    #     elif k.endswith(':') and k[:-1].isalpha():
    #         colon_end.add(k)
    #     elif k.startswith('"') and k[1:].isalpha():
    #         double_quotation_mask_start.add(k)
    #     elif k.endswith('"') and k[:-1].isalpha():
    #         double_quotation_mask_end.add(k)
    #     elif k.endswith('"?') and k[:-2].isalpha():
    #         double_quotation_mask_question_mask_end.add(k)
    #     elif k.startswith('"') and k.endswith('"') and k[1:-1].isalpha():
    #         double_quotation_mask_start_end.add(k)
    #     elif k.startswith(';') and k[1:].isalpha():
    #         semicolon_start.add(k)
    #     elif k.endswith(';') and k[:-1].isalpha():
    #         semicolon_end.add(k)
    #     elif check_hyphen_mid(k):
    #         hyphen_mid.add(k)
    #     elif k.endswith('?') and check_hyphen_mid(k[:-1]):
    #         hyphen_mid_question_mark_end.add(k)
    #     elif check_slash_mid(k):
    #         slash_mid.add(k)
    #     elif k.startswith('(') and k[1:].isalpha():
    #         left_bracket_start.add(k)
    #     elif k.endswith('(') and k[:-1].isalpha():
    #         left_bracket_end.add(k)
    #     elif k.startswith(')') and k[1:].isalpha():
    #         right_bracket_start.add(k)
    #     elif k.endswith(')') and k[:-1].isalpha():
    #         right_bracket_end.add(k)
    #     elif k.endswith(')?') and k[:-2].isalpha():
    #         right_bracket_question_mark_end.add(k)
    #     elif k.startswith('(') and k.endswith(')') and k[1:-1].isalpha():
    #         bracket_start_end.add(k)
    #     elif k.startswith('[') and k[1:].isalpha():
    #         left_square_bracket_start.add(k)
    #     elif k.endswith('[') and k[:-1].isalpha():
    #         left_square_bracket_end.add(k)
    #     elif k.startswith(']') and k[1:].isalpha():
    #         right_square_bracket_start.add(k)
    #     elif k.endswith(']') and k[:-1].isalpha():
    #         right_square_bracket_end.add(k)
    #     elif k.startswith(']') and k.endswith(']') and k[1:-1].isalpha():
    #         square_bracket_start_end.add(k)
    #     elif check_underline_mid(k):
    #         underline_mid.add(k)
    #     elif check_full_stop_mid(k):
    #         full_stop_mid.add(k)
    #     elif check_contain_numeric(k):
    #         contain_numeric.add(k)
    #
    #     else:
    #         remain.add(k)
    #
    # print(len(question_mark_start))  # 5
    # print(len(question_mark_end))  # 72946
    # print(len(slice_question_mark_end))  # 976
    # print(len(slice_end))  # 2761
    # print(len(full_stop_start))  # 254
    # print(len(full_stop_end))  # 13627
    # print(len(full_stop_question_mark_end))  # 1811
    # print(len(comma_start))  # 85
    # print(len(comma_end))  # 31779
    # print(len(exclamatory_mark_start))  # 0
    # print(len(exclamatory_mark_end))  # 0
    # print(len(pure_number))  # 2759
    # print(len(whose_end))  # 9883
    # print(len(whose_question_mark_end))  # 512
    # print(len(special_whose_end))  # 1553
    # print(len(colon_start))  # 9
    # print(len(colon_end))  # 2501
    # print(len(double_quotation_mask_start))  # 6009
    # print(len(double_quotation_mask_end))  # 5639
    # print(len(double_quotation_mask_question_mask_end))  # 3597
    # print(len(double_quotation_mask_start_end))  # 5473
    # print(len(semicolon_start))  # 2
    # print(len(semicolon_end))  # 886
    # print(len(hyphen_mid))  # 15337
    # print(len(hyphen_mid_question_mark_end))  # 3297
    # print(len(slash_mid))  # 14694
    # print(len(left_bracket_start))  # 5228
    # print(len(left_bracket_end))  # 7
    # print(len(right_bracket_start))  # 1
    # print(len(right_bracket_end))  # 6744
    # print(len(right_bracket_question_mark_end))  # 4936
    # print(len(bracket_start_end))  # 4474
    # print(len(left_square_bracket_start))  # 123
    # print(len(left_square_bracket_end))  # 0
    # print(len(right_square_bracket_start))  # 1
    # print(len(right_square_bracket_end))  # 88
    # print(len(square_bracket_start_end))  # 0
    # print(len(underline_mid))  # 70
    # print(len(full_stop_mid))  # 2905
    # print(len(contain_numeric))  # 40962
    # print(len(remain))  # 55307
    #
    # list(remain)[-1000:]
    #
    # temp_list = [ele[:2] for ele in remain if ele.endswith(')?')]
    # c = Counter([ele[:2] for ele in list(remain)])
    # c.most_common()
