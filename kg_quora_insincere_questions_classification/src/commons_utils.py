import datetime

QUESTION_EXAMPLE = """"What is~!@#$%^&*()_+}{|":<>?`[]\;',./' the book ""Nothing Succeeds Like Success"" about?"""""

VOCAB_SIZE_THRESHOLD_CPU = 50000
UNK_IND = 0


def time_now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
