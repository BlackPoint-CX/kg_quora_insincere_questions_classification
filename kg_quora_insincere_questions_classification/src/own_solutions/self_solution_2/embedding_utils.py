from tqdm import tqdm
from config_utils import EMBEDDINGS_GLOVE_PATH, EMBEDDINGS_FASETEXT_PATH, EMBEDDINGS_PARA_PATH, MAX_FEATURES
from keras.callbacks import *

tqdm.pandas()

EMBEDDINGS_METHOD_LIST =  [1, 2, 3, 5, 6, 7, 9, 15, 16, 17, 19]
# EMBEDDINGS_METHOD_LIST =  [19]
# EMBEDDINGS_METHOD_LIST = [19]


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_glove(word_index, max_features=MAX_FEATURES):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDINGS_GLOVE_PATH) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_fasttext(word_index, max_features=MAX_FEATURES):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDINGS_FASETEXT_PATH) if
        len(o) > 100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index, max_features=MAX_FEATURES):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDINGS_PARA_PATH, encoding="utf8", errors='ignore') if
        len(o) > 100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_glove_new(word_idx, max_features=MAX_FEATURES):
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDINGS_GLOVE_PATH) if o.split(" ")[0] in word_idx)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features + 1, embed_size))
    for word, i in word_idx.items():
        # if i >= MAX_FEATURES:
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


EMBEDDINGS_GLOVE = None
EMBEDDINGS_FASTTEXT = None
EMBEDDINGS_PARA = None


def global_all_embeddings_init(word_index, max_features):
    global_load_glove(word_index, max_features)
    global_load_fasttext(word_index, max_features)
    global_load_para(word_index, max_features)


def global_load_glove(word_index, max_features):
    global EMBEDDINGS_GLOVE
    EMBEDDINGS_GLOVE = load_glove(word_index=word_index, max_features=max_features)


def global_load_fasttext(word_index, max_features):
    global EMBEDDINGS_FASTTEXT
    EMBEDDINGS_FASTTEXT = load_fasttext(word_index=word_index, max_features=max_features)


def global_load_para(word_index, max_features):
    global EMBEDDINGS_PARA
    EMBEDDINGS_PARA = load_para(word_index=word_index, max_features=max_features)


def load_embeddings_factory(embedding_method=1):
    assert embedding_method in EMBEDDINGS_METHOD_LIST
    global EMBEDDINGS_GLOVE, EMBEDDINGS_FASTTEXT, EMBEDDINGS_PARA
    # Single
    if embedding_method == 1:  # Only using glove
        return EMBEDDINGS_GLOVE
    elif embedding_method == 2:  # Only using fasttext
        return EMBEDDINGS_FASTTEXT
    elif embedding_method == 3:  # Only using para
        return EMBEDDINGS_PARA

    # Combination - mean
    elif embedding_method == 5:  # Using mean of glove and param
        return np.mean([EMBEDDINGS_GLOVE, EMBEDDINGS_PARA], axis=0)
    elif embedding_method == 6:  # Using mean of glove and fasttext
        return np.mean([EMBEDDINGS_GLOVE, EMBEDDINGS_FASTTEXT], axis=0)
    elif embedding_method == 7:  # Using mean of fasttext and param
        return np.mean([EMBEDDINGS_FASTTEXT, EMBEDDINGS_PARA], axis=0)

    elif embedding_method == 9:  # Using mean of fasttext, param and glove
        return np.mean([EMBEDDINGS_FASTTEXT, EMBEDDINGS_PARA, EMBEDDINGS_GLOVE], axis=0)

    # Combination - concat
    elif embedding_method == 15:  # Using concat of glove and param
        return np.concatenate((EMBEDDINGS_GLOVE, EMBEDDINGS_PARA), axis=1)
    elif embedding_method == 16:  # Using concat of of glove and fasttext
        return np.concatenate((EMBEDDINGS_GLOVE, EMBEDDINGS_FASTTEXT), axis=1)
    elif embedding_method == 17:  # Using concat of fasttext and param
        return np.concatenate((EMBEDDINGS_FASTTEXT, EMBEDDINGS_PARA), axis=1)

    elif embedding_method == 19:  # Using mean of fasttext, param and glove
        return np.concatenate((EMBEDDINGS_FASTTEXT, EMBEDDINGS_PARA, EMBEDDINGS_GLOVE), axis=1)
