import tensorflow as tf
from collections import namedtuple


class BatchedInput(namedtuple('BatchedInput', ['initializer', 'idx', 'sequence', 'sequence_length', 'label'])):
    pass


def build_train_iterator(train_dataset, vocab_table, batch_size, buffer_size=None, num_parallel_calls=4):
    """
    Build iterator for batch input from input dataset(for train file).
    :param train_dataset:
    :param vocab_table:
    :param batch_size:
    :param buffer_size:
    :param num_parallel_calls:
    :return:
    """
    if not buffer_size:
        buffer_size = batch_size * 1000

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

    train_dataset = train_dataset.map(
        map_func=lambda idx, question_text, label: (
            idx, tf.string_split([question_text]).values, label),
        num_parallel_calls=num_parallel_calls)

    train_dataset = train_dataset.map(
        map_func=lambda idx, question_text, label: (idx, tf.cast(vocab_table.lookup(question_text), tf.int32), label),
        num_parallel_calls=num_parallel_calls)

    train_dataset = train_dataset.map(map_func=lambda idx, sequence, label: (idx, sequence, tf.size(sequence), label))

    batched_input_dataset = train_dataset.padded_batch(batch_size=batch_size,
                                                       padded_shapes=(
                                                           tf.TensorShape([]),
                                                           tf.TensorShape([None]),
                                                           tf.TensorShape([]),
                                                           tf.TensorShape([])
                                                       ),
                                                       padding_values=(
                                                           '',
                                                           0,
                                                           0,
                                                           0
                                                       ))

    iterator = batched_input_dataset.make_initializable_iterator()
    (idx, sequence, sequence_length, label) = (iterator.get_next())
    return BatchedInput(initializer=iterator.initializer,
                        idx=idx,
                        sequence=sequence,
                        sequence_length=sequence_length,
                        label=label
                        )


def build_test_iterator(test_dataset, vocab_table, batch_size, num_parallel_calls=4):
    """
    Build iterator for batch input from input dataset(for train file).
    :param test_dataset:
    :param vocab_table:
    :param batch_size:
    :param num_parallel_calls:
    :return:
    """
    test_dataset = test_dataset.map(
        map_func=lambda idx, question_text: (tf.cast(idx, tf.string), tf.string_split(question_text).values),
        num_parallel_calls=num_parallel_calls)

    test_dataset = test_dataset.map(
        map_func=lambda idx, question_text: (idx, tf.cast(vocab_table.lookup(question_text), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    test_dataset = test_dataset.map(map_func=lambda idx, sequence: (idx, sequence, tf.size(sequence)),
                                    num_parallel_calls=num_parallel_calls)

    batched_test_dataset = test_dataset.padded_batch(batch_size=batch_size,
                                                     padded_shapes=(
                                                         tf.TensorShape([]),
                                                         tf.TensorShape([None]),
                                                         tf.TensorShape([])
                                                     ),
                                                     padding_values=(
                                                         '',
                                                         '',
                                                         0
                                                     ))

    iterator = batched_test_dataset.make_initializable_iterator()
    (idx, sequence, sequence_length) = (iterator.get_next())
    return BatchedInput(initializer=iterator.initializer,
                        idx=idx,
                        sequence=sequence,
                        sequence_length=sequence_length,
                        label=None
                        )
