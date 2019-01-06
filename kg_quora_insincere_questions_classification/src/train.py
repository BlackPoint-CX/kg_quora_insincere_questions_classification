import os
from commons_utils import time_now
from config_utils import LOG_DIR, SUMMARY_DIR, MODEL_DIR
from model_helper import build_train_tuple, build_or_load_model
import tensorflow as tf
import logging

logging.basicConfig(filename=os.path.join(LOG_DIR, 'train.log'),
                    filemode='w+')


def train(hparams):
    train_tuple = build_train_tuple(hparams=hparams)

    train_sess = tf.Session()

    summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_DIR, 'train.summary'))

    with train_tuple.graph.as_default():
        train_model, global_step = build_or_load_model(train_tuple.model, hparams.model_dir)

    while global_step < hparams.num_train_steps:
        try:
            _, train_output_tuple = train_model.train(sess=train_sess)
        except tf.errors.OutOfRangeError:
            now = time_now()
            logging.info('Finished One Epoch at %s' % now)

            train_sess.run(train_tuple.iterator.initializer)
            continue

        train_summary, train_loss, global_step, batch_size, grad_norm = train_output_tuple
        summary_writer.add_summary(train_summary, global_step=global_step)

    train_model.saver.save(sess=train_sess,
                           save_path=os.path.join(MODEL_DIR, ''),
                           global_step=global_step)

    summary_writer.close()
