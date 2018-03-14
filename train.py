# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午4:41
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import settings
import tensorflow as tf
import models
import dataset
import os

BATCH_SIZE = settings.BATCH_SIZE

x = tf.placeholder(tf.int32, [None, None])
y = tf.placeholder(tf.float32, [None, 1])
emb_keep = tf.placeholder(tf.float32)
rnn_keep = tf.placeholder(tf.float32)

model = models.Model(x, y, emb_keep, rnn_keep)


data = dataset.Dataset(0)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(settings.TRAIN_TIMES):
        x, y = data.next_batch(BATCH_SIZE)
        loss, _ = sess.run([model.loss, model.optimize],
                           {model.data: x, model.label: y, model.emb_keep: settings.EMB_KEEP_PROB,
                            model.rnn_keep: settings.RNN_KEEP_PROB})
        if step % settings.SHOW_STEP == 0:
            print 'step {},loss is {}'.format(step, loss)
        if step % settings.SAVE_STEP == 0:
            saver.save(sess, os.path.join(settings.CKPT_PATH, settings.MODEL_NAME), model.global_step)
