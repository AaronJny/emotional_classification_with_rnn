# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午5:09
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import settings
import tensorflow as tf
import models
import dataset
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''

BATCH_SIZE = settings.BATCH_SIZE

x = tf.placeholder(tf.int32, [None, None])
y = tf.placeholder(tf.float32, [None, 1])
emb_keep = tf.placeholder(tf.float32)
rnn_keep = tf.placeholder(tf.float32)

model = models.Model(x, y, emb_keep, rnn_keep)


data = dataset.Dataset(0)

restore_variables=model.ema.variables_to_restore()

saver = tf.train.Saver(restore_variables)

with tf.Session() as sess:
    while True:
        ckpt = tf.train.get_checkpoint_state(settings.CKPT_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
        acc = sess.run([model.acc],
                       {model.data: data.data, model.label: data.labels, model.emb_keep: 1.0, model.rnn_keep: 1.0})
        print 'acc is ', acc
        time.sleep(1)
