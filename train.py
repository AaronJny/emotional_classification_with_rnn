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

# 数据
x = tf.placeholder(tf.int32, [None, None])
# 标签
y = tf.placeholder(tf.float32, [None, 1])
# emb层的dropout保留率
emb_keep = tf.placeholder(tf.float32)
# rnn层的dropout保留率
rnn_keep = tf.placeholder(tf.float32)

# 创建一个模型
model = models.Model(x, y, emb_keep, rnn_keep)

# 创建数据集对象
data = dataset.Dataset(0)

saver = tf.train.Saver()

with tf.Session() as sess:
    # 全局初始化
    sess.run(tf.global_variables_initializer())
    # 迭代训练
    for step in range(settings.TRAIN_TIMES):
        # 获取一个batch进行训练
        x, y = data.next_batch(BATCH_SIZE)
        loss, _ = sess.run([model.loss, model.optimize],
                           {model.data: x, model.label: y, model.emb_keep: settings.EMB_KEEP_PROB,
                            model.rnn_keep: settings.RNN_KEEP_PROB})
        # 输出loss
        if step % settings.SHOW_STEP == 0:
            print 'step {},loss is {}'.format(step, loss)
        # 保存模型
        if step % settings.SAVE_STEP == 0:
            saver.save(sess, os.path.join(settings.CKPT_PATH, settings.MODEL_NAME), model.global_step)
