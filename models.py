# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:57
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import tensorflow as tf
import functools
import settings

HIDDEN_SIZE = 128
NUM_LAYERS = 2


def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model(object):
    def __init__(self, data, lables, emb_keep, rnn_keep):
        self.data = data
        self.label = lables
        self.emb_keep = emb_keep
        self.rnn_keep = rnn_keep
        self.predict
        self.loss
        self.global_step
        self.ema
        self.optimize
        self.acc

    @define_scope
    def predict(self):
        embedding = tf.get_variable('embedding', [settings.VOCAB_SIZE, HIDDEN_SIZE])

        lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), self.rnn_keep) for _ in
                     range(NUM_LAYERS)]

        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)

        input = tf.nn.embedding_lookup(embedding, self.data)
        dropout_input = tf.nn.dropout(input, self.emb_keep)

        outputs, last_state = tf.nn.dynamic_rnn(cell, dropout_input, dtype=tf.float32)
        last_output = outputs[:, -1, :]

        weights = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, 1]), dtype=tf.float32, name='weights')
        bias = tf.Variable(0, dtype=tf.float32, name='bias')

        logits = tf.matmul(last_output, weights) + bias

        return logits

    @define_scope
    def ema(self):
        ema=tf.train.ExponentialMovingAverage(settings.EMA_RATE,self.global_step)
        return ema


    @define_scope
    def loss(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.predict)
        loss = tf.reduce_mean(loss)
        return loss

    @define_scope
    def global_step(self):
        global_step = tf.Variable(0, trainable=False)
        return global_step

    @define_scope
    def optimize(self):
        learn_rate = tf.train.exponential_decay(settings.LEARN_RATE, self.global_step, settings.LR_DECAY_STEP,
                                                settings.LR_DECAY)
        optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, global_step=self.global_step)
        ave_op=self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([optimizer,ave_op]):
            train_op=tf.no_op('train')
        return train_op

    @define_scope
    def acc(self):
        output = tf.nn.sigmoid(self.predict)
        ok0 = tf.logical_and(tf.less_equal(output, 0.5), tf.equal(self.label, 0))
        ok1 = tf.logical_and(tf.greater(output, 0.5), tf.equal(self.label, 1))
        ok = tf.logical_or(ok0, ok1)
        acc = tf.reduce_mean(tf.cast(ok, dtype=tf.float32))
        return acc


if __name__ == '__main__':
    x = tf.placeholder(tf.int32, [8, 20])
    y = tf.placeholder(tf.float32, [8, 1])
    model = Model(x, y,0.8,0.8)
