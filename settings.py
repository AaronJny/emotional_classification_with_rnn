# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:44
# @Author  : AaronJny
# @Email   : Aaron__7@163.com

# 源数据路径
ORIGIN_NEG = 'data/rt-polarity.neg'

ORIGIN_POS = 'data/rt-polarity.pos'
# 转码后的数据路径
NEG_TXT = 'data/neg.txt'

POS_TXT = 'data/pos.txt'
# 词汇表路径
VOCAB_PATH = 'data/vocab.txt'
# 词向量路径
NEG_VEC = 'data/neg.vec'

POS_VEC = 'data/pos.vec'
# 训练集路径
TRAIN_DATA = 'data/train'
# 开发集路径
DEV_DATA = 'data/dev'
# 测试集路径
TEST_DATA = 'data/test'
# 模型保存路径
CKPT_PATH = 'ckpt'
# 模型名称
MODEL_NAME = 'model'
# 词汇表大小
VOCAB_SIZE = 10000
# 初始学习率
LEARN_RATE = 0.0001
# 学习率衰减
LR_DECAY = 0.99
# 衰减频率
LR_DECAY_STEP = 1000
# 总训练次数
TRAIN_TIMES = 2000
# 显示训练loss的频率
SHOW_STEP = 10
# 保存训练模型的频率
SAVE_STEP = 100
# 训练集占比
TRAIN_RATE = 0.8
# 开发集占比
DEV_RATE = 0.1
# 测试集占比
TEST_RATE = 0.1
# BATCH大小
BATCH_SIZE = 64
# emb层dropout保留率
EMB_KEEP_PROB = 0.5
# rnn层dropout保留率
RNN_KEEP_PROB = 0.5
# 移动平均衰减率
EMA_RATE = 0.99
