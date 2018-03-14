# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:44
# @Author  : AaronJny
# @Email   : Aaron__7@163.com


ORIGIN_NEG='data/rt-polarity.neg'

ORIGIN_POS='data/rt-polarity.pos'

NEG_TXT = 'data/neg.txt'

POS_TXT = 'data/pos.txt'

VOCAB_PATH = 'data/vocab.txt'

NEG_VEC = 'data/neg.vec'

POS_VEC = 'data/pos.vec'

TRAIN_DATA='data/train'

DEV_DATA='data/dev'

TEST_DATA='data/test'

CKPT_PATH='ckpt'

MODEL_NAME='model'

VOCAB_SIZE = 10000

LEARN_RATE=0.0001

LR_DECAY=0.99

LR_DECAY_STEP=1000

TRAIN_TIMES=2000

SHOW_STEP=10

SAVE_STEP=100

TRAIN_RATE=0.8

DEV_RATE=0.1

TEST_RATE=0.1

BATCH_SIZE=64

EMB_KEEP_PROB=0.5

RNN_KEEP_PROB=0.5

EMA_RATE=0.99
