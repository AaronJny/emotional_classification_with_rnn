# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:44
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import settings


def read_vocab_list():
    with open(settings.VOCAB_PATH, 'r') as f:
        vocab_list = f.read().strip().split('\n')
    return vocab_list


def read_word_to_id_dict():
    vocab_list = read_vocab_list()
    word2id = dict(zip(vocab_list, range(len(vocab_list))))
    return word2id


def read_id_to_word_dict():
    vocab_list = read_vocab_list()
    id2word = dict(zip(range(len(vocab_list)), vocab_list))
    return id2word


def get_id_by_word(word, word2id):
    if word in word2id:
        return word2id[word]
    else:
        return word2id['<unkown>']
