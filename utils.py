# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:44
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import settings


def read_vocab_list():
    """
    读取词汇表
    :return:由词汇表中所有单词组成的列表
    """
    with open(settings.VOCAB_PATH, 'r') as f:
        vocab_list = f.read().strip().split('\n')
    return vocab_list


def read_word_to_id_dict():
    """
    生成一个单词到编号的映射
    :return:单词到编号的字典
    """
    vocab_list = read_vocab_list()
    word2id = dict(zip(vocab_list, range(len(vocab_list))))
    return word2id


def read_id_to_word_dict():
    """
    生成一个编号到单词的映射
    :return:编号到单词的字典
    """
    vocab_list = read_vocab_list()
    id2word = dict(zip(range(len(vocab_list)), vocab_list))
    return id2word


def get_id_by_word(word, word2id):
    """
    给定一个单词和字典，获得单词在字典中的编号
    :param word: 给定单词
    :param word2id: 单词到编号的映射
    :return: 若单词在字典中，返回对应的编号 否则，返回word2id['<unkown>']
    """
    if word in word2id:
        return word2id[word]
    else:
        return word2id['<unkown>']
