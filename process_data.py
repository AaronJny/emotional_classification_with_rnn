# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:28
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import collections
import settings
import utils
import numpy as np


def create_vocab():
    """
    创建词汇表，写入文件中
    :return:
    """
    # 存放出现的所有单词
    word_list = []
    # 从文件中读取数据，拆分单词
    with open(settings.NEG_TXT, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            word_list.extend(words)
    with open(settings.POS_TXT, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            word_list.extend(words)
    # 统计单词出现的次数
    counter = collections.Counter(word_list)

    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # 选取高频词
    word_list = [word[0] for word in sorted_words]

    word_list = ['<unkown>'] + word_list[:settings.VOCAB_SIZE - 1]
    # 将词汇表写入文件中
    with open(settings.VOCAB_PATH, 'w') as f:
        for word in word_list:
            f.write(word + '\n')


def create_vec(txt_path, vec_path):
    """
    根据词汇表生成词向量
    :param txt_path: 影评文件路径
    :param vec_path: 输出词向量路径
    :return:
    """
    # 获取单词到编号的映射
    word2id = utils.read_word_to_id_dict()
    # 将语句转化成向量
    vec = []
    with open(txt_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            tmp_vec = [str(utils.get_id_by_word(word, word2id)) for word in line.strip().split()]
            vec.append(tmp_vec)
    # 写入文件中
    with open(vec_path, 'w') as f:
        for tmp_vec in vec:
            f.write(' '.join(tmp_vec) + '\n')


def cut_train_dev_test():
    """
    使用轮盘赌法，划分训练集、开发集和测试集
    打乱，并写入不同文件中
    :return:
    """
    # 三个位置分别存放训练、开发、测试
    data = [[], [], []]
    labels = [[], [], []]
    # 累加概率 rate [0.8,0.1,0.1]  cumsum_rate [0.8,0.9,1.0]
    rate = np.array([settings.TRAIN_RATE, settings.DEV_RATE, settings.TEST_RATE])
    cumsum_rate = np.cumsum(rate)
    # 使用轮盘赌法划分数据集
    with open(settings.POS_VEC, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            tmp_data = [int(word) for word in line.strip().split()]
            tmp_label = [1, ]
            index = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
            data[index].append(tmp_data)
            labels[index].append(tmp_label)
    with open(settings.NEG_VEC, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            tmp_data = [int(word) for word in line.strip().split()]
            tmp_label = [0, ]
            index = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
            data[index].append(tmp_data)
            labels[index].append(tmp_label)
    # 计算一下实际上分割出来的比例
    print '最终分割比例', np.array([map(len, data)], dtype=np.float32) / sum(map(len, data))
    # 打乱数据，写入到文件中
    shuffle_data(data[0], labels[0], settings.TRAIN_DATA)
    shuffle_data(data[1], labels[1], settings.DEV_DATA)
    shuffle_data(data[2], labels[2], settings.TEST_DATA)


def shuffle_data(x, y, path):
    """
    填充数据，生成np数组
    打乱数据，写入文件中
    :param x: 数据
    :param y: 标签
    :param path: 保存路径
    :return:
    """
    # 计算影评的最大长度
    maxlen = max(map(len, x))
    # 填充数据
    data = np.zeros([len(x), maxlen], dtype=np.int32)
    for row in range(len(x)):
        data[row, :len(x[row])] = x[row]
    label = np.array(y)
    # 打乱数据
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)
    # 保存数据
    np.save(path + '_data', data)
    np.save(path + '_labels', label)


def decode_file(infile, outfile):
    """
    将文件的编码从'Windows-1252'转为Unicode
    :param infile: 输入文件路径
    :param outfile: 输出文件路径
    :return:
    """
    with open(infile, 'r') as f:
        txt = f.read().decode('Windows-1252')
    with open(outfile, 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    # 解码文件
    decode_file(settings.ORIGIN_POS, settings.POS_TXT)
    decode_file(settings.ORIGIN_NEG, settings.NEG_TXT)
    # 创建词汇表
    create_vocab()
    # 生成词向量
    create_vec(settings.NEG_TXT, settings.NEG_VEC)
    create_vec(settings.POS_TXT, settings.POS_VEC)
    # 划分数据集
    cut_train_dev_test()
