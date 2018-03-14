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
    word_list = []
    with open(settings.NEG_TXT, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            if len(line)>100:
                continue
            words = line.strip().split()
            word_list.extend(words)
    with open(settings.POS_TXT, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            word_list.extend(words)

    counter = collections.Counter(word_list)

    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    word_list = [word[0] for word in sorted_words]

    word_list = ['<unkown>'] + word_list[:9999]

    with open(settings.VOCAB_PATH, 'w') as f:
        for word in word_list:
            f.write(word + '\n')


def create_vec(txt_path, vec_path):
    word2id = utils.read_word_to_id_dict()

    vec = []
    with open(txt_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            tmp_vec = [str(utils.get_id_by_word(word, word2id)) for word in line.strip().split()]
            vec.append(tmp_vec)

    with open(vec_path, 'w') as f:
        for tmp_vec in vec:
            f.write(' '.join(tmp_vec) + '\n')


def cut_train_dev_test():
    data = [[], [], []]
    labels = [[], [], []]
    rate = np.array([settings.TRAIN_RATE, settings.DEV_RATE, settings.TEST_RATE])
    cumsum_rate = np.cumsum(rate)
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

    print '最终分割比例',np.array([map(len, data)], dtype=np.float32) / sum(map(len, data))

    shuffle_data(data[0],labels[0],settings.TRAIN_DATA)
    shuffle_data(data[1],labels[1],settings.DEV_DATA)
    shuffle_data(data[2],labels[2],settings.TEST_DATA)


def shuffle_data(x,y,path):
    maxlen=max(map(len,x))
    data=np.zeros([len(x),maxlen],dtype=np.int32)
    for row in range(len(x)):
        data[row,:len(x[row])]=x[row]
    label=np.array(y)

    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)
    np.save(path+'_data', data)
    np.save(path+'_labels',label)


def decode_file(infile,outfile):
    with open(infile,'r') as f:
        txt=f.read().decode('Windows-1252')
    with open(outfile,'w') as f:
        f.write(txt)


if __name__ == '__main__':
    decode_file(settings.ORIGIN_POS,settings.POS_TXT)
    decode_file(settings.ORIGIN_NEG,settings.NEG_TXT)
    create_vocab()
    create_vec(settings.NEG_TXT,settings.NEG_VEC)
    create_vec(settings.POS_TXT,settings.POS_VEC)
    cut_train_dev_test()
