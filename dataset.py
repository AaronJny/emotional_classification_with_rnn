# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午3:33
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import numpy as np
import settings


class Dataset(object):
    def __init__(self, data_kind=0):
        self.data, self.labels = self.read_data(data_kind)
        self.start = 0
        self.data_size = len(self.data)

    def read_data(self, data_kind):
        data_path = [settings.TRAIN_DATA, settings.DEV_DATA, settings.TEST_DATA][data_kind]
        data = np.load(data_path + '_data.npy')
        labels = np.load(data_path + '_labels.npy')
        return data, labels

    def next_batch(self, batch_size):
        start = self.start
        end = min(start + batch_size, self.data_size)
        self.start = end
        if self.start >= self.data_size:
            self.start = 0
        return self.data[start:end], self.labels[start:end]


if __name__ == '__main__':
    Dataset()
